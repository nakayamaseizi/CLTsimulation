"""
pareto_optimizer.py
===================
有孔ラミナ保護層のパレート最適化。

【設計変数】
  d_mm    : 孔径 [mm]（同一試験体内は一定）
  N_holes : 孔数（試験体面積 W×H 内の総孔数）
  W_mm    : 試験体幅 [mm]
  H_mm    : 試験体高さ [mm]
  t_lam   : ラミナ厚さ [mm]（12mm or 24mm）
  n_lam   : ラミナ枚数（1〜8、総厚 ≤ 96mm）

【導出量】
  vf     : 開口率 = N × π × d² / (4 × W × H)
  p_eff  : 等価ピッチ = sqrt(W×H/N) [mm]（3D unit cell サイズ）
  total  : 保護層総厚 = t_lam * n_lam [mm]

【目的関数（2目的）】
  F1 : 60分後CLT面温度 [°C]  → 最小化（耐火性能↑）
  F2 : 断熱抵抗 R [m²·K/W]  → 最大化（断熱性能↑）

【物性モデルの選択】
  d≤18mm かつ t_lam≤30mm : 池畑(2021)実験式（PerforatedWoodAdvanced）
  それ以外                : 並列混合則（PerforatedWoodProperties）

【アルゴリズム】
  1. (d, N, t_lam, n_lam) の全有効組み合わせを生成
  2. (vf, total_mm) が同一の組み合わせは1回だけシミュレーション（重複除去）
  3. 各点で F1（シミュレーション）・F2（解析式）を評価
  4. パレートフロント（非支配解集合）を特定
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

_AIR_K: float = 0.026     # 空気の熱伝導率 [W/m·K]
_WOOD_K_RT: float = 0.12  # Eurocode 5 スギ 室温熱伝導率 [W/m·K]
_CLT_LAYER_MM: float = 30.0   # CLT 各層厚さ [mm]
_CLT_N_LAYERS: int = 3        # CLT 層数（3層90mm スギ 標準）
_CLT_RHO: float = 410.0       # CLT スギ密度 [kg/m³]
_RSI: float = 0.13  # 室内側表面熱抵抗 [m²K/W]（ISO 6946 / JIS A 2102 水平熱流）
_RSE: float = 0.04  # 加熱面側表面熱抵抗 [m²K/W]


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class ParetoCandidate:
    """1つの設計変数の組み合わせ"""
    d_mm: float
    N_holes: int                     # 孔数
    W_mm: float                      # 試験体幅 [mm]
    H_mm: float                      # 試験体高さ [mm]
    t_lam_mm: float
    n_lam: int
    t_face_mm: float = 0.0           # 表側（火側）無孔パネル厚 [mm]
    face_mat: str = "sugi"           # 表側パネル材料（"sugi", "funen_ki" など）

    # 導出量（後から設定）
    vf: float = field(init=False)
    total_mm: float = field(init=False)   # 有孔ラミナ層の総厚 [mm]
    T_clt_60: float = float("inf")   # 60分後CLT面温度 [°C]
    R_value: float = 0.0             # 断熱抵抗 [m²·K/W]
    is_pareto: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        self.vf = _compute_vf(self.d_mm, self.N_holes, self.W_mm, self.H_mm)
        self.total_mm = self.t_lam_mm * self.n_lam

    @property
    def p_eff_mm(self) -> float:
        """等価ピッチ（3D unit cell サイズ）= sqrt(W×H/N) [mm]"""
        if self.N_holes <= 0 or self.d_mm <= 0:
            return max(self.W_mm, self.H_mm)
        return float(np.sqrt(self.W_mm * self.H_mm / self.N_holes))

    @property
    def r_key(self) -> tuple:
        """1Dシミュレーション結果の共有キー（vf・総厚・表面パネル厚・材料が同じなら同一結果）"""
        return (round(self.vf, 4), float(self.total_mm), float(self.t_face_mm), self.face_mat)

    @property
    def protection_thickness_m(self) -> float:
        """有孔ラミナ層の厚さ [m]"""
        return self.total_mm / 1000.0

    @property
    def total_protection_mm(self) -> float:
        """表面パネル + 有孔ラミナ層の合計厚 [mm]"""
        return self.t_face_mm + self.total_mm

    @property
    def k_eff_rt(self) -> float:
        """有孔ラミナ層の室温換算有効熱伝導率 [W/m·K]"""
        return (1.0 - self.vf) * _WOOD_K_RT + self.vf * _AIR_K

    @property
    def r_analytical(self) -> float:
        """室温換算解析断熱抵抗（表面パネル + 有孔層）[m²·K/W]"""
        # 表面パネル（無孔）
        r_face = (self.t_face_mm / 1000.0) / _WOOD_K_RT if self.t_face_mm > 0 else 0.0
        # 有孔ラミナ層
        k = self.k_eff_rt
        r_perf = (self.total_mm / 1000.0) / k if k > 0 else 0.0
        return r_face + r_perf

    @property
    def lambda_eff(self) -> float:
        """有孔ラミナ層の室温等価熱伝導率 λ_eff [W/m·K]（並列混合則）"""
        return self.k_eff_rt

    @property
    def U_value(self) -> float:
        """全体熱貫流率 U [W/(m²·K)]（室温・稳常状態）

        構成: Rse(0.04) | 表面パネル | 有孔ラミナ | CLT 3×30mm | Rsi(0.13)
        """
        r_clt = (_CLT_N_LAYERS * _CLT_LAYER_MM / 1000.0) / _WOOD_K_RT
        r_total = _RSE + self.r_analytical + r_clt + _RSI
        return 1.0 / r_total if r_total > 0 else 0.0


@dataclass
class ParetoOptState:
    """最適化実行状態"""
    status: str = "idle"     # idle / running / done / stopped / error
    total: int = 0
    done: int = 0
    candidates: list[ParetoCandidate] = field(default_factory=list)
    pareto_front: list[ParetoCandidate] = field(default_factory=list)
    error_msg: str = ""
    elapsed_s: float = 0.0
    solver_mode: str = "1D"  # "1D" | "3D"

    @property
    def progress(self) -> float:
        return self.done / max(self.total, 1)

    @property
    def est_remaining_s(self) -> float:
        if self.done == 0 or self.elapsed_s == 0:
            return 0.0
        rate = self.done / self.elapsed_s
        return (self.total - self.done) / max(rate, 1e-6)


# ---------------------------------------------------------------------------
# モジュールレベルのシングルトン
# ---------------------------------------------------------------------------

_state = ParetoOptState()
_stop_event = threading.Event()
_lock = threading.Lock()


def get_pareto_state() -> ParetoOptState:
    """現在の最適化状態を返す。旧デプロイ版で欠落したフィールドはパッチで補完する。"""
    global _state
    # 旧バージョンのインスタンスに新フィールドが無い場合はパッチ（リセットしない）
    if not hasattr(_state, "solver_mode"):
        _state.solver_mode = "1D"
    if not hasattr(_state, "error_msg"):
        _state.error_msg = ""
    return _state


def stop_pareto() -> None:
    """実行中の最適化に中断シグナルを送る。"""
    _stop_event.set()


# ---------------------------------------------------------------------------
# 公開 API：最適化開始
# ---------------------------------------------------------------------------

def start_pareto_optimization(
    d_list: list[float],
    N_list: list[int],
    W_mm: float,
    H_mm: float,
    t_lam_list: list[float],
    n_lam_max: int = 8,
    base_mat: str = "sugi",
    base_rho: float = 400.0,
    n_cells: int = 10,
    t_end_min: float = 60.0,
    solver_mode: str = "1D",
    n_cells_xy: int = 6,
    t_face_list: list[float] | None = None,
    face_mat: str = "sugi",
) -> None:
    """パレート最適化をバックグラウンドで開始する。

    Parameters
    ----------
    d_list : list[float]
        孔径の候補リスト [mm]（0 = 無孔）
    N_list : list[int]
        孔数の候補リスト
    W_mm : float
        試験体幅 [mm]
    H_mm : float
        試験体高さ [mm]
    t_lam_list : list[float]
        ラミナ厚さの候補リスト [mm]（例: [12.0, 24.0]）
    n_lam_max : int
        ラミナ枚数の最大値（上限: 8）
    base_mat : str
        保護層の材料キー（デフォルト: "sugi"）
    base_rho : float
        保護層の密度 [kg/m³]
    n_cells : int
        1D FVM のセル数/層
    t_end_min : float
        加熱時間 [分]
    solver_mode : str
        "1D"（高速・池畑実験式）または "3D"（精密・燃え抜け考慮）
    n_cells_xy : int
        3Dモード用 YZ 方向のセル数（孔の形状精度に影響）
    t_face_list : list[float] or None
        表側（火側）無孔パネル厚の候補リスト [mm]。0 = 無し。None の場合は [0.0]。
    """
    global _state
    if t_face_list is None:
        t_face_list = [0.0]

    with _lock:
        if _state.status == "running":
            return

        candidates = _generate_candidates(
            d_list, N_list, W_mm, H_mm, t_lam_list, n_lam_max, t_face_list, face_mat
        )
        _stop_event.clear()
        _state = ParetoOptState(
            status="running",
            total=len(candidates),
            candidates=candidates,
            solver_mode=solver_mode,
        )

    t = threading.Thread(
        target=_run_pareto_thread,
        args=(candidates, base_mat, base_rho, n_cells, t_end_min, solver_mode, n_cells_xy),
        daemon=True,
        name="pareto-opt-worker",
    )
    t.start()


# ---------------------------------------------------------------------------
# 設計空間の生成
# ---------------------------------------------------------------------------

def _compute_vf(d_mm: float, N: int, W_mm: float, H_mm: float) -> float:
    """孔径 d、孔数 N、試験体面積 W×H に基づく開口率を計算する。

    開口率[%] = N × π × d² / (4 × W × H) × 100
    戻り値は割合（0〜1）。
    """
    if d_mm <= 0 or N <= 0 or W_mm <= 0 or H_mm <= 0:
        return 0.0
    return (N * np.pi * d_mm ** 2) / (4.0 * W_mm * H_mm)


def _generate_candidates(
    d_list: list[float],
    N_list: list[int],
    W_mm: float,
    H_mm: float,
    t_lam_list: list[float],
    n_lam_max: int,
    t_face_list: list[float] | None = None,
    face_mat: str = "sugi",
) -> list[ParetoCandidate]:
    """有効な全候補を生成する。

    Notes
    -----
    - d=0（無孔）の場合、孔数に依らず同一結果になるため
      各 (t_lam, n_lam, t_face) につき代表1点（N=0）だけ生成する。
    - 同じ (vf, total_mm, t_face_mm) を与える複数の (d, N) 組み合わせは
      シミュレーションキャッシュで重複除去するが、候補としてはそれぞれ残す
      （異なる物理設計として別の行に表示するため）。
    - 孔径 d が等価ピッチ p_eff = sqrt(W×H/N) 以上になる組み合わせは除外。
    - t_face_list: 表側（火側）無孔パネル厚候補 [mm]。0 = 無し。
    """
    if t_face_list is None:
        t_face_list = [0.0]

    max_total_mm = 96.0  # 有孔ラミナ層の最大総厚（表面パネルは含まない）
    candidates: list[ParetoCandidate] = []
    solid_keys_seen: set = set()

    for t_face in sorted(t_face_list):
        for t_lam in sorted(t_lam_list):
            n_max = min(n_lam_max, int(max_total_mm / t_lam))
            for n_lam in range(1, n_max + 1):
                # 無孔候補：(t_lam, n_lam, t_face) ごとに1点だけ
                solid_key = (t_lam, n_lam, t_face)
                if solid_key not in solid_keys_seen and 0.0 in d_list:
                    solid_keys_seen.add(solid_key)
                    candidates.append(ParetoCandidate(
                        d_mm=0.0, N_holes=0, W_mm=W_mm, H_mm=H_mm,
                        t_lam_mm=t_lam, n_lam=n_lam,
                        t_face_mm=t_face, face_mat=face_mat,
                    ))

                for N in sorted(N_list):
                    for d in sorted(d_list):
                        if d <= 0 or N <= 0:
                            continue
                        # 孔が等価ピッチより大きければ物理的に不正
                        p_eff = float(np.sqrt(W_mm * H_mm / N))
                        if d >= p_eff:
                            continue
                        if _compute_vf(d, N, W_mm, H_mm) > 0.79:
                            continue

                        candidates.append(ParetoCandidate(
                            d_mm=d, N_holes=N, W_mm=W_mm, H_mm=H_mm,
                            t_lam_mm=t_lam, n_lam=n_lam,
                            t_face_mm=t_face, face_mat=face_mat,
                        ))

    return candidates


# ---------------------------------------------------------------------------
# バックグラウンドスレッド
# ---------------------------------------------------------------------------

def _run_pareto_thread(
    candidates: list[ParetoCandidate],
    base_mat: str,
    base_rho: float,
    n_cells: int,
    t_end_min: float,
    solver_mode: str = "1D",
    n_cells_xy: int = 6,
) -> None:
    """最適化のメインスレッド処理。"""
    wall_start = time.monotonic()
    use_3d = solver_mode == "3D"

    sim_cache: dict = {}

    for c in candidates:
        if _stop_event.is_set():
            break
        try:
            # キャッシュキー（表面パネル材料も含む）
            if use_3d:
                key = (round(c.d_mm, 2), round(c.p_eff_mm, 2),
                       float(c.total_mm), float(c.t_face_mm), c.face_mat)
            else:
                key = c.r_key  # (vf_rounded, total_mm, t_face_mm, face_mat)

            if key not in sim_cache:
                if use_3d:
                    T_clt = _run_sim_one_3d(
                        c, base_mat, base_rho, n_cells_xy, n_cells, t_end_min
                    )
                else:
                    T_clt = _run_sim_one(c, base_mat, base_rho, n_cells, t_end_min)
                sim_cache[key] = T_clt
            c.T_clt_60 = sim_cache[key]
            c.R_value = c.r_analytical

        except Exception as e:
            c.error = str(e)

        with _lock:
            _state.done += 1
            _state.elapsed_s = time.monotonic() - wall_start

    # パレートフロント計算
    valid = [c for c in candidates if c.T_clt_60 < float("inf") and not c.error]
    pareto = _compute_pareto_front(valid)

    with _lock:
        _state.status = "done" if not _stop_event.is_set() else "stopped"
        _state.candidates = candidates
        _state.pareto_front = pareto
        _state.elapsed_s = time.monotonic() - wall_start


def _run_sim_one(
    c: ParetoCandidate,
    base_mat: str,
    base_rho: float,
    n_cells: int,
    t_end_min: float,
) -> float:
    """1ケースのシミュレーションを実行し、60分後CLT面温度を返す。

    物性モデル: 常に池畑(2021)実験式（PerforatedWoodAdvanced）を使用。
    d>18mm や t_lam>30mm は式内でクランプして外挿適用。
    """
    from clt_fire_sim.materials import (
        PerforatedWoodAdvanced,
        make_properties,
    )
    from clt_fire_sim.boundary import ISO834HeatedBC, ConvRadCoolingBC
    from clt_fire_sim.solver.fvm_1d import FVM1DSolver, MultiLayerProperties, make_multi_layer_mesh

    protect_m = c.protection_thickness_m  # 有孔ラミナ層のみ [m]
    face_m = c.t_face_mm / 1000.0         # 表面パネル [m]
    clt_layer_m = _CLT_LAYER_MM / 1000.0

    base_props = make_properties(base_mat, base_rho, 0.12)
    clt_props = make_properties("sugi", _CLT_RHO, 0.12)
    # 表面パネル専用物性（材料が base_mat と異なる場合は個別生成）
    face_props = make_properties(c.face_mat, base_rho, 0.12) if c.t_face_mm > 0 else None

    # 有孔ラミナ層の物性値（常に池畑実験式）
    if c.vf > 0:
        protect_props = PerforatedWoodAdvanced(
            base_props=base_props,
            void_fraction=c.vf,
            hole_diameter_mm=c.d_mm,
            hole_depth_mm=c.t_lam_mm,
            layer_thickness_mm=c.t_lam_mm,
        )
    else:
        protect_props = base_props  # 無孔 → 素材そのまま

    # レイヤー構成（表面パネルあり / なし）
    if face_m > 0 and face_props is not None:
        thicknesses = [face_m, protect_m] + [clt_layer_m] * _CLT_N_LAYERS
        props_list = [face_props, protect_props] + [clt_props] * _CLT_N_LAYERS
    else:
        thicknesses = [protect_m] + [clt_layer_m] * _CLT_N_LAYERS
        props_list = [protect_props] + [clt_props] * _CLT_N_LAYERS

    mesh = make_multi_layer_mesh(thicknesses, n_cells_per_layer=n_cells, ratio=1.05)
    props = MultiLayerProperties(thicknesses, props_list)
    props.setup(mesh.x_centers)

    bc_l = ISO834HeatedBC(alpha_c=25.0, eps_m=0.8, eps_f=1.0)
    bc_r = ConvRadCoolingBC(alpha_c=9.0, eps_m=0.8, T_inf=20.0)
    solver = FVM1DSolver(mesh, props, bc_l, bc_r, T_init=20.0)

    result = solver.solve(
        t_end=int(t_end_min) * 60,
        dt_base=10, dt_min=2, dt_max=30,
        n_picard=2, record_interval=60,
    )

    # CLT面温度（深さ = 表面パネル + 有孔ラミナ層）を抽出
    x_mm = mesh.x_centers * 1000.0
    idx_depth = np.argmin(np.abs(x_mm - c.total_protection_mm))
    idx_t60 = np.argmin(np.abs(result["times"] / 60.0 - t_end_min))
    return float(result["temperatures"][idx_t60, idx_depth])


def _run_sim_one_3d(
    c: ParetoCandidate,
    base_mat: str,
    base_rho: float,
    n_cells_xy: int,
    n_cells_per_layer: int,
    t_end_min: float,
) -> float:
    """3D FVMソルバー（燃え抜けON）で1ケースのシミュレーション。

    ユニットセル（p_mm × p_mm × 全厚）を対象に、保護層に円孔を設け
    burn_through=True で孔内部を ISO 834 曲線温度に固定する。
    CLT面温度はユニットセルの YZ 断面平均値を返す。
    """
    from clt_fire_sim.solver.fvm_3d import (
        FVM3DSolver, make_clt_mesh_3d, setup_multi_layer_props_3d,
    )
    from clt_fire_sim.materials import make_properties
    from clt_fire_sim.boundary import ISO834HeatedBC, ConvRadCoolingBC

    protect_m = c.protection_thickness_m  # 有孔ラミナ層 [m]
    face_m = c.t_face_mm / 1000.0         # 表面パネル [m]
    clt_layer_m = _CLT_LAYER_MM / 1000.0
    p_eff_m = c.p_eff_mm / 1000.0  # 等価ピッチ = sqrt(W×H/N) [m]

    # レイヤー構成（表面パネルあり / なし）
    if face_m > 0:
        layer_thicknesses = [face_m, protect_m] + [clt_layer_m] * _CLT_N_LAYERS
        n_face_x = n_cells_per_layer   # 表面パネルのxセル数
        n_perf_x_start = n_cells_per_layer  # 有孔層開始インデックス
    else:
        layer_thicknesses = [protect_m] + [clt_layer_m] * _CLT_N_LAYERS
        n_face_x = 0
        n_perf_x_start = 0

    # 3Dメッシュ生成（ユニットセル = p_eff × p_eff mm のYZ断面）
    mesh = make_clt_mesh_3d(
        layer_thicknesses=layer_thicknesses,
        specimen_width=p_eff_m,
        specimen_height=p_eff_m,
        n_cells_per_layer=n_cells_per_layer,
        mesh_ratio=1.05,
        n_cells_y=n_cells_xy,
        n_cells_z=n_cells_xy,
    )

    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz

    # void_mask 生成（有孔ラミナ層のみ、中央に円孔）
    void_mask = None
    if c.d_mm > 0.0:
        void_mask = np.zeros((nx, ny, nz), dtype=bool)
        r_sq = (c.d_mm / 2.0) ** 2
        half_p = c.p_eff_mm / 2.0
        perf_start = n_perf_x_start
        perf_end = perf_start + n_cells_per_layer
        for j in range(ny):
            y_c = (j + 0.5) / ny * c.p_eff_mm
            for k in range(nz):
                z_c = (k + 0.5) / nz * c.p_eff_mm
                if (y_c - half_p) ** 2 + (z_c - half_p) ** 2 <= r_sq:
                    void_mask[perf_start:perf_end, j, k] = True

    # 物性値（孔は void_mask で表現。表面パネルは c.face_mat を使用）
    base_props = make_properties(base_mat, base_rho, 0.12)
    clt_props = make_properties("sugi", _CLT_RHO, 0.12)
    if face_m > 0:
        face_props = make_properties(c.face_mat, base_rho, 0.12)
        layer_props = [face_props, base_props] + [clt_props] * _CLT_N_LAYERS
    else:
        layer_props = [base_props] + [clt_props] * _CLT_N_LAYERS
    props = setup_multi_layer_props_3d(layer_thicknesses, layer_props, mesh)

    bc_l = ISO834HeatedBC(alpha_c=25.0, eps_m=0.8, eps_f=1.0)
    bc_r = ConvRadCoolingBC(alpha_c=9.0, eps_m=0.8, T_inf=20.0)
    solver = FVM3DSolver(
        mesh=mesh, props=props,
        bc_left=bc_l, bc_right=bc_r,
        T_init=20.0,
        void_mask=void_mask,
        burn_through=True,
    )

    result = solver.solve(
        t_end=int(t_end_min) * 60,
        dt_base=10, dt_min=2, dt_max=30,
        n_picard=2, record_interval=60,
    )

    # 保護層全体直後（x = face + protect）の断面YZ平均温度を抽出
    total_layers = len(layer_thicknesses)
    n_protect_layers = (1 if face_m > 0 else 0) + 1  # 表面パネル+有孔層
    idx_x = n_protect_layers * n_cells_per_layer  # CLT最初のセル
    idx_x = min(idx_x, nx - 1)
    idx_t = np.argmin(np.abs(result["times"] / 60.0 - t_end_min))
    T_field = result["temperatures"][idx_t].reshape(nx, ny, nz)
    return float(T_field[idx_x, :, :].mean())


# ---------------------------------------------------------------------------
# パレートフロント計算
# ---------------------------------------------------------------------------

def _compute_pareto_front(candidates: list[ParetoCandidate]) -> list[ParetoCandidate]:
    """
    F1（CLT面温度）最小化 × F2（断熱抵抗）最大化 のパレート非支配解を返す。

    候補 q が候補 p を支配する条件:
      q.T_clt_60 ≤ p.T_clt_60  かつ  q.R_value ≥ p.R_value
      かつ少なくとも一方で真に優れる
    """
    # 全体数が多い場合に O(n²) が重いため、ソートで枝刈り
    # F1 昇順ソート → その中で F2 最大を追跡してパレート判定
    sorted_cands = sorted(candidates, key=lambda c: (c.T_clt_60, -c.R_value))

    pareto: list[ParetoCandidate] = []
    best_r = -float("inf")

    for c in sorted_cands:
        # F1 が現在以下で F2 が自分より大きい解がすでに存在するか？
        if c.R_value >= best_r:
            # 支配されていない → パレート解
            c.is_pareto = True
            pareto.append(c)
            best_r = c.R_value

    return pareto


# ---------------------------------------------------------------------------
# ユーティリティ関数
# ---------------------------------------------------------------------------

def get_default_d_list() -> list[float]:
    """デフォルトの孔径候補リスト [mm]。"""
    return [0.0, 6.0, 10.0, 14.0, 18.0, 24.0, 30.0]


def get_default_N_list() -> list[int]:
    """デフォルトの孔数候補リスト。"""
    return [1, 4, 9, 16, 25, 36]


# ===========================================================================
# 5層CLT コア加工モード（core5）
# ===========================================================================
#
# 【試験体構成（固定）】
#   加熱面 → スギ30mm（無加工）| 加工30mm ×3（同一加工・千鳥配置）| スギ30mm → 非加熱面
#
# 【千鳥配置と均質化の関係】
#   2〜4層目の孔・スリットは隣接層と重ならない（貫通経路を作らない）配置とする。
#   これは 1D 均質化モデル（池畑式・ISO空気層）が暗黙に仮定する
#   「空隙が厚み方向の連続経路を作らない」条件そのものであり、
#   1D 探索の妥当性を支える。3D 検証では千鳥を void_mask で陽に再現する。
#
# 【最適化フロー】
#   1. 1D 全列挙（~2秒/ケース）で (T_非加熱面, R) を評価
#   2. パレートフロントを特定
#   3. フロント解（最大 CORE5_N_3D_MAX 点）を千鳥 3D で自動検証
# ---------------------------------------------------------------------------

_CORE5_LAYER_MM: float = 30.0   # 各層厚 [mm]
_CORE5_RHO: float = 400.0       # スギ密度 [kg/m³]
_CORE5_MC: float = 0.12         # 含水率

# F1 の評価深さ [mm]：加熱側から 2 層目の裏面（2層目と3層目の界面）
# 構成 [1:無加工30][2:加工30][3:加工30][4:加工30][5:無加工30] の 30+30=60mm
_CORE5_EVAL_DEPTH_MM: float = 60.0
# 耐火性能の判定基準温度 [°C]（グラフの基準線）
CORE5_TEMP_CRITERION: float = 100.0
CORE5_N_3D_MAX: int = 5         # 3D 自動検証するフロント解の既定数
CORE5_N_X_1D: int = 12          # 1D 探索の層あたり x セル数
CORE5_N_X_3D: int = 5           # 3D 検証の層あたり x セル数

# 【重要】炭化深さは x 方向メッシュに強く依存する（実測: 層あたり
# n=4→44.7mm, n=5→43.3mm, n=12→42.0mm, n=20→41.5mm）。
# 1D 探索(n=12)と 3D 検証(n=5)を直接比べると解像度差が「3D効果」に
# 化けるため、3D 検証時には同一解像度の 1D も併走させて差を取る。

# 3D 1ケースの所要時間の実測（キャリブレーション用）
# スリット n_xy=8, n_x/層=5 → 1600 セルで 46 秒
# 有孔   n_xy=12, n_x/層=5 → 3600 セルで 288 秒
# 疎行列解が超線形に効くため cells² でスケールする近似を用いる
_CORE5_3D_REF_CELLS: float = 1600.0
_CORE5_3D_REF_SEC: float = 46.0


def estimate_core5_3d_seconds(c: "Core5Candidate", n_x_per_layer: int = CORE5_N_X_3D) -> float:
    """core5 の 3D 検証 1 ケースの所要時間 [秒] を見積もる。"""
    cells = _core5_n_xy(c) ** 2 * (5 * n_x_per_layer)
    return _CORE5_3D_REF_SEC * (cells / _CORE5_3D_REF_CELLS) ** 2


@dataclass
class Core5Candidate:
    """5層コア加工CLTの設計候補。

    process_type="hole" : d_mm=孔径, pitch_mm=ピッチp, depth_mm=孔深さH
    process_type="slit" : d_mm=スリット幅W, pitch_mm=ピッチP, depth_mm=スリット深さD
    """
    process_type: str
    d_mm: float
    pitch_mm: float
    depth_mm: float

    vf: float = field(init=False)             # 断面開孔率
    beyond_validated: bool = field(init=False)  # 実験式の裏付け範囲外か
    # F1: 加熱側から2層目裏面（深さ60mm）の温度 @加熱終了 [°C]
    T_iface: float = float("inf")             # 1D・細メッシュ
    T_iface_3d: float = float("nan")          # 3D検証値
    T_iface_1d_matched: float = float("nan")  # 3Dと同一x解像度の1D（差分の基準）
    R_value: float = 0.0                      # F2: 全層の室温断熱抵抗 [m²K/W]
    char_depth: float = float("nan")          # 参考: 炭化深さ(300°C) [mm]
    char_depth_3d: float = float("nan")       # 参考: 3D の炭化深さ [mm]
    T_unexposed: float = float("nan")         # 参考: 非加熱面温度 [°C]
    T_unexposed_3d: float = float("nan")      # 参考: 3D の非加熱面温度 [°C]
    is_pareto: bool = False
    verified_3d: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        if self.process_type == "hole":
            self.vf = float(np.pi * (self.d_mm / 2.0) ** 2 / self.pitch_mm ** 2)
            # 池畑式の適用範囲: Φ3〜18, H6〜30。較正開孔率は約10%のため
            # そこから大きく離れる場合も外挿としてマークする
            self.beyond_validated = (
                not (3.0 <= self.d_mm <= 18.0)
                or not (6.0 <= self.depth_mm <= 30.0)
                or self.vf > 0.25
                or self.vf < 0.03
            )
        else:
            self.vf = float(min(self.d_mm / self.pitch_mm, 0.95))
            # スリットの実測裏付けは D ≤ 12mm（池畑 2021）
            self.beyond_validated = self.depth_mm > 12.0

    @property
    def sim_key(self) -> tuple:
        """1D結果の共有キー（等価物性が同じなら同一結果）。

        有孔: k_eff は (vf, Φ, H) で決まる。
        スリット: W は開孔率経由でのみ効くため (vf, D) で決まる。
        """
        if self.process_type == "hole":
            return ("hole", round(self.vf, 4), float(self.d_mm), float(self.depth_mm))
        return ("slit", round(self.vf, 4), float(self.depth_mm))

    @property
    def sim_key_3d(self) -> tuple:
        """3D結果の共有キー。

        3D は void_mask の形状がそのまま効くため、1D と違って
        幾何パラメータ 3 つすべてを区別する必要がある。
        """
        return (self.process_type, float(self.d_mm),
                float(self.pitch_mm), float(self.depth_mm))

    @property
    def effect_3d(self) -> float:
        """真の3D効果 [°C]（同一x解像度の1Dとの差）。

        T_iface（細メッシュ1D）との単純差はメッシュ解像度差を含むため、
        3D検証と同じ x セル数で走らせた 1D を基準に取る。
        """
        if not self.verified_3d or np.isnan(self.T_iface_1d_matched):
            return float("nan")
        return self.T_iface_3d - self.T_iface_1d_matched

    @property
    def T_iface_eff(self) -> float:
        """表示・判定に使う F1（3D検証済みならその値、なければ1D）。"""
        if self.verified_3d and np.isfinite(self.T_iface_3d):
            return self.T_iface_3d
        return self.T_iface

    @property
    def label(self) -> str:
        if self.process_type == "hole":
            return f"Φ{self.d_mm:g}×p{self.pitch_mm:g}×H{self.depth_mm:g}"
        return f"W{self.d_mm:g}×P{self.pitch_mm:g}×D{self.depth_mm:g}"


@dataclass
class Core5OptState:
    """core5 最適化の実行状態。"""
    status: str = "idle"     # idle / running / done / stopped / error
    phase: str = ""          # "1d" / "3d" / ""
    total: int = 0
    done: int = 0
    total_3d: int = 0
    done_3d: int = 0
    candidates: list[Core5Candidate] = field(default_factory=list)
    pareto_front: list[Core5Candidate] = field(default_factory=list)
    elapsed_s: float = 0.0
    error_msg: str = ""
    process_type: str = "hole"
    t_end_min: float = 60.0
    mode_3d: str = "front"                       # "front" | "all" | "none"
    front_is_3d: bool = False                    # フロントを3D値で判定したか
    reference: "Core5Reference | None" = None    # 無加工5層CLT（1D細メッシュ）
    reference_3d: "Core5Reference | None" = None  # 3Dと同一x解像度の基準

    @property
    def progress(self) -> float:
        tot = self.total + self.total_3d
        return (self.done + self.done_3d) / max(tot, 1)


_core5_state = Core5OptState()


def get_core5_state() -> Core5OptState:
    return _core5_state


def generate_core5_candidates(
    process_type: str,
    v1_list: list[float],
    v2_list: list[float],
    v3_list: list[float],
) -> list[Core5Candidate]:
    """有効な候補を全列挙する。

    v1 = 孔径d / スリット幅W、v2 = ピッチ、v3 = 深さ。
    除外条件:
      - v1 ≥ v2（孔・スリットが隣と接触）
      - 有孔で開孔率 > 79%
      - 有孔で Φ≥18 かつ H≥24（池畑 2021 が対流発生により除外した領域）
    """
    cands: list[Core5Candidate] = []
    for v1 in sorted(set(float(v) for v in v1_list)):
        for v2 in sorted(set(float(v) for v in v2_list)):
            for v3 in sorted(set(float(v) for v in v3_list)):
                if v1 <= 0 or v2 <= 0 or v3 <= 0:
                    continue
                if v1 >= v2:
                    continue
                if v3 > _CORE5_LAYER_MM:
                    continue
                if process_type == "hole":
                    if v1 >= 18.0 and v3 >= 24.0:
                        continue  # 池畑実験で対流により除外された領域
                    if np.pi * (v1 / 2.0) ** 2 / v2 ** 2 > 0.79:
                        continue
                cands.append(Core5Candidate(process_type, v1, v2, v3))
    return cands


def _build_core5_proc_props(c: Core5Candidate):
    """加工層（2〜4層目）の等価物性値オブジェクトを返す。"""
    from clt_fire_sim.materials import (
        PerforatedWoodAdvanced, SlittedWoodProperties, make_properties,
    )
    base = make_properties("sugi", _CORE5_RHO, _CORE5_MC)
    if c.process_type == "hole":
        proc = PerforatedWoodAdvanced(
            base_props=base, void_fraction=c.vf,
            hole_diameter_mm=c.d_mm, hole_depth_mm=c.depth_mm,
            layer_thickness_mm=_CORE5_LAYER_MM,
        )
    else:
        proc = SlittedWoodProperties(
            base_props=base,
            slit_width_mm=c.d_mm, slit_depth_mm=c.depth_mm,
            slit_pitch_mm=c.pitch_mm, layer_thickness_mm=_CORE5_LAYER_MM,
        )
    return base, proc


def _temp_at_depth(
    x_centers_m: np.ndarray, T_row: np.ndarray, depth_mm: float,
) -> float:
    """指定深さにおける温度 [°C] を線形補間で求める。

    セル中心の温度分布を線形補間するため、メッシュ解像度が変わっても
    同じ物理位置の温度を比較できる（炭化深さのようなメッシュ敏感性がない）。
    """
    x_mm = np.asarray(x_centers_m, dtype=float) * 1000.0
    return float(np.interp(float(depth_mm), x_mm, np.asarray(T_row, dtype=float)))


def _char_depth_mm(x_centers_m: np.ndarray, T_row: np.ndarray) -> float:
    """300°C 等温線の深さ [mm] を線形補間で求める。

    加熱面側から見て最初に 300°C を下回る位置を炭化フロントとする。
    """
    x_mm = x_centers_m * 1000.0
    hot = T_row >= 300.0
    if not hot.any():
        return 0.0
    i = int(np.argmax(~hot)) if (~hot).any() else len(T_row)
    if i == 0:
        return 0.0
    if i >= len(T_row):
        return float(x_mm[-1])
    T0, T1 = T_row[i - 1], T_row[i]
    x0, x1 = x_mm[i - 1], x_mm[i]
    if abs(T0 - T1) < 1e-9:
        return float(x0)
    return float(x0 + (T0 - 300.0) / (T0 - T1) * (x1 - x0))


def _solve_core5_1d(
    props_list: list,
    t_end_min: float,
    n_cells: int,
) -> tuple[float, float, float]:
    """5層構成の1Dソルバーを走らせ、評価量を返す。

    Returns
    -------
    (T_iface, char_depth_mm, T_unexposed)
        2層目裏面(深さ60mm)温度 [°C]、炭化深さ [mm]、非加熱面温度 [°C]。
    """
    from clt_fire_sim.boundary import ISO834HeatedBC, ConvRadCoolingBC
    from clt_fire_sim.solver.fvm_1d import (
        FVM1DSolver, MultiLayerProperties, make_multi_layer_mesh,
    )

    t_m = _CORE5_LAYER_MM / 1000.0
    thicknesses = [t_m] * 5

    mesh = make_multi_layer_mesh(thicknesses, n_cells_per_layer=n_cells, ratio=1.05)
    props = MultiLayerProperties(thicknesses, props_list)
    props.setup(mesh.x_centers)

    bc_l = ISO834HeatedBC(alpha_c=25.0, eps_m=0.8, eps_f=1.0)
    bc_r = ConvRadCoolingBC(alpha_c=9.0, eps_m=0.8, T_inf=20.0)
    solver = FVM1DSolver(mesh, props, bc_l, bc_r, T_init=20.0)
    result = solver.solve(
        t_end=int(t_end_min) * 60,
        dt_base=10, dt_min=2, dt_max=30,
        n_picard=2, record_interval=60,
    )

    temps = result["temperatures"]
    idx_t = int(np.argmin(np.abs(result["times"] / 60.0 - t_end_min)))
    row = temps[idx_t]
    return (
        _temp_at_depth(mesh.x_centers, row, _CORE5_EVAL_DEPTH_MM),
        _char_depth_mm(mesh.x_centers, row),
        float(row[-1]),
    )


def _run_sim_core5_1d(
    c: Core5Candidate,
    t_end_min: float,
    n_cells: int = 12,
) -> tuple[float, float, float, float]:
    """1ケースの1Dシミュレーション。

    Returns
    -------
    (T_iface, char_depth_mm, T_unexposed, R_value)
        2層目裏面温度 [°C]、炭化深さ [mm]、非加熱面温度 [°C]、
        室温断熱抵抗 [m²K/W]。
    """
    base, proc = _build_core5_proc_props(c)
    T_iface, char_mm, T_unexposed = _solve_core5_1d(
        [base, proc, proc, proc, base], t_end_min, n_cells,
    )

    # 室温断熱抵抗（表面熱抵抗は含まない）
    t_m = _CORE5_LAYER_MM / 1000.0
    k20 = np.array([20.0])
    k_proc = float(proc.get_k_array(k20)[0])
    k_base = float(base.get_k_array(k20)[0])
    R = 2.0 * t_m / k_base + 3.0 * t_m / k_proc
    return T_iface, char_mm, T_unexposed, R


@dataclass
class Core5Reference:
    """比較基準：無加工の 5 層 5 プライ CLT（スギ 30mm×5 = 150mm）。"""
    T_iface: float = float("nan")      # 2層目裏面温度 [°C]
    char_depth: float = float("nan")   # 炭化深さ [mm]
    T_unexposed: float = float("nan")  # 非加熱面温度 [°C]
    R_value: float = float("nan")      # 室温断熱抵抗 [m²K/W]
    n_cells: int = 0                   # 層あたり x セル数（比較の公平性用）


def compute_core5_reference(
    t_end_min: float = 60.0, n_cells: int = CORE5_N_X_1D,
) -> Core5Reference:
    """無加工5層CLTの基準値を計算する。

    加工なしなので空隙がなく、1D と 3D は物理的に等価。
    ただしメッシュ解像度の影響を除くため n_cells を明示的に受け取る。
    """
    from clt_fire_sim.materials import make_properties

    base = make_properties("sugi", _CORE5_RHO, _CORE5_MC)
    T_iface, char_mm, T_unexposed = _solve_core5_1d(
        [base] * 5, t_end_min, n_cells,
    )
    t_m = _CORE5_LAYER_MM / 1000.0
    k_base = float(base.get_k_array(np.array([20.0]))[0])
    return Core5Reference(
        T_iface=T_iface, char_depth=char_mm, T_unexposed=T_unexposed,
        R_value=5.0 * t_m / k_base, n_cells=int(n_cells),
    )


def _build_core5_void_mask_3d(
    c: Core5Candidate, mesh,
) -> np.ndarray:
    """千鳥配置の void_mask を生成する。

    ユニットセルは p×p（スリットは P×P）。YZ 境界は断熱＝対称面なので、
    セル中心の孔も四隅の孔（1/4×4）も同じ面積密度の完全な孔として機能する。

    有孔:  2・4層目は孔中心を (p/2, p/2) 格子（セル中心に1孔）、
           3層目は半ピッチずらして (0, 0) 格子（四隅に1/4孔×4）。
    スリット: 2・4層目は z 方向の帯（y にピッチ P）、3層目は 90° 回転して
           y 方向の帯（直交積層 CLT と整合する千鳥）。
    孔・スリットは各ラミナの加熱面側の面から深さ depth_mm まで。
    """
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    x = mesh.x_centers  # [m]
    t_m = _CORE5_LAYER_MM / 1000.0
    Ly = float(np.sum(mesh.dy))
    Lz = float(np.sum(mesh.dz))
    y_c = (np.cumsum(mesh.dy) - mesh.dy / 2.0)
    z_c = (np.cumsum(mesh.dz) - mesh.dz / 2.0)

    void = np.zeros((nx, ny, nz), dtype=bool)
    depth_m = c.depth_mm / 1000.0
    starts = [t_m, 2 * t_m, 3 * t_m]  # 2〜4層目の開始位置

    for li, start in enumerate(starts):
        x_in = (x >= start) & (x < start + depth_m)
        if not x_in.any():
            continue
        if c.process_type == "hole":
            p_m = c.pitch_mm / 1000.0
            r2 = (c.d_mm / 2000.0) ** 2
            # 孔中心の格子: 2・4層目は (p/2, p/2) 系、3層目は半ピッチずらして (0, 0) 系。
            # どちらも 2p×2p のユニットセル内に 4 孔ぶん含まれ、面積密度は等しい。
            off = p_m / 2.0 if li != 1 else 0.0
            u_y = np.mod(y_c - off, p_m)
            dy = np.minimum(u_y, p_m - u_y)
            u_z = np.mod(z_c - off, p_m)
            dz = np.minimum(u_z, p_m - u_z)
            in_hole = (dy[:, None] ** 2 + dz[None, :] ** 2) <= r2  # (ny, nz)
            void[np.ix_(x_in, np.arange(ny), np.arange(nz))] |= in_hole[None, :, :]
        else:
            P_m = c.pitch_mm / 1000.0
            W_m = c.d_mm / 1000.0
            if li != 1:
                # 2・4層目: z 方向の帯（y にピッチ）
                u = np.mod(y_c, P_m)
                band = np.abs(u - P_m / 2.0) <= W_m / 2.0  # (ny,)
                in_slit = np.broadcast_to(band[:, None], (ny, nz))
            else:
                # 3層目: 90°回転（y 方向の帯、z にピッチ）
                u = np.mod(z_c, P_m)
                band = np.abs(u - P_m / 2.0) <= W_m / 2.0  # (nz,)
                in_slit = np.broadcast_to(band[None, :], (ny, nz))
            void[np.ix_(x_in, np.arange(ny), np.arange(nz))] |= in_slit[None, :, :]

    return void


def _core5_n_xy(c: Core5Candidate, n_min: int = 6, n_max: int = 12) -> int:
    """空隙を解像できる YZ 方向セル数を決める。

    孔径（スリット幅）を最低 3〜4 セル程度で表現できるようにする。
    粗すぎると空隙がセル中心の隙間に落ちて void_mask が空になる一方、
    細かすぎると 3D 解析コストが二乗で効くため上限を設ける。
    """
    need = int(np.ceil(4.0 * c.pitch_mm / max(c.d_mm, 1e-6)))
    return int(np.clip(need, n_min, n_max))


def _run_sim_core5_3d(
    c: Core5Candidate,
    t_end_min: float,
    n_cells_per_layer: int = CORE5_N_X_3D,
    n_cells_xy: int | None = None,
) -> tuple[float, float]:
    """千鳥配置3D FVMで1ケースを検証する。

    孔は千鳥（貫通経路なし）のため燃え抜けモードは使わず、
    void セルには静止空気＋空隙輻射（fvm_3d の温度依存処理）を適用する。

    Returns
    -------
    (T_iface, char_depth_mm, T_unexposed)
        YZ平均温度分布から求めた 2層目裏面(深さ60mm)温度 [°C]、
        炭化深さ [mm]、非加熱面温度 [°C]。
    """
    from clt_fire_sim.boundary import ISO834HeatedBC, ConvRadCoolingBC
    from clt_fire_sim.materials import make_properties
    from clt_fire_sim.solver.fvm_1d import MultiLayerProperties
    from clt_fire_sim.solver.fvm_3d import FVM3DSolver, make_clt_mesh_3d

    t_m = _CORE5_LAYER_MM / 1000.0
    layer_m = [t_m] * 5
    # ユニットセル寸法: 有孔=2p×2p（千鳥の最小周期）、スリット=P×P
    # ユニットセルは p×p（対称境界により千鳥を表現できる最小周期）
    L = c.pitch_mm / 1000.0
    if n_cells_xy is None:
        n_cells_xy = _core5_n_xy(c)

    mesh = make_clt_mesh_3d(
        layer_thicknesses=layer_m,
        specimen_width=L, specimen_height=L,
        n_cells_per_layer=n_cells_per_layer,
        mesh_ratio=1.05,
        n_cells_y=n_cells_xy, n_cells_z=n_cells_xy,
    )

    base = make_properties("sugi", _CORE5_RHO, _CORE5_MC)
    props = MultiLayerProperties(layer_m, [base] * 5)
    props.setup(np.repeat(mesh.x_centers, mesh.ny * mesh.nz))

    void_mask = _build_core5_void_mask_3d(c, mesh)

    bc_l = ISO834HeatedBC(alpha_c=25.0, eps_m=0.8, eps_f=1.0)
    bc_r = ConvRadCoolingBC(alpha_c=9.0, eps_m=0.8, T_inf=20.0)
    solver = FVM3DSolver(
        mesh=mesh, props=props, bc_left=bc_l, bc_right=bc_r,
        T_init=20.0, void_mask=void_mask, burn_through=False,
    )
    result = solver.solve(
        t_end=int(t_end_min) * 60,
        dt_base=10, dt_min=2, dt_max=30,
        n_picard=2, record_interval=120,
    )
    idx_t = int(np.argmin(np.abs(result["times"] / 60.0 - t_end_min)))
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    T_field = result["temperatures"][idx_t].reshape(nx, ny, nz)
    T_profile = T_field.mean(axis=(1, 2))  # YZ平均の厚み方向分布
    return (
        _temp_at_depth(mesh.x_centers, T_profile, _CORE5_EVAL_DEPTH_MM),
        _char_depth_mm(mesh.x_centers, T_profile),
        float(T_profile[-1]),
    )


def _compute_core5_front(
    cands: list[Core5Candidate], use_3d: bool = False,
) -> list[Core5Candidate]:
    """F1(2層目裏面温度)最小化 × F2(断熱抵抗R)最大化 の非支配解を返す。

    use_3d=True の場合は 3D 検証値で判定する（全候補3Dモード用）。
    """
    def f1(c: Core5Candidate) -> float:
        return c.T_iface_3d if use_3d else c.T_iface

    usable = [c for c in cands if np.isfinite(f1(c))]
    sorted_c = sorted(usable, key=lambda c: (f1(c), -c.R_value))
    front: list[Core5Candidate] = []
    best_r = -float("inf")
    for c in sorted_c:
        if c.R_value >= best_r:
            c.is_pareto = True
            front.append(c)
            best_r = c.R_value
    return front


def start_core5_optimization(
    process_type: str,
    v1_list: list[float],
    v2_list: list[float],
    v3_list: list[float],
    t_end_min: float = 60.0,
    n_cells_1d: int = CORE5_N_X_1D,
    n_cells_per_layer_3d: int = CORE5_N_X_3D,
    n_cells_xy_3d: int | None = None,
    n_3d_max: int = CORE5_N_3D_MAX,
    mode_3d: str = "front",
) -> None:
    """5層コア加工CLTのパレート最適化をバックグラウンドで開始する。

    Parameters
    ----------
    mode_3d : str
        "front" : 1D全列挙 → フロント解のみ千鳥3Dで検証（既定）
        "all"   : 1D全列挙 → **全候補**を千鳥3Dで解析し、フロントも3D値で判定
        "none"  : 1Dのみ
    """
    global _core5_state
    with _lock:
        if _core5_state.status == "running" or _state.status == "running":
            return
        candidates = generate_core5_candidates(process_type, v1_list, v2_list, v3_list)
        _stop_event.clear()
        _core5_state = Core5OptState(
            status="running", phase="1d",
            total=len(candidates), candidates=candidates,
            process_type=process_type, t_end_min=t_end_min,
            mode_3d=mode_3d,
        )

    t = threading.Thread(
        target=_run_core5_thread,
        args=(candidates, t_end_min, n_cells_1d,
              n_cells_per_layer_3d, n_cells_xy_3d, n_3d_max, mode_3d),
        daemon=True, name="core5-opt-worker",
    )
    t.start()


def _run_core5_thread(
    candidates: list[Core5Candidate],
    t_end_min: float,
    n_cells_1d: int,
    n_cells_per_layer_3d: int,
    n_cells_xy_3d: int,
    n_3d_max: int,
    mode_3d: str = "front",
) -> None:
    wall_start = time.monotonic()
    cache: dict = {}
    cache_1d_matched: dict = {}

    # ---- 比較基準（無加工5層CLT）----
    try:
        ref = compute_core5_reference(t_end_min, n_cells_1d)
        ref3 = (compute_core5_reference(t_end_min, n_cells_per_layer_3d)
                if mode_3d != "none" else None)
        with _lock:
            _core5_state.reference = ref
            _core5_state.reference_3d = ref3
    except Exception:
        pass

    # ---- フェーズ1: 1D 全列挙 ----
    for c in candidates:
        if _stop_event.is_set():
            break
        try:
            if c.sim_key not in cache:
                cache[c.sim_key] = _run_sim_core5_1d(c, t_end_min, n_cells_1d)
            c.T_iface, c.char_depth, c.T_unexposed, c.R_value = cache[c.sim_key]
        except Exception as e:
            c.error = str(e)
        with _lock:
            _core5_state.done += 1
            _core5_state.elapsed_s = time.monotonic() - wall_start

    valid = [c for c in candidates if np.isfinite(c.T_iface) and not c.error]
    front = _compute_core5_front(valid)

    # ---- フェーズ2: 千鳥3D解析 ----
    stopped = _stop_event.is_set()
    if not stopped and valid and mode_3d != "none":
        if mode_3d == "all":
            selected = valid
        elif front and n_3d_max > 0:
            # フロントを R 軸に沿って等間隔に最大 n_3d_max 点選ぶ
            front_sorted = sorted(front, key=lambda c: c.R_value)
            if len(front_sorted) > n_3d_max:
                idx = np.unique(np.round(
                    np.linspace(0, len(front_sorted) - 1, n_3d_max)
                ).astype(int))
                selected = [front_sorted[i] for i in idx]
            else:
                selected = front_sorted
        else:
            selected = []

        if selected:
            with _lock:
                _core5_state.phase = "3d"
                _core5_state.total_3d = len(selected)
                _core5_state.pareto_front = front

            cache_3d: dict = {}
            for c in selected:
                if _stop_event.is_set():
                    break
                try:
                    if c.sim_key_3d not in cache_3d:
                        cache_3d[c.sim_key_3d] = _run_sim_core5_3d(
                            c, t_end_min, n_cells_per_layer_3d, n_cells_xy_3d,
                        )
                    c.T_iface_3d, c.char_depth_3d, c.T_unexposed_3d = \
                        cache_3d[c.sim_key_3d]
                    # 同一x解像度の1Dを併走させ、メッシュ差を除いた3D効果を得る
                    if c.sim_key not in cache_1d_matched:
                        cache_1d_matched[c.sim_key] = _run_sim_core5_1d(
                            c, t_end_min, n_cells=n_cells_per_layer_3d,
                        )
                    c.T_iface_1d_matched = cache_1d_matched[c.sim_key][0]
                    c.verified_3d = True
                except Exception as e:
                    c.error = f"3D: {e}"
                with _lock:
                    _core5_state.done_3d += 1
                    _core5_state.elapsed_s = time.monotonic() - wall_start

    # ---- 全候補3Dならフロントを3D値で判定し直す ----
    front_is_3d = False
    verified = [c for c in valid if c.verified_3d and np.isfinite(c.T_iface_3d)]
    if mode_3d == "all" and verified and not _stop_event.is_set():
        for c in candidates:
            c.is_pareto = False
        front = _compute_core5_front(verified, use_3d=True)
        front_is_3d = True

    with _lock:
        _core5_state.status = "stopped" if _stop_event.is_set() else "done"
        _core5_state.phase = ""
        _core5_state.candidates = candidates
        _core5_state.pareto_front = front
        _core5_state.front_is_3d = front_is_3d
        _core5_state.elapsed_s = time.monotonic() - wall_start
