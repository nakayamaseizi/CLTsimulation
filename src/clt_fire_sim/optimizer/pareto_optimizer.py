"""
pareto_optimizer.py
===================
有孔ラミナ保護層のパレート最適化。

【設計変数】
  d_mm   : 孔径 [mm]（同一試験体内は一定）
  p_mm   : 孔ピッチ（中心間距離）[mm]（同一試験体内は一定）
  t_lam  : ラミナ厚さ [mm]（12mm or 24mm）
  n_lam  : ラミナ枚数（1〜8、総厚 ≤ 96mm）

【導出量】
  vf     : 空洞率 = π*(d/2)² / p²（円孔・正方形配置）
  total  : 保護層総厚 = t_lam * n_lam [mm]

【目的関数（2目的）】
  F1 : 60分後CLT面温度 [°C]  → 最小化（耐火性能↑）
  F2 : 断熱抵抗 R [m²·K/W]  → 最大化（断熱性能↑）

【物性モデルの選択】
  d≤18mm かつ t_lam≤30mm : 池畑(2021)実験式（PerforatedWoodAdvanced）
  それ以外                : 並列混合則（PerforatedWoodProperties）

【アルゴリズム】
  1. (d, p, t_lam, n_lam) の全有効組み合わせを生成
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


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class ParetoCandidate:
    """1つの設計変数の組み合わせ"""
    d_mm: float
    p_mm: float
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
        self.vf = _compute_vf(self.d_mm, self.p_mm)
        self.total_mm = self.t_lam_mm * self.n_lam

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
    """現在の最適化状態を返す。"""
    return _state


def stop_pareto() -> None:
    """実行中の最適化に中断シグナルを送る。"""
    _stop_event.set()


# ---------------------------------------------------------------------------
# 公開 API：最適化開始
# ---------------------------------------------------------------------------

def start_pareto_optimization(
    d_list: list[float],
    p_list: list[float],
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
    p_list : list[float]
        孔ピッチの候補リスト [mm]
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
            d_list, p_list, t_lam_list, n_lam_max, t_face_list, face_mat
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

def _compute_vf(d_mm: float, p_mm: float) -> float:
    """円孔・正方形配置の空洞率を計算する。"""
    if d_mm <= 0 or p_mm <= 0:
        return 0.0
    return np.pi * (d_mm / 2.0) ** 2 / p_mm ** 2


def _generate_candidates(
    d_list: list[float],
    p_list: list[float],
    t_lam_list: list[float],
    n_lam_max: int,
    t_face_list: list[float] | None = None,
    face_mat: str = "sugi",
) -> list[ParetoCandidate]:
    """有効な全候補を生成する。

    Notes
    -----
    - d=0（無孔）の場合、ピッチに依らず同一結果になるため
      各 (t_lam, n_lam, t_face) につき代表1点（p=p_list[0]）だけ生成する。
    - 同じ (vf, total_mm, t_face_mm) を与える複数の (d, p) 組み合わせは
      シミュレーションキャッシュで重複除去するが、候補としてはそれぞれ残す
      （異なる物理設計として別の行に表示するため）。
    - t_face_list: 表側（火側）無孔パネル厚候補 [mm]。0 = 無し。
    """
    if t_face_list is None:
        t_face_list = [0.0]

    max_total_mm = 96.0  # 有孔ラミナ層の最大総厚（表面パネルは含まない）
    candidates: list[ParetoCandidate] = []
    solid_keys_seen: set = set()
    min_p = sorted(p_list)[0] if p_list else 30.0

    for t_face in sorted(t_face_list):
        for t_lam in sorted(t_lam_list):
            n_max = min(n_lam_max, int(max_total_mm / t_lam))
            for n_lam in range(1, n_max + 1):
                # 無孔候補：(t_lam, n_lam, t_face) ごとに1点だけ
                solid_key = (t_lam, n_lam, t_face)
                if solid_key not in solid_keys_seen and 0.0 in d_list:
                    solid_keys_seen.add(solid_key)
                    candidates.append(ParetoCandidate(
                        d_mm=0.0, p_mm=min_p, t_lam_mm=t_lam,
                        n_lam=n_lam, t_face_mm=t_face, face_mat=face_mat,
                    ))

                for p in sorted(p_list):
                    for d in sorted(d_list):
                        if d <= 0:
                            continue
                        if d >= p:
                            continue
                        if _compute_vf(d, p) > 0.79:
                            continue

                        candidates.append(ParetoCandidate(
                            d_mm=d, p_mm=p, t_lam_mm=t_lam,
                            n_lam=n_lam, t_face_mm=t_face, face_mat=face_mat,
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
                key = (round(c.d_mm, 2), round(c.p_mm, 2),
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
    p_m = c.p_mm / 1000.0

    # レイヤー構成（表面パネルあり / なし）
    if face_m > 0:
        layer_thicknesses = [face_m, protect_m] + [clt_layer_m] * _CLT_N_LAYERS
        n_face_x = n_cells_per_layer   # 表面パネルのxセル数
        n_perf_x_start = n_cells_per_layer  # 有孔層開始インデックス
    else:
        layer_thicknesses = [protect_m] + [clt_layer_m] * _CLT_N_LAYERS
        n_face_x = 0
        n_perf_x_start = 0

    # 3Dメッシュ生成（ユニットセル = p×p mm のYZ断面）
    mesh = make_clt_mesh_3d(
        layer_thicknesses=layer_thicknesses,
        specimen_width=p_m,
        specimen_height=p_m,
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
        half_p = c.p_mm / 2.0
        perf_start = n_perf_x_start
        perf_end = perf_start + n_cells_per_layer
        for j in range(ny):
            y_c = (j + 0.5) / ny * c.p_mm
            for k in range(nz):
                z_c = (k + 0.5) / nz * c.p_mm
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


def get_default_p_list() -> list[float]:
    """デフォルトのピッチ候補リスト [mm]。"""
    return [30.0, 40.0, 50.0, 60.0, 80.0, 100.0]
