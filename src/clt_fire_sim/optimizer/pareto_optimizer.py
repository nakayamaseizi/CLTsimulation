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

    # 導出量（後から設定）
    vf: float = field(init=False)
    total_mm: float = field(init=False)
    T_clt_60: float = float("inf")   # 60分後CLT面温度 [°C]
    R_value: float = 0.0             # 断熱抵抗 [m²·K/W]
    is_pareto: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        self.vf = _compute_vf(self.d_mm, self.p_mm)
        self.total_mm = self.t_lam_mm * self.n_lam

    @property
    def r_key(self) -> tuple[float, float]:
        """シミュレーション結果の共有キー（同じ vf・総厚は同じ結果）"""
        return (round(self.vf, 4), float(self.total_mm))

    @property
    def protection_thickness_m(self) -> float:
        return self.total_mm / 1000.0

    @property
    def k_eff_rt(self) -> float:
        """室温換算有効熱伝導率 [W/m·K]"""
        return (1.0 - self.vf) * _WOOD_K_RT + self.vf * _AIR_K

    @property
    def r_analytical(self) -> float:
        """室温換算解析断熱抵抗 [m²·K/W]"""
        k = self.k_eff_rt
        if k <= 0:
            return 0.0
        return (self.total_mm / 1000.0) / k

    def use_ikehata_model(self) -> bool:
        """池畑(2021)実験式の適用可否"""
        return (
            3.0 <= self.d_mm <= 18.0
            and 6.0 <= self.t_lam_mm <= 30.0
            and self.d_mm > 0
        )


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
    """
    global _state

    with _lock:
        if _state.status == "running":
            return

        candidates = _generate_candidates(d_list, p_list, t_lam_list, n_lam_max)
        _stop_event.clear()
        _state = ParetoOptState(
            status="running",
            total=len(candidates),
            candidates=candidates,
        )

    t = threading.Thread(
        target=_run_pareto_thread,
        args=(candidates, base_mat, base_rho, n_cells, t_end_min),
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
) -> list[ParetoCandidate]:
    """有効な全候補を生成する。

    Notes
    -----
    - d=0（無孔）の場合、ピッチに依らず同一結果になるため
      各 (t_lam, n_lam) につき代表1点（p=p_list[0]）だけ生成する。
    - 同じ (vf, total_mm) を与える複数の (d, p) 組み合わせは
      シミュレーションキャッシュで重複除去するが、候補としてはそれぞれ残す
      （異なる物理設計として別の行に表示するため）。
    """
    max_total_mm = 96.0
    candidates: list[ParetoCandidate] = []
    # d=0（無孔）は (t_lam, n_lam) ごとに1点だけ生成するための追跡
    solid_keys_seen: set[tuple[float, int]] = set()

    for t_lam in sorted(t_lam_list):
        n_max = min(n_lam_max, int(max_total_mm / t_lam))
        for n_lam in range(1, n_max + 1):
            # 無孔候補：1つだけ（最小ピッチで代表）
            solid_key = (t_lam, n_lam)
            if solid_key not in solid_keys_seen and 0.0 in d_list:
                solid_keys_seen.add(solid_key)
                candidates.append(ParetoCandidate(
                    d_mm=0.0, p_mm=sorted(p_list)[0], t_lam_mm=t_lam, n_lam=n_lam
                ))

            for p in sorted(p_list):
                for d in sorted(d_list):
                    if d <= 0:
                        continue  # d=0 は上で別途追加済み
                    if d >= p:
                        continue  # 孔が重なる（物理的に無効）
                    if _compute_vf(d, p) > 0.79:
                        continue  # 孔が接触する限界を超える

                    candidates.append(ParetoCandidate(d_mm=d, p_mm=p, t_lam_mm=t_lam, n_lam=n_lam))

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
) -> None:
    """最適化のメインスレッド処理。"""
    wall_start = time.monotonic()

    # (vf, total_mm) → シミュレーション結果のキャッシュ
    sim_cache: dict[tuple[float, float], float] = {}

    for c in candidates:
        if _stop_event.is_set():
            break
        try:
            key = c.r_key
            if key not in sim_cache:
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
    """1ケースのシミュレーションを実行し、60分後CLT面温度を返す。"""
    from clt_fire_sim.materials import (
        PerforatedWoodProperties,
        PerforatedWoodAdvanced,
        make_properties,
    )
    from clt_fire_sim.boundary import ISO834HeatedBC, ConvRadCoolingBC
    from clt_fire_sim.solver.fvm_1d import FVM1DSolver, MultiLayerProperties, make_multi_layer_mesh

    protect_m = c.protection_thickness_m
    clt_layer_m = _CLT_LAYER_MM / 1000.0

    thicknesses = [protect_m] + [clt_layer_m] * _CLT_N_LAYERS

    # 保護層の物性値モデル選択
    base_props = make_properties(base_mat, base_rho, 0.12)
    if c.vf > 0 and c.use_ikehata_model():
        protect_props = PerforatedWoodAdvanced(
            base_props=base_props,
            void_fraction=c.vf,
            hole_diameter_mm=c.d_mm,
            hole_depth_mm=c.t_lam_mm,
            layer_thickness_mm=c.t_lam_mm,
        )
    elif c.vf > 0:
        protect_props = PerforatedWoodProperties(
            base_props=base_props,
            void_fraction=c.vf,
        )
    else:
        protect_props = base_props

    clt_props = make_properties("sugi", _CLT_RHO, 0.12)
    props_list = [protect_props] + [clt_props] * _CLT_N_LAYERS

    mesh = make_multi_layer_mesh(thicknesses, n_cells_per_layer=n_cells, ratio=1.05)
    props = MultiLayerProperties(thicknesses, props_list)
    props.setup(mesh.x_centers)

    bc_l = ISO834HeatedBC(alpha_c=25.0, eps_m=0.8, eps_f=1.0)
    bc_r = ConvRadCoolingBC(alpha_c=9.0, eps_m=0.8, T_inf=20.0)
    solver = FVM1DSolver(mesh, props, bc_l, bc_r, T_init=20.0)

    result = solver.solve(
        t_end=int(t_end_min) * 60,
        dt_base=10,
        dt_min=2,
        dt_max=30,
        n_picard=2,
        record_interval=60,
    )

    # CLT面温度（深さ = protect_m）を抽出
    x_mm = mesh.x_centers * 1000.0
    idx_depth = np.argmin(np.abs(x_mm - c.total_mm))
    idx_t60 = np.argmin(np.abs(result["times"] / 60.0 - t_end_min))
    return float(result["temperatures"][idx_t60, idx_depth])


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
