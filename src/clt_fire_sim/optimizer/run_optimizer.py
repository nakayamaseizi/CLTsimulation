"""
run_optimizer.py
================
【役割】
孔配置最適化のバックグラウンド実行エンジン。

【流れ】
1. ベースライン（孔なし）3D シミュレーションを実行
2. 各候補パターンを順番に 3D シミュレーション
3. 非加熱面温度・炭化深さで性能ランキング
4. 最良パターンを状態に記録

【公開 API】
- get_opt_state()     : 現在の最適化状態を返す
- start_optimization(): バックグラウンドで最適化を開始
- stop_optimization() : 実行中の最適化を中断
- reset_opt_state()   : アイドル状態にリセット
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from clt_fire_sim.optimizer.hole_pattern import HolePattern, make_void_mask_3d


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class PatternResult:
    """単一パターンの評価結果。"""
    pattern: HolePattern
    result: dict | None = None
    T_unheated_final_C: float = float("nan")   # 解析終了時の非加熱面平均温度 [°C]
    char_depth_final_mm: float = float("nan")  # 解析終了時の平均炭化深さ [mm]
    elapsed_s: float = 0.0
    error: str = ""


@dataclass
class OptState:
    """最適化全体の実行状態。"""
    status: str = "idle"            # idle | running | done | stopped | error
    total: int = 0                  # 実行予定パターン数（ベースライン含む）
    completed: int = 0              # 完了済みパターン数
    current_name: str = ""          # 現在実行中のパターン名
    progress_frac: float = 0.0      # 現在パターンの進捗 (0〜1)
    baseline: PatternResult | None = None
    results: list[PatternResult] = field(default_factory=list)
    best_idx: int = -1              # results リスト中の最良パターンインデックス
    error_msg: str = ""
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# モジュールレベルのシングルトン
# ---------------------------------------------------------------------------

_opt_state = OptState()
_stop_event = threading.Event()
_thread: threading.Thread | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def get_opt_state() -> OptState:
    """現在の最適化状態を返す。"""
    return _opt_state


def reset_opt_state() -> None:
    """アイドル状態にリセットする。"""
    global _opt_state
    with _lock:
        _opt_state = OptState()


def stop_optimization() -> None:
    """実行中の最適化に中断シグナルを送る。"""
    _stop_event.set()


def start_optimization(
    config: Any,
    patterns: list[HolePattern],
    panel_Ly: float,
    panel_Lz: float,
    ny: int,
    nz: int,
    burn_through: bool = True,
) -> None:
    """バックグラウンドスレッドで最適化を開始する。

    Parameters
    ----------
    config : CLTConfig
        シミュレーション設定。
    patterns : list[HolePattern]
        評価する候補パターンリスト。
    panel_Ly, panel_Lz : float
        CLT パネルの断面サイズ（幅・長さ）[m]。
    ny, nz : int
        y, z 方向のメッシュセル数。
    burn_through : bool
        True の場合、孔内部を ISO 834 温度に固定（燃え抜けモード）。
        デフォルト True（実挙動に近い保守的な評価）。
    """
    global _opt_state, _thread

    with _lock:
        if _opt_state.status == "running":
            return
        _stop_event.clear()
        _opt_state = OptState(
            status="running",
            total=len(patterns) + 1,   # +1: ベースライン
        )

    _thread = threading.Thread(
        target=_run_optimization_thread,
        args=(config, patterns, panel_Ly, panel_Lz, ny, nz, burn_through),
        daemon=True,
        name="clt-optimizer",
    )
    _thread.start()


# ---------------------------------------------------------------------------
# バックグラウンドスレッド
# ---------------------------------------------------------------------------

def _run_optimization_thread(
    config: Any,
    patterns: list[HolePattern],
    panel_Ly: float,
    panel_Lz: float,
    ny: int,
    nz: int,
    burn_through: bool = True,
) -> None:
    """最適化のメイン処理（バックグラウンドスレッド）。"""
    wall_start = time.monotonic()

    try:
        from clt_fire_sim.boundary import ConvRadCoolingBC, ISO834HeatedBC
        from clt_fire_sim.solver.fvm_1d import MultiLayerProperties
        from clt_fire_sim.solver.fvm_3d import (
            FVM3DSolver, make_clt_mesh_3d, setup_multi_layer_props_3d,
        )

        sim = config.simulation
        spec = config.specimen
        bc_cfg = config.boundary

        # ── メッシュ（全パターン共通）────────────────────────────────
        layer_thicknesses_m = [l.thickness_mm / 1000.0 for l in spec.layers]
        mesh = make_clt_mesh_3d(
            layer_thicknesses=layer_thicknesses_m,
            n_cells_per_layer=sim.n_cells_per_layer,
            specimen_width=panel_Ly,
            specimen_height=panel_Lz,
            n_cells_y=ny,
            n_cells_z=nz,
            mesh_ratio=sim.mesh_ratio,
        )

        # ── 物性値（全パターン共通）──────────────────────────────────
        from clt_fire_sim.web.runner import _make_layer_props
        layer_props = [_make_layer_props(l) for l in spec.layers]
        props = setup_multi_layer_props_3d(layer_thicknesses_m, layer_props, mesh)

        # ── 境界条件（全パターン共通）────────────────────────────────
        bc_left = ISO834HeatedBC(
            alpha_c=bc_cfg.heated.alpha_c,
            eps_m=bc_cfg.heated.eps_m,
            eps_f=bc_cfg.heated.eps_f,
        )
        bc_right = ConvRadCoolingBC(
            alpha_c=bc_cfg.unheated.alpha_c,
            eps_m=bc_cfg.unheated.eps_m,
            T_inf=bc_cfg.unheated.T_inf,
        )
        T_init = bc_cfg.unheated.T_inf
        t_end_s = sim.t_end_min * 60.0

        def _progress_cb(t_now, t_end, T):
            with _lock:
                _opt_state.progress_frac = t_now / t_end if t_end > 0 else 0.0

        # ── ベースライン（孔なし）────────────────────────────────────
        _set_state(current_name="ベースライン（孔なし）", progress_frac=0.0)

        baseline_pr = _run_one(
            mesh, props, bc_left, bc_right, T_init, t_end_s, sim,
            void_mask=None, progress_cb=_progress_cb,
            stop_event=_stop_event,
        )
        with _lock:
            _opt_state.baseline = baseline_pr
            _opt_state.completed = 1

        if _stop_event.is_set():
            _finalize(results=[], best_idx=-1, wall_start=wall_start, stopped=True)
            return

        # ── 各候補パターン ───────────────────────────────────────────
        results: list[PatternResult] = []
        for idx, pattern in enumerate(patterns):
            if _stop_event.is_set():
                break

            _set_state(
                current_name=pattern.name,
                progress_frac=0.0,
            )

            void_3d = make_void_mask_3d(pattern.void_yz, mesh.nx)
            pr = _run_one(
                mesh, props, bc_left, bc_right, T_init, t_end_s, sim,
                void_mask=void_3d,
                burn_through=burn_through,
                progress_cb=_progress_cb,
                stop_event=_stop_event,
                pattern=pattern,
            )
            results.append(pr)

            with _lock:
                _opt_state.results = results.copy()
                _opt_state.completed = idx + 2   # +2: baseline + this

        # ── 最良パターン選定（非加熱面温度が最低のもの）────────────
        valid = [
            (i, r) for i, r in enumerate(results)
            if not r.error and not np.isnan(r.T_unheated_final_C)
        ]
        best_idx = min(valid, key=lambda x: x[1].T_unheated_final_C)[0] if valid else -1

        _finalize(results=results, best_idx=best_idx, wall_start=wall_start)

    except Exception as exc:
        import traceback
        with _lock:
            _opt_state.status = "error"
            _opt_state.error_msg = str(exc)
            _opt_state.elapsed_s = time.monotonic() - wall_start
        traceback.print_exc()


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _run_one(
    mesh, props, bc_left, bc_right,
    T_init, t_end_s, sim,
    void_mask=None,
    burn_through: bool = True,
    progress_cb=None,
    stop_event=None,
    pattern: HolePattern | None = None,
) -> PatternResult:
    """1 パターンのシミュレーションを実行して PatternResult を返す。"""
    from clt_fire_sim.solver.fvm_3d import FVM3DSolver

    pat = pattern or HolePattern(
        "ベースライン", np.zeros((1, 1), bool), 0.0, "孔なし"
    )
    t0 = time.monotonic()

    try:
        solver = FVM3DSolver(
            mesh, props, bc_left, bc_right,
            T_init=T_init,
            void_mask=void_mask,
            burn_through=(burn_through and void_mask is not None),
        )
        result = solver.solve(
            t_end=t_end_s,
            dt_base=sim.dt_base_s,
            dt_min=sim.dt_min_s,
            dt_max=sim.dt_max_s,
            n_picard=sim.n_picard,
            record_interval=sim.record_interval_s,
            stop_event=stop_event,
            progress_callback=progress_cb,
        )

        T_unheated = _extract_T_unheated(result, mesh)
        char_mm = float(result["char_depths"][-1]) * 1000.0

        return PatternResult(
            pattern=pat,
            result=result,
            T_unheated_final_C=T_unheated,
            char_depth_final_mm=char_mm,
            elapsed_s=time.monotonic() - t0,
        )
    except Exception as exc:
        return PatternResult(
            pattern=pat,
            error=str(exc),
            elapsed_s=time.monotonic() - t0,
        )


def _extract_T_unheated(result: dict, mesh) -> float:
    """非加熱面（x=Lx）の平均温度を抽出する。"""
    T_mat = result["temperatures"]   # (n_times, N)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    T_last = T_mat[-1].reshape(nx, ny, nz)
    return float(T_last[-1, :, :].mean())


def _set_state(**kwargs) -> None:
    with _lock:
        for k, v in kwargs.items():
            setattr(_opt_state, k, v)


def _finalize(
    results: list[PatternResult],
    best_idx: int,
    wall_start: float,
    stopped: bool = False,
) -> None:
    with _lock:
        _opt_state.results = results
        _opt_state.best_idx = best_idx
        _opt_state.status = "stopped" if stopped else "done"
        _opt_state.elapsed_s = time.monotonic() - wall_start
        _opt_state.progress_frac = 1.0
