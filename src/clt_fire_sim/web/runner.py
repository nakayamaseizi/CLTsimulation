"""
runner.py（web 版）
===================
【役割】
Web UI からのシミュレーション実行を管理するモジュール。

バックグラウンドスレッドでソルバーを実行し、
進捗をモジュールレベルの状態オブジェクトに書き込む。
Streamlit の UI スレッドはこの状態を定期的に読み取って表示する。

【設計】
- 状態はモジュールレベルのシングルトン（_sim_state）
- ローカル単一ユーザー用途のため複数ユーザー同時実行は考慮しない
- threading.Event で安全な中断（KeyboardInterrupt ではなくフラグ経由）

【公開 API】
- get_state()        : 現在の SimState を取得
- start_simulation() : バックグラウンドで解析を開始
- stop_simulation()  : 実行中の解析を中断
- reset_state()      : アイドル状態にリセット

【CLI の runner.py との関係】
このファイルは clt_fire_sim.web.runner（Web専用）。
既存の clt_fire_sim.runner（CLIランナー）とは別モジュールで共存する。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 状態クラス
# ---------------------------------------------------------------------------

@dataclass
class SimState:
    """シミュレーション実行状態を保持するデータクラス。

    Attributes
    ----------
    status : str
        実行状態: "idle" | "running" | "done" | "stopped" | "error"
    progress : float
        進捗率 0.0〜1.0。
    current_time_min : float
        現在のシミュレーション時刻 [分]。
    t_end_min : float
        解析終了時刻 [分]（目標値）。
    max_temp_C : float
        現時点での最高温度 [°C]。
    char_depth_mm : float
        現時点での炭化深さ [mm]。
    elapsed_s : float
        実際の計算経過時間 [s]（wall clock）。
    est_remaining_s : float
        推定残り計算時間 [s]。
    result : dict or None
        完了後の結果辞書（solve() の返り値 + evaluation）。
    error_msg : str
        エラーメッセージ（status=="error" のとき）。
    run_id : str
        実行 ID（make_run_id() で生成）。
    out_dir : Path or None
        結果出力ディレクトリ。
    h5_path : Path or None
        保存した HDF5 ファイルのパス。
    """

    status: str = "idle"
    progress: float = 0.0
    current_time_min: float = 0.0
    t_end_min: float = 0.0
    max_temp_C: float = 20.0
    char_depth_mm: float = 0.0
    elapsed_s: float = 0.0
    est_remaining_s: float = 0.0
    result: dict | None = None
    error_msg: str = ""
    run_id: str = ""
    out_dir: Path | None = None
    h5_path: Path | None = None


# ---------------------------------------------------------------------------
# モジュールレベルのシングルトン
# ---------------------------------------------------------------------------

_sim_state = SimState()
_stop_event = threading.Event()
_thread: threading.Thread | None = None
_lock = threading.Lock()  # 状態書き込みの排他制御


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def get_state() -> SimState:
    """現在のシミュレーション状態を返す。"""
    return _sim_state


def reset_state() -> None:
    """アイドル状態にリセットする。"""
    global _sim_state
    with _lock:
        _sim_state = SimState()


def stop_simulation() -> None:
    """実行中のシミュレーションに中断シグナルを送る。

    バックグラウンドスレッドの次のタイムステップで停止する。
    """
    _stop_event.set()


def start_simulation(
    config: Any,
    outputs_base: Path = Path("outputs"),
) -> None:
    """バックグラウンドスレッドでシミュレーションを開始する。

    すでに実行中の場合は何もしない（Streamlit の多重クリック防止）。

    Parameters
    ----------
    config : CLTConfig
        シミュレーション設定。
    outputs_base : Path
        結果保存ルートディレクトリ。
    """
    global _sim_state, _thread

    with _lock:
        if _sim_state.status == "running":
            return

        # 前回の結果をリセットし、新しい実行を準備
        from clt_fire_sim.web.result_store import make_run_id
        run_id = make_run_id(config.specimen.name)
        out_dir = Path(outputs_base) / run_id

        _stop_event.clear()
        _sim_state = SimState(
            status="running",
            t_end_min=config.simulation.t_end_min,
            run_id=run_id,
            out_dir=out_dir,
        )

    # デーモンスレッドとして起動（メインプロセス終了時に自動終了）
    _thread = threading.Thread(
        target=_run_simulation_thread,
        args=(config, out_dir, run_id),
        daemon=True,
        name="clt-sim-worker",
    )
    _thread.start()


# ---------------------------------------------------------------------------
# バックグラウンドスレッドの実処理
# ---------------------------------------------------------------------------

def _run_simulation_thread(
    config: Any,
    out_dir: Path,
    run_id: str,
) -> None:
    """バックグラウンドスレッドのメイン処理。

    1. ソルバーを構築
    2. 進捗コールバック付きで solve() を実行
    3. 完了後に HDF5 + レポートを保存
    """
    import numpy as np

    from clt_fire_sim.boundary import ConvRadCoolingBC, ISO834HeatedBC
    from clt_fire_sim.materials import make_properties
    from clt_fire_sim.report import compute_charring_rate, evaluate_performance
    from clt_fire_sim.solver.fvm_1d import (
        FVM1DSolver,
        MultiLayerProperties,
        make_multi_layer_mesh,
    )

    wall_start = time.monotonic()

    try:
        sim = config.simulation
        spec = config.specimen
        bc_cfg = config.boundary

        # ---- メッシュ構築 ----
        layer_thicknesses_m = [
            layer.thickness_mm / 1000.0 for layer in spec.layers
        ]
        mesh = make_multi_layer_mesh(
            layer_thicknesses=layer_thicknesses_m,
            n_cells_per_layer=sim.n_cells_per_layer,
            ratio=sim.mesh_ratio,
        )

        # ---- 物性値 ----
        layer_props = [
            make_properties(
                material=layer.material,
                rho_0=layer.rho_0_kg_m3,
                moisture_content=layer.moisture_content,
            )
            for layer in spec.layers
        ]
        props = MultiLayerProperties(layer_thicknesses_m, layer_props)
        props.setup(mesh.x_centers)

        # ---- 境界条件 ----
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
        solver = FVM1DSolver(mesh, props, bc_left, bc_right, T_init=T_init)

        t_end_s = sim.t_end_min * 60.0
        eval_times_s = [t * 60.0 for t in config.evaluation.eval_times_min]

        # ---- 進捗コールバック ----
        def _progress_cb(current_t: float, t_end: float, T: np.ndarray) -> None:
            """記録タイミングごとに状態を更新する。"""
            elapsed = time.monotonic() - wall_start
            frac = current_t / t_end if t_end > 0 else 0.0
            est_remaining = (elapsed / frac - elapsed) if frac > 1e-6 else 0.0

            with _lock:
                _sim_state.progress = min(frac, 1.0)
                _sim_state.current_time_min = current_t / 60.0
                _sim_state.max_temp_C = float(T.max())
                _sim_state.char_depth_mm = _extract_char_depth_mm(mesh, T)
                _sim_state.elapsed_s = elapsed
                _sim_state.est_remaining_s = max(0.0, est_remaining)

        # ---- ソルバー実行 ----
        result = solver.solve(
            t_end=t_end_s,
            dt_base=sim.dt_base_s,
            dt_min=sim.dt_min_s,
            dt_max=sim.dt_max_s,
            n_picard=sim.n_picard,
            record_interval=sim.record_interval_s,
            eval_times=eval_times_s,
            progress_callback=_progress_cb,
            stop_event=_stop_event,
        )

        # ---- 中断チェック ----
        if _stop_event.is_set():
            with _lock:
                _sim_state.status = "stopped"
                _sim_state.progress = _sim_state.current_time_min / sim.t_end_min
            return

        # ---- 性能評価 ----
        result["evaluation"] = evaluate_performance(result, config=config)
        result["config"] = config

        # ---- 結果保存（HDF5 + レポート）----
        from clt_fire_sim.report import generate_report
        from clt_fire_sim.web.result_store import save_result_hdf5

        h5_path = save_result_hdf5(result, config, run_id, out_dir)
        generate_report(result, config=config, out_dir=out_dir)

        with _lock:
            _sim_state.status = "done"
            _sim_state.progress = 1.0
            _sim_state.result = result
            _sim_state.h5_path = h5_path
            _sim_state.elapsed_s = time.monotonic() - wall_start

    except Exception as exc:
        import traceback
        with _lock:
            _sim_state.status = "error"
            _sim_state.error_msg = str(exc)
            _sim_state.elapsed_s = time.monotonic() - wall_start
        # デバッグ用にスタックトレースを stderr に出力
        traceback.print_exc()


def _extract_char_depth_mm(mesh: Any, T: Any) -> float:
    """現在の温度プロファイルから炭化深さ [mm] を計算する。"""
    import numpy as np

    x = mesh.x_centers
    # 300°C を超えているセルの最大位置
    charred = T >= 300.0
    if not np.any(charred):
        return 0.0
    return float(x[charred].max()) * 1000.0
