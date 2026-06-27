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

import logging
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
    mode: str = "1D"
    logs: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# モジュールレベルのシングルトン
# ---------------------------------------------------------------------------

_sim_state = SimState()
_stop_event = threading.Event()
_thread: threading.Thread | None = None
_lock = threading.Lock()  # 状態書き込みの排他制御


# ---------------------------------------------------------------------------
# ログハンドラ（バックグラウンドスレッドのログを SimState.logs に蓄積）
# ---------------------------------------------------------------------------

class _SimLogHandler(logging.Handler):
    """ソルバーログを SimState.logs に追記するハンドラ。"""

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        with _lock:
            _sim_state.logs.append(msg)


_log_handler = _SimLogHandler()
_log_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
                                             datefmt="%H:%M:%S"))

_WATCHED_LOGGERS = [
    "clt_fire_sim.solver.fvm_1d",
    "clt_fire_sim.solver.fvm_3d",
    "clt_fire_sim.runner",
]


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
    mode: str = "1D",
    specimen_width_mm: float = 300.0,
    specimen_height_mm: float = 300.0,
    n_cells_y: int = 4,
    n_cells_z: int = 4,
) -> None:
    """バックグラウンドスレッドでシミュレーションを開始する。

    すでに実行中の場合は何もしない（Streamlit の多重クリック防止）。

    Parameters
    ----------
    config : CLTConfig
        シミュレーション設定。
    outputs_base : Path
        結果保存ルートディレクトリ。
    mode : str
        "1D" または "3D"。
    specimen_width_mm / specimen_height_mm : float
        3D モード時の試験体幅・高さ [mm]。
    n_cells_y / n_cells_z : int
        3D モード時の y/z 方向セル数。
    """
    global _sim_state, _thread

    with _lock:
        if _sim_state.status == "running":
            return

        from clt_fire_sim.web.result_store import make_run_id
        run_id = make_run_id(config.specimen.name)
        out_dir = Path(outputs_base) / run_id

        _stop_event.clear()
        _sim_state = SimState(
            status="running",
            t_end_min=config.simulation.t_end_min,
            run_id=run_id,
            out_dir=out_dir,
            mode=mode,
        )

    # ログハンドラを登録
    for name in _WATCHED_LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        if _log_handler not in lg.handlers:
            lg.addHandler(_log_handler)

    _thread = threading.Thread(
        target=_run_simulation_thread,
        args=(config, out_dir, run_id, mode,
              specimen_width_mm, specimen_height_mm, n_cells_y, n_cells_z),
        daemon=True,
        name="clt-sim-worker",
    )
    _thread.start()


# ---------------------------------------------------------------------------
# バックグラウンドスレッドの実処理
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """SimState.logs にタイムスタンプ付きで直接追記する（ソルバー外のステップ用）。"""
    ts = time.strftime("%H:%M:%S")
    with _lock:
        _sim_state.logs.append(f"{ts}  INFO   web.runner — {msg}")


def _run_simulation_thread(
    config: Any,
    out_dir: Path,
    run_id: str,
    mode: str = "1D",
    specimen_width_mm: float = 300.0,
    specimen_height_mm: float = 300.0,
    n_cells_y: int = 4,
    n_cells_z: int = 4,
) -> None:
    """バックグラウンドスレッドのメイン処理。

    1. ソルバーを構築（mode に応じて 1D / 3D を選択）
    2. 進捗コールバック付きで solve() を実行
    3. 完了後に HDF5 + レポートを保存
    """
    import numpy as np

    from clt_fire_sim.boundary import ConvRadCoolingBC, CoolingFurnaceBC, ISO834HeatedBC
    from clt_fire_sim.boundary import iso834_temperature
    from clt_fire_sim.runner import _build_layer_properties, _evaluate_char_stop
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

        layer_thicknesses_m = [la.thickness_mm / 1000.0 for la in spec.layers]

        # ---- 物性値（1D / 3D 共通）----
        layer_props = [_build_layer_properties(la) for la in spec.layers]
        contact_resistances = [getattr(la, "contact_resistance_m2KW", 0.0) for la in spec.layers]

        # ---- 境界条件（1D / 3D 共通）----
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

        if mode == "3D":
            # ---- 3D メッシュ ----
            from clt_fire_sim.solver.fvm_3d import FVM3DSolver, make_clt_mesh_3d
            _log(f"3Dメッシュ構築: 幅={specimen_width_mm:.0f}mm 高さ={specimen_height_mm:.0f}mm "
                 f"ny={n_cells_y} nz={n_cells_z}")
            mesh = make_clt_mesh_3d(
                layer_thicknesses=layer_thicknesses_m,
                specimen_width=specimen_width_mm / 1000.0,
                specimen_height=specimen_height_mm / 1000.0,
                n_cells_per_layer=sim.n_cells_per_layer,
                mesh_ratio=sim.mesh_ratio,
                n_cells_y=n_cells_y,
                n_cells_z=n_cells_z,
            )
            _log(f"3Dメッシュ完了: {mesh.nx}×{mesh.ny}×{mesh.nz}={mesh.nx*mesh.ny*mesh.nz}セル")

            # 3D 用 MultiLayerProperties は x_centers（1D 等価）で setup
            mesh_1d = make_multi_layer_mesh(
                layer_thicknesses=layer_thicknesses_m,
                n_cells_per_layer=sim.n_cells_per_layer,
                ratio=sim.mesh_ratio,
            )
            props = MultiLayerProperties(layer_thicknesses_m, layer_props, contact_resistances)
            props.setup(mesh_1d.x_centers)

            solver = FVM3DSolver(mesh, props, bc_left, bc_right, T_init=T_init)

        else:
            # ---- 1D メッシュ ----
            _log(f"1Dメッシュ構築: {len(layer_thicknesses_m)}層 "
                 f"各{sim.n_cells_per_layer}セル")
            mesh = make_multi_layer_mesh(
                layer_thicknesses=layer_thicknesses_m,
                n_cells_per_layer=sim.n_cells_per_layer,
                ratio=sim.mesh_ratio,
            )
            _log(f"1Dメッシュ完了: {mesh.n_cells}セル")
            props = MultiLayerProperties(layer_thicknesses_m, layer_props, contact_resistances)
            props.setup(mesh.x_centers)
            solver = FVM1DSolver(mesh, props, bc_left, bc_right, T_init=T_init)

        t_end_s = sim.t_end_min * 60.0
        eval_times_s = [t * 60.0 for t in config.evaluation.eval_times_min]

        # ---- 放冷フェーズ境界条件 ----
        t_cooling_s = 0.0
        bc_left_cooling = None
        if sim.cooling_time_h > 0.0:
            T_at_heat_end = iso834_temperature(sim.t_end_min)
            bc_left_cooling = CoolingFurnaceBC(
                T_end_C=T_at_heat_end,
                t_heat_end_s=t_end_s,
                tau_s=sim.cooling_tau_min * 60.0,
                alpha_c=bc_cfg.heated.alpha_c,
                eps_m=bc_cfg.heated.eps_m,
                eps_f=bc_cfg.heated.eps_f,
                T_ambient=bc_cfg.unheated.T_inf,
            )
            t_cooling_s = sim.cooling_time_h * 3600.0

        t_total_s = t_end_s + t_cooling_s
        _log(f"解析開始: モード={mode} 加熱時間={sim.t_end_min:.0f}分 "
             f"放冷時間={sim.cooling_time_h*60:.0f}分 dt={sim.dt_base_s}s")

        # ---- 進捗コールバック ----
        def _progress_cb(current_t: float, t_end: float, T: np.ndarray) -> None:
            elapsed = time.monotonic() - wall_start
            frac = current_t / t_total_s if t_total_s > 0 else 0.0
            est_remaining = (elapsed / frac - elapsed) if frac > 1e-6 else 0.0
            T_flat = T.ravel()
            ts = time.strftime("%H:%M:%S")
            log_line = (f"{ts}  INFO   solver — "
                        f"t={current_t/60:.1f}分 "
                        f"T_max={float(T_flat.max()):.0f}°C "
                        f"char={_extract_char_depth_mm(mesh, T_flat if mode=='1D' else T_flat):.1f}mm "
                        f"elapsed={elapsed:.1f}s")
            with _lock:
                _sim_state.progress = min(frac, 1.0)
                _sim_state.current_time_min = current_t / 60.0
                _sim_state.max_temp_C = float(T_flat.max())
                _sim_state.char_depth_mm = _extract_char_depth_mm(mesh, T_flat)
                _sim_state.elapsed_s = elapsed
                _sim_state.est_remaining_s = max(0.0, est_remaining)
                _sim_state.logs.append(log_line)

        # ---- ソルバー実行 ----
        _solve_kwargs: dict = dict(
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
        if mode == "1D":
            # 放冷フェーズは1Dのみ対応
            _solve_kwargs["t_cooling"] = t_cooling_s
            _solve_kwargs["bc_left_cooling"] = bc_left_cooling
            _solve_kwargs["cooling_dt_base"] = max(sim.dt_base_s, 30.0)
        elif t_cooling_s > 0:
            _log("⚠️ 3Dモードは放冷フェーズ未対応のため、加熱フェーズのみ計算します。")
        result = solver.solve(**_solve_kwargs)

        # ---- 中断チェック ----
        if _stop_event.is_set():
            with _lock:
                _sim_state.status = "stopped"
                _sim_state.progress = _sim_state.current_time_min / (t_total_s / 60.0)
            return

        # ---- 性能評価 ----
        result["evaluation"] = evaluate_performance(result, config=config)
        result["config"] = config
        if mode == "3D":
            result["mesh"] = mesh

        # ---- 燃え止まり判定 ----
        if config.evaluation.char_stop_enabled and t_cooling_s > 0.0:
            result["char_stop"] = _evaluate_char_stop(result, config, mesh)
        else:
            result["char_stop"] = None

        # ---- 結果保存（HDF5 + レポート）----
        from clt_fire_sim.report import generate_report
        from clt_fire_sim.web.result_store import save_result_hdf5

        h5_path = save_result_hdf5(result, config, run_id, out_dir)
        generate_report(result, config=config, out_dir=out_dir)

        elapsed_total = time.monotonic() - wall_start
        _log(f"解析完了: 計算時間={elapsed_total:.1f}s")
        with _lock:
            _sim_state.status = "done"
            _sim_state.progress = 1.0
            _sim_state.result = result
            _sim_state.h5_path = h5_path
            _sim_state.elapsed_s = elapsed_total

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        _log(f"エラー: {exc}")
        for line in tb.splitlines():
            with _lock:
                _sim_state.logs.append(line)
        with _lock:
            _sim_state.status = "error"
            _sim_state.error_msg = str(exc)
            _sim_state.elapsed_s = time.monotonic() - wall_start
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


# _make_layer_props は clt_fire_sim.runner._build_layer_properties に統合済み
# この関数は後方互換性のため残す（呼び出し元がなければ不要）
def _make_layer_props(layer: Any) -> Any:
    """[後方互換] _build_layer_properties の旧名称。"""
    from clt_fire_sim.runner import _build_layer_properties
    return _build_layer_properties(layer)
