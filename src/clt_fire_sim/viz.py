"""
viz.py
======
【役割】
シミュレーション結果の可視化（matplotlib）。

【生成できるグラフ】
1. plot_charring_progress  ─ 炭化深さの時刻歴 + Eurocode 参照線
2. plot_temperature_profiles ─ 厚み方向の温度分布（複数時刻）
3. plot_surface_history    ─ 加熱面・非加熱面温度の時刻歴
4. plot_summary_panel      ─ 上記 3 枚を並べた概要図

【使い方】
    from clt_fire_sim.viz import plot_summary_panel
    fig = plot_summary_panel(result, config, out_path="outputs/summary.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .boundary import iso834_temperature


# ---------------------------------------------------------------------------
# フォント設定
# ---------------------------------------------------------------------------

def _setup_font() -> None:
    """日本語フォントと unicode マイナスを設定する。

    Windows: Yu Gothic、Linux (Streamlit Cloud): Noto Sans CJP / IPAGothic、
    どれも無ければ警告なしでデフォルトフォントを使う。
    """
    import matplotlib.font_manager as fm

    # 優先順位順に試す
    candidates = ["Yu Gothic", "Noto Sans CJK JP", "IPAGothic", "IPAPGothic", "TakaoPGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            break
    # どれも見つからなければデフォルトのまま（エラーは出さない）
    matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 個別プロット関数
# ---------------------------------------------------------------------------

def plot_charring_progress(
    result: dict[str, Any],
    beta_ref: float = 0.65,
    tolerance: float = 0.15,
    ax: Any = None,
) -> Any:
    """炭化深さの時刻歴と Eurocode 5 参照値を描画する。

    Parameters
    ----------
    result : dict
        solve() または run_from_config() の返り値。
    beta_ref : float
        Eurocode 5 の標準炭化速度 [mm/min]。デフォルト 0.65。
    tolerance : float
        許容誤差の割合（±）。デフォルト 0.15（±15%）。
    ax : Axes or None
        描画先 Axes。None の場合は新規作成。

    Returns
    -------
    matplotlib.axes.Axes
    """
    _setup_font()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    times_min = result["times"] / 60.0
    char_mm = result["char_depths"] * 1000.0
    t_max = times_min.max()

    ax.plot(times_min, char_mm, "b-", lw=2, label="解析結果（300°C 等温面）")

    t_ref = np.linspace(0, t_max, 200)
    ax.plot(t_ref, beta_ref * t_ref, "r--", lw=1.5,
            label=f"Eurocode β₀={beta_ref} mm/min")
    ax.fill_between(
        t_ref,
        (1.0 - tolerance) * beta_ref * t_ref,
        (1.0 + tolerance) * beta_ref * t_ref,
        alpha=0.2, color="red",
        label=f"許容範囲 ±{tolerance * 100:.0f}%",
    )

    ax.set_xlabel("時間 [分]")
    ax.set_ylabel("炭化深さ [mm]")
    ax.set_xlim(0, t_max)
    ax.set_ylim(0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_temperature_profiles(
    result: dict[str, Any],
    times_min: list[float] | None = None,
    x_max_mm: float = 120.0,
    ax: Any = None,
) -> Any:
    """厚み方向の温度分布を複数時刻で描画する。

    Parameters
    ----------
    result : dict
        solve() または run_from_config() の返り値。
    times_min : list[float] or None
        描画する時刻 [分]。None の場合は等間隔で最大 7 点選択。
    x_max_mm : float
        x 軸の最大値 [mm]。
    ax : Axes or None
        描画先 Axes。

    Returns
    -------
    matplotlib.axes.Axes
    """
    _setup_font()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    all_times_min = result["times"] / 60.0
    t_max = all_times_min.max()

    if times_min is None:
        # 等間隔で最大 7 時刻を選択
        n_pts = min(7, len(all_times_min))
        times_min = np.linspace(t_max / n_pts, t_max, n_pts).tolist()

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(times_min)))

    x_mm = result["x_centers"] * 1000.0

    for t_target, color in zip(times_min, colors):
        idx = int(np.argmin(np.abs(all_times_min - t_target)))
        t_actual = all_times_min[idx]

        # 3D の場合は最初の y,z スライス（または 1D）
        T_arr = result["temperatures"][idx]
        if T_arr.ndim == 1 and len(T_arr) > len(x_mm):
            # 3D flat array: take first y,z slice (j=0, k=0)
            ny_nz = len(T_arr) // len(x_mm)
            T_plot = T_arr[::ny_nz]
        else:
            T_plot = T_arr

        ax.plot(x_mm, T_plot, color=color, lw=1.5,
                label=f"{t_actual:.0f}分")

    ax.axhline(300, color="k", ls="--", lw=0.8, alpha=0.7, label="炭化温度 300°C")
    ax.set_xlabel("加熱面からの深さ [mm]")
    ax.set_ylabel("温度 [°C]")
    ax.set_xlim(0, min(x_max_mm, x_mm.max()))
    ax.set_ylim(0)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    return ax


def plot_surface_history(
    result: dict[str, Any],
    ax: Any = None,
) -> Any:
    """加熱面・非加熱面温度と ISO 834 ガス温度の時刻歴を描画する。

    Parameters
    ----------
    result : dict
        solve() または run_from_config() の返り値。
    ax : Axes or None
        描画先 Axes。

    Returns
    -------
    matplotlib.axes.Axes
    """
    _setup_font()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    times_min = result["times"] / 60.0
    T_mat = result["temperatures"]

    # 加熱面温度（i=0 セル）
    if T_mat.ndim == 2:
        T_heated = T_mat[:, 0]    # 1D: 最左端セル
        T_unheated = T_mat[:, -1]  # 1D: 最右端セル
    else:
        T_heated = T_mat[:, 0]
        T_unheated = T_mat[:, -1]

    # ISO 834 ガス温度
    T_iso = np.array([iso834_temperature(t) for t in times_min])

    ax.plot(times_min, T_iso, "k--", lw=1.2, label="ISO 834 ガス温度")
    ax.plot(times_min, T_heated, "r-", lw=2, label="加熱面温度（第 1 セル）")
    ax.plot(times_min, T_unheated, "b-", lw=2, label="非加熱面温度（最終セル）")
    ax.axhline(160.0, color="b", ls=":", lw=1.0, alpha=0.7, label="遮熱基準 160°C")

    ax.set_xlabel("時間 [分]")
    ax.set_ylabel("温度 [°C]")
    ax.set_xlim(0, times_min.max())
    ax.set_ylim(0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# 概要パネル図
# ---------------------------------------------------------------------------

def plot_summary_panel(
    result: dict[str, Any],
    config: Any = None,
    out_path: str | Path | None = None,
    times_min_profiles: list[float] | None = None,
) -> Any:
    """3 パネルの概要図を生成する。

    左: 炭化深さ時刻歴 + Eurocode 参照
    中: 厚み方向温度分布（複数時刻）
    右: 加熱面・非加熱面温度時刻歴

    Parameters
    ----------
    result : dict
        solve() または run_from_config() の返り値。
    config : CLTConfig or None
        設定オブジェクト（タイトルに名前を表示する場合に使用）。
    out_path : str or Path or None
        保存先ファイルパス（None の場合は保存しない）。
    times_min_profiles : list[float] or None
        温度分布プロットに使う時刻 [分]。

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_font()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- 炭化深さ ---
    plot_charring_progress(result, ax=axes[0])

    # 炭化速度を回帰して追記
    from .report import compute_charring_rate
    beta = compute_charring_rate(result)
    axes[0].set_title(f"炭化進行  β={beta:.3f} mm/min")

    # --- 温度分布 ---
    plot_temperature_profiles(result, times_min=times_min_profiles, ax=axes[1])
    axes[1].set_title("厚み方向の温度分布")

    # --- 表面温度 ---
    plot_surface_history(result, ax=axes[2])
    axes[2].set_title("表面温度の時刻歴")

    # メインタイトル
    name = ""
    if config is not None and hasattr(config, "specimen"):
        name = f": {config.specimen.name}"
    fig.suptitle(f"CLT 耐火シミュレーション結果{name}", fontsize=12)

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# CSV 出力
# ---------------------------------------------------------------------------

def save_results_csv(
    result: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """シミュレーション結果の時刻歴を CSV ファイルに保存する。

    【CSV の列】
    time_s, time_min, char_depth_mm, T_heated_C, T_unheated_C, iso834_gas_C

    Parameters
    ----------
    result : dict
        solve() または run_from_config() の返り値。
    out_path : str or Path
        保存先ファイルパス（.csv）。

    Returns
    -------
    Path
        保存したファイルのパス。
    """
    import csv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    times_s = result["times"]
    times_min = times_s / 60.0
    char_mm = result["char_depths"] * 1000.0
    T_mat = result["temperatures"]

    T_heated = T_mat[:, 0] if T_mat.ndim == 2 else T_mat[:, 0]
    T_unheated = T_mat[:, -1] if T_mat.ndim == 2 else T_mat[:, -1]
    T_iso = np.array([iso834_temperature(t) for t in times_min])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s", "time_min", "char_depth_mm",
            "T_heated_C", "T_unheated_C", "iso834_gas_C",
        ])
        for row in zip(
            times_s.tolist(),
            times_min.tolist(),
            char_mm.tolist(),
            T_heated.tolist(),
            T_unheated.tolist(),
            T_iso.tolist(),
        ):
            writer.writerow([f"{v:.4f}" for v in row])

    return out_path
