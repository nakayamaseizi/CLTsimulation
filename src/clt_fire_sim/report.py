"""
report.py
=========
【役割】
シミュレーション結果の集計・評価・レポート生成。

【提供する関数】
1. compute_charring_rate   ─ 線形回帰で炭化速度 β [mm/min] を算出
2. evaluate_performance    ─ 耐火・遮熱性能を判定して辞書を返す
3. generate_report         ─ グラフ・CSV・テキストを outputs/ に一括保存

【使い方】
    from clt_fire_sim.report import generate_report
    paths = generate_report(result, config, out_dir="outputs/run01")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .boundary import iso834_temperature


# ---------------------------------------------------------------------------
# 炭化速度
# ---------------------------------------------------------------------------

def compute_charring_rate(
    result: dict[str, Any],
    t_start_min: float = 10.0,
    t_end_min: float = 60.0,
) -> float:
    """線形回帰で炭化速度 β [mm/min] を算出する。

    Parameters
    ----------
    result : dict
        solve() または run_from_config() の返り値。
    t_start_min : float
        回帰区間の開始時刻 [分]。初期の炭化遅れを除外するため 10 分以降を推奨。
    t_end_min : float
        回帰区間の終了時刻 [分]。

    Returns
    -------
    float
        炭化速度 β [mm/min]。回帰点が 2 点未満の場合は NaN を返す。
    """
    times_min = result["times"] / 60.0
    char_mm = result["char_depths"] * 1000.0

    mask = (times_min >= t_start_min) & (times_min <= t_end_min) & (char_mm > 1.0)
    if mask.sum() < 2:
        return float("nan")

    coeffs = np.polyfit(times_min[mask], char_mm[mask], 1)
    return float(coeffs[0])


# ---------------------------------------------------------------------------
# 性能評価
# ---------------------------------------------------------------------------

def evaluate_performance(
    result: dict[str, Any],
    config: Any = None,
    eval_times_min: list[float] | None = None,
    T_init: float = 20.0,
    insulation_limit: float = 160.0,
) -> dict[str, dict]:
    """耐火・遮熱性能を評価して判定辞書を返す。

    Parameters
    ----------
    result : dict
        solve() または run_from_config() の返り値。
    config : CLTConfig or None
        設定オブジェクト。指定すると設定値を優先して使用する。
    eval_times_min : list[float] or None
        評価時刻 [分]。None の場合は [60, 75, 90] を使用。
    T_init : float
        初期温度 [°C]。config が None の場合に使用。
    insulation_limit : float
        遮熱基準温度 [°C]（非加熱面の絶対上限）。config が None の場合に使用。

    Returns
    -------
    dict[str, dict]
        {
          "60min": {
            "char_depth_mm": float,
            "unheated_face_temp_C": float,
            "unheated_face_rise_K": float,
            "insulation_ok": bool,
          }, ...
        }
    """
    if config is not None:
        eval_times_min = getattr(config.evaluation, "eval_times_min", eval_times_min)
        T_init = getattr(config.boundary.unheated, "T_inf", T_init)
        insulation_limit = getattr(
            config.evaluation, "unheated_face_temp_limit", insulation_limit
        )

    if eval_times_min is None:
        eval_times_min = [60.0, 75.0, 90.0]

    times_min = result["times"] / 60.0
    judgments: dict[str, dict] = {}

    for t_eval in eval_times_min:
        idx = int(np.argmin(np.abs(times_min - t_eval)))
        T_profile = result["temperatures"][idx]
        char_d_mm = float(result["char_depths"][idx]) * 1000.0

        T_unheated = float(T_profile.ravel()[-1])
        delta_T = T_unheated - T_init

        judgments[f"{t_eval:.0f}min"] = {
            "char_depth_mm": round(char_d_mm, 2),
            "unheated_face_temp_C": round(T_unheated, 2),
            "unheated_face_rise_K": round(delta_T, 2),
            "insulation_ok": bool(T_unheated < insulation_limit),
        }

    return judgments


# ---------------------------------------------------------------------------
# レポート生成
# ---------------------------------------------------------------------------

def generate_report(
    result: dict[str, Any],
    config: Any = None,
    out_dir: str | Path = "outputs",
) -> dict[str, Path]:
    """グラフ・CSV・テキストサマリーを一括生成して保存する。

    Parameters
    ----------
    result : dict
        solve() または run_from_config() の返り値。
    config : CLTConfig or None
        設定オブジェクト（タイトルや評価基準に使用）。
    out_dir : str or Path
        出力先ディレクトリ。存在しない場合は自動作成。

    Returns
    -------
    dict[str, Path]
        保存したファイルのパス辞書:
        - "summary_png"  : 3 パネル概要グラフ
        - "charring_png" : 炭化深さ単独グラフ
        - "profiles_png" : 温度分布単独グラフ
        - "surface_png"  : 表面温度単独グラフ
        - "csv"          : 時刻歴 CSV
        - "txt"          : テキストサマリー
    """
    import matplotlib
    matplotlib.use("Agg")  # GUI なし環境でも動作

    from .viz import (
        plot_charring_progress,
        plot_summary_panel,
        plot_surface_history,
        plot_temperature_profiles,
        save_results_csv,
    )
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # ---- 3 パネル概要グラフ ----
    summary_path = out_dir / "summary.png"
    plot_summary_panel(result, config=config, out_path=summary_path)
    plt.close("all")
    paths["summary_png"] = summary_path

    # ---- 炭化深さ単独 ----
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_charring_progress(result, ax=ax)
    beta = compute_charring_rate(result)
    ax.set_title(f"炭化進行  β={beta:.3f} mm/min")
    fig.tight_layout()
    p = out_dir / "charring.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["charring_png"] = p

    # ---- 温度分布単独 ----
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_temperature_profiles(result, ax=ax)
    ax.set_title("厚み方向の温度分布")
    fig.tight_layout()
    p = out_dir / "profiles.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["profiles_png"] = p

    # ---- 表面温度単独 ----
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_surface_history(result, ax=ax)
    ax.set_title("表面温度の時刻歴")
    fig.tight_layout()
    p = out_dir / "surface.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["surface_png"] = p

    # ---- CSV ----
    csv_path = out_dir / "results.csv"
    save_results_csv(result, csv_path)
    paths["csv"] = csv_path

    # ---- テキストサマリー ----
    txt_path = out_dir / "summary.txt"
    _write_text_summary(result, config, txt_path)
    paths["txt"] = txt_path

    return paths


def _write_text_summary(
    result: dict[str, Any],
    config: Any,
    out_path: Path,
) -> None:
    """テキスト形式のサマリーファイルを書き出す。"""
    lines: list[str] = []

    name = ""
    if config is not None and hasattr(config, "specimen"):
        name = config.specimen.name
    lines.append(f"CLT 耐火シミュレーション結果サマリー")
    lines.append(f"試験体: {name}" if name else "試験体: (未指定)")
    lines.append("")

    # 炭化速度
    beta = compute_charring_rate(result)
    lines.append(f"炭化速度 β = {beta:.3f} mm/min  (Eurocode 5 目標: 0.65 mm/min)")
    lines.append("")

    # 性能評価
    judgments = evaluate_performance(result, config=config)
    lines.append("--- 性能評価 ---")
    for t_key, jdg in judgments.items():
        status = "合格" if jdg["insulation_ok"] else "不合格"
        lines.append(
            f"  {t_key}: 炭化深さ={jdg['char_depth_mm']:.1f}mm, "
            f"非加熱面={jdg['unheated_face_temp_C']:.1f}°C "
            f"(+{jdg['unheated_face_rise_K']:.1f}K) "
            f"遮熱性={status}"
        )
    lines.append("")

    # シミュレーション設定
    if config is not None and hasattr(config, "simulation"):
        sim = config.simulation
        lines.append("--- シミュレーション設定 ---")
        lines.append(f"  解析時間: {sim.t_end_min:.0f} 分")
        lines.append(f"  基準時間刻み: {sim.dt_base_s:.1f} s")
        lines.append(f"  Picard 反復: {sim.n_picard} 回")
        lines.append(f"  層あたりセル数: {sim.n_cells_per_layer}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
