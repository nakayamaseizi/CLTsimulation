"""
optimizer_ui.py
===============
【役割】
孔配置最適化タブの Streamlit UI。

【タブ構成】
⚙️ 設定    : 断面サイズ・空洞率・孔タイプを設定して最適化を開始
📊 進捗    : リアルタイム進捗表示
🏆 結果    : パターン比較テーブル・最良パターンの断面図・温度グラフ
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import streamlit as st

from clt_fire_sim.optimizer import run_optimizer
from clt_fire_sim.optimizer.hole_pattern import generate_candidates


# ---------------------------------------------------------------------------
# メインレンダリング関数
# ---------------------------------------------------------------------------

def render_optimizer_tab(config: Any) -> None:
    """最適化タブ全体を描画する。app.py から呼び出す。"""
    opt_state = run_optimizer.get_opt_state()

    sub_settings, sub_progress, sub_result = st.tabs([
        "⚙️ 設定・実行",
        "📊 進捗",
        "🏆 結果比較",
    ])

    with sub_settings:
        _render_settings(config, opt_state)

    with sub_progress:
        _render_progress(opt_state)

    with sub_result:
        _render_results(opt_state, config)


# ---------------------------------------------------------------------------
# 設定・実行タブ
# ---------------------------------------------------------------------------

def _render_settings(config: Any, opt_state: Any) -> None:
    st.markdown("### ⚙️ 孔配置最適化の設定")
    st.caption(
        "CLT 断面に孔・スリットを設けた際の最適配置を探索します。\n"
        "ベースライン（孔なし）＋複数パターンを 3D シミュレーションで評価します。"
    )

    # ── 断面サイズ ───────────────────────────────────────────────────
    st.markdown("#### 断面サイズ（y-z 平面）")
    col1, col2 = st.columns(2)
    Ly_mm = col1.number_input("幅 Ly [mm]", value=600, min_value=100, max_value=3000, step=50)
    Lz_mm = col2.number_input("長さ Lz [mm]", value=1000, min_value=100, max_value=6000, step=50)
    Ly = Ly_mm / 1000.0
    Lz = Lz_mm / 1000.0

    # ── メッシュ解像度 ───────────────────────────────────────────────
    st.markdown("#### メッシュ解像度（y-z 方向）")
    st.caption("粗いほど計算が速い。精度と速度のバランスで選択。")
    mesh_opt = st.select_slider(
        "解像度",
        options=["粗い（速い）", "標準", "細かい（遅い）"],
        value="標準",
    )
    ny_nz_map = {"粗い（速い）": (8, 10), "標準": (12, 16), "細かい（遅い）": (20, 25)}
    ny, nz = ny_nz_map[mesh_opt]
    st.caption(f"→ y: {ny} セル、z: {nz} セル（計 {ny*nz} セル / 断面）")

    # ── 空洞率 ───────────────────────────────────────────────────────
    st.markdown("#### 目標空洞率")
    target_vf = st.slider(
        "断面に占める孔の割合 [%]",
        min_value=5, max_value=40, value=20, step=5,
    ) / 100.0

    # ── 孔タイプ ─────────────────────────────────────────────────────
    st.markdown("#### 探索する孔タイプ")
    cols = st.columns(4)
    use_grid    = cols[0].checkbox("格子円形", value=True)
    use_stagger = cols[1].checkbox("千鳥円形", value=True)
    use_hslot   = cols[2].checkbox("水平スリット", value=True)
    use_vslot   = cols[3].checkbox("縦スリット", value=True)

    hole_types = []
    if use_grid:    hole_types.append("grid_circle")
    if use_stagger: hole_types.append("staggered_circle")
    if use_hslot:   hole_types.append("h_slot")
    if use_vslot:   hole_types.append("v_slot")

    if not hole_types:
        st.warning("少なくとも 1 つの孔タイプを選択してください。")
        return

    # 候補パターン数のプレビュー
    with st.spinner("候補パターンを生成中..."):
        preview_patterns = generate_candidates(
            ny, nz, Ly, Lz,
            target_vf=target_vf,
            hole_types=hole_types,
            max_candidates=10,
        )
    n_patterns = len(preview_patterns)
    st.info(
        f"📐 候補パターン: **{n_patterns} 種類**"
        f"（ベースライン含め計 {n_patterns + 1} 回シミュレーション）\n\n"
        f"加熱時間 {config.simulation.t_end_min:.0f} 分の解析 × {n_patterns + 1} 回"
    )

    # 候補プレビュー（小さな断面図）
    with st.expander("候補パターンのプレビュー", expanded=False):
        _show_pattern_grid(preview_patterns, Ly, Lz)

    # ── 実行ボタン ───────────────────────────────────────────────────
    st.divider()
    if opt_state.status == "running":
        if st.button("⏹️ 中断", use_container_width=True, type="secondary"):
            run_optimizer.stop_optimization()
            st.rerun()
    else:
        if opt_state.status in ("done", "stopped", "error"):
            if st.button("🔄 リセット", use_container_width=True):
                run_optimizer.reset_opt_state()
                st.rerun()

        run_disabled = opt_state.status == "running"
        if st.button(
            "🔍 最適化を開始",
            type="primary",
            use_container_width=True,
            disabled=run_disabled,
        ):
            # セッション状態に設定を保存
            st.session_state["opt_Ly"] = Ly
            st.session_state["opt_Lz"] = Lz
            st.session_state["opt_ny"] = ny
            st.session_state["opt_nz"] = nz
            st.session_state["opt_patterns"] = preview_patterns

            run_optimizer.start_optimization(
                config=config,
                patterns=preview_patterns,
                panel_Ly=Ly,
                panel_Lz=Lz,
                ny=ny,
                nz=nz,
            )
            st.rerun()


# ---------------------------------------------------------------------------
# 進捗タブ
# ---------------------------------------------------------------------------

def _render_progress(opt_state: Any) -> None:
    if opt_state.status == "idle":
        st.info("「⚙️ 設定・実行」タブで最適化を開始してください。")
        return

    # 全体進捗
    total_done = opt_state.completed
    total_all = opt_state.total
    overall = total_done / total_all if total_all > 0 else 0.0
    st.progress(overall, text=f"全体: {total_done} / {total_all} パターン完了")

    # 現在のパターン進捗
    if opt_state.status == "running":
        st.markdown(f"**現在実行中:** {opt_state.current_name}")
        st.progress(
            opt_state.progress_frac,
            text=f"{opt_state.progress_frac * 100:.0f}%",
        )

        mc1, mc2 = st.columns(2)
        mc1.metric("完了パターン", f"{total_done} / {total_all}")
        mc2.metric("経過時間", f"{opt_state.elapsed_s:.0f} 秒")

        import time
        time.sleep(0.5)
        st.rerun()

    elif opt_state.status == "done":
        st.success(f"✅ 最適化完了！　計算時間: {opt_state.elapsed_s:.0f} 秒")
        st.info("「🏆 結果比較」タブで結果を確認してください。")

    elif opt_state.status == "stopped":
        st.warning("⏹️ 最適化を中断しました。")

    elif opt_state.status == "error":
        st.error(f"❌ エラー: {opt_state.error_msg}")

    # 途中経過テーブル
    if opt_state.results:
        st.markdown("#### 完了済みパターンの暫定結果")
        _show_comparison_table(opt_state)


# ---------------------------------------------------------------------------
# 結果比較タブ
# ---------------------------------------------------------------------------

def _render_results(opt_state: Any, config: Any) -> None:
    if opt_state.status not in ("done", "stopped") or not opt_state.results:
        st.info("最適化が完了すると結果がここに表示されます。")
        return

    import plotly.graph_objects as go

    Ly = st.session_state.get("opt_Ly", 0.6)
    Lz = st.session_state.get("opt_Lz", 1.0)

    # ── 比較テーブル ──────────────────────────────────────────────
    st.markdown("### 📊 全パターン比較")
    _show_comparison_table(opt_state)

    # ── 最良パタームの断面図 ─────────────────────────────────────
    best_idx = opt_state.best_idx
    if best_idx < 0 or best_idx >= len(opt_state.results):
        st.warning("有効な結果が得られませんでした。")
        return

    best = opt_state.results[best_idx]
    baseline = opt_state.baseline

    st.markdown("---")
    st.markdown(f"### 🏆 最良パターン: **{best.pattern.name}**")
    st.caption(best.pattern.description)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 断面図（孔の配置）")
        fig_cross = _make_cross_section_fig(best.pattern, Ly, Lz)
        st.plotly_chart(fig_cross, use_container_width=True)
        st.metric("空洞率", f"{best.pattern.volume_fraction * 100:.1f} %")

    with col_r:
        st.markdown("#### 性能比較")
        if baseline:
            delta_T = best.T_unheated_final_C - baseline.T_unheated_final_C
            delta_char = best.char_depth_final_mm - baseline.char_depth_final_mm
            st.metric(
                "非加熱面温度（最終）",
                f"{best.T_unheated_final_C:.1f} °C",
                delta=f"{delta_T:+.1f} K",
                delta_color="inverse",
            )
            st.metric(
                "炭化深さ（最終）",
                f"{best.char_depth_final_mm:.1f} mm",
                delta=f"{delta_char:+.1f} mm",
                delta_color="inverse",
            )
        else:
            st.metric("非加熱面温度", f"{best.T_unheated_final_C:.1f} °C")
            st.metric("炭化深さ", f"{best.char_depth_final_mm:.1f} mm")

    # ── 炭化深さ時刻歴の比較グラフ ────────────────────────────────
    if baseline and baseline.result and best.result:
        st.markdown("---")
        st.markdown("### 📈 炭化深さ比較（ベースライン vs 最良）")
        fig = go.Figure()

        for pr, color, dash, label in [
            (baseline, "royalblue", "solid",   "ベースライン（孔なし）"),
            (best,     "firebrick", "dash",    f"最良: {best.pattern.name}"),
        ]:
            if pr.result:
                t_min = pr.result["times"] / 60.0
                char_mm = pr.result["char_depths"] * 1000.0
                fig.add_trace(go.Scatter(
                    x=t_min.tolist(), y=char_mm.tolist(),
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2.5, dash=dash),
                    hovertemplate="時刻: %{x:.1f}分<br>炭化深さ: %{y:.2f}mm<extra></extra>",
                ))

        fig.update_layout(
            xaxis_title="時間 [分]",
            yaxis_title="炭化深さ [mm]",
            height=380,
            hovermode="x unified",
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 非加熱面温度の比較
        st.markdown("### 🌡️ 非加熱面温度比較")
        fig2 = go.Figure()

        fig2.add_hline(
            y=160, line_dash="dot", line_color="blue",
            annotation_text="遮熱基準 160°C",
            annotation_position="bottom right",
        )

        for pr, color, dash, label in [
            (baseline, "royalblue", "solid",   "ベースライン（孔なし）"),
            (best,     "firebrick", "dash",    f"最良: {best.pattern.name}"),
        ]:
            if pr.result:
                t_min = pr.result["times"] / 60.0
                T_mat = pr.result["temperatures"]
                mesh = pr.result.get("mesh")
                if mesh is not None:
                    nx, ny2, nz2 = mesh.nx, mesh.ny, mesh.nz
                    T_unheated = np.array([
                        T_mat[ti].reshape(nx, ny2, nz2)[-1, :, :].mean()
                        for ti in range(len(t_min))
                    ])
                else:
                    T_unheated = T_mat[:, -1]

                fig2.add_trace(go.Scatter(
                    x=t_min.tolist(), y=T_unheated.tolist(),
                    mode="lines", name=label,
                    line=dict(color=color, width=2.5, dash=dash),
                    hovertemplate="時刻: %{x:.1f}分<br>非加熱面: %{y:.1f}°C<extra></extra>",
                ))

        fig2.update_layout(
            xaxis_title="時間 [分]",
            yaxis_title="非加熱面温度 [°C]",
            height=380,
            hovermode="x unified",
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── ビューアに送るボタン ─────────────────────────────────────
    if best.result:
        if st.button(
            "🔬 この結果をビューアで詳しく見る",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["viewer_result"] = best.result
            st.session_state["viewer_label"] = (
                f"最適化結果: {best.pattern.name}"
            )
            st.success("✅ ビューアに読み込みました。「🔬 ビューア」タブを開いてください。")


# ---------------------------------------------------------------------------
# ヘルパー描画関数
# ---------------------------------------------------------------------------

def _show_comparison_table(opt_state: Any) -> None:
    """全パターンの比較テーブルを表示する。"""
    import pandas as pd

    rows = []
    # ベースライン行
    if opt_state.baseline:
        bl = opt_state.baseline
        rows.append({
            "パターン": "ベースライン（孔なし）",
            "空洞率 [%]": "0.0",
            "非加熱面温度 [°C]": (
                f"{bl.T_unheated_final_C:.1f}" if not math.isnan(bl.T_unheated_final_C) else "—"
            ),
            "炭化深さ [mm]": (
                f"{bl.char_depth_final_mm:.1f}" if not math.isnan(bl.char_depth_final_mm) else "—"
            ),
            "ベースライン比 ΔT [K]": "—",
            "評価": "",
        })

    bl_T = (
        opt_state.baseline.T_unheated_final_C
        if opt_state.baseline and not math.isnan(opt_state.baseline.T_unheated_final_C)
        else float("nan")
    )

    for i, pr in enumerate(opt_state.results):
        if pr.error:
            rows.append({
                "パターン": pr.pattern.name,
                "空洞率 [%]": f"{pr.pattern.volume_fraction * 100:.1f}",
                "非加熱面温度 [°C]": "エラー",
                "炭化深さ [mm]": "—",
                "ベースライン比 ΔT [K]": "—",
                "評価": "❌",
            })
            continue

        delta_str = "—"
        if not math.isnan(bl_T) and not math.isnan(pr.T_unheated_final_C):
            delta = pr.T_unheated_final_C - bl_T
            delta_str = f"{delta:+.1f}"

        medal = ""
        if i == opt_state.best_idx:
            medal = "🏆"

        rows.append({
            "パターン": f"{medal} {pr.pattern.name}",
            "空洞率 [%]": f"{pr.pattern.volume_fraction * 100:.1f}",
            "非加熱面温度 [°C]": (
                f"{pr.T_unheated_final_C:.1f}" if not math.isnan(pr.T_unheated_final_C) else "—"
            ),
            "炭化深さ [mm]": (
                f"{pr.char_depth_final_mm:.1f}" if not math.isnan(pr.char_depth_final_mm) else "—"
            ),
            "ベースライン比 ΔT [K]": delta_str,
            "評価": medal,
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.drop(columns=["評価"]), use_container_width=True, hide_index=True)


def _make_cross_section_fig(pattern: Any, Ly: float, Lz: float) -> Any:
    """孔パターンの y-z 断面図を Plotly で描画する。"""
    import plotly.graph_objects as go

    void_yz = pattern.void_yz  # (ny, nz) bool
    ny, nz = void_yz.shape
    dy = Ly / ny * 1000   # mm
    dz = Lz / nz * 1000

    # 木材セル（白）と孔セル（水色）
    z_matrix = void_yz.astype(float)  # 0=木材, 1=空洞

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix.tolist(),
        x=(np.arange(nz) * dz + dz / 2).tolist(),
        y=(np.arange(ny) * dy + dy / 2).tolist(),
        colorscale=[[0, "#8B4513"], [1, "#87CEEB"]],   # 茶色=木材, 水色=空洞
        showscale=False,
        hovertemplate="y: %{y:.0f}mm<br>z: %{x:.0f}mm<br>%{customdata}<extra></extra>",
        customdata=np.where(void_yz, "孔（空気）", "木材").tolist(),
        zmin=0, zmax=1,
    ))

    fig.update_layout(
        xaxis_title="z [mm]",
        yaxis_title="y [mm]",
        height=300,
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis=dict(range=[0, Lz * 1000]),
        yaxis=dict(range=[0, Ly * 1000]),
    )
    return fig


def _show_pattern_grid(patterns: list, Ly: float, Lz: float) -> None:
    """候補パターンのプレビューグリッドを表示する。"""
    n = len(patterns)
    cols = st.columns(min(n, 5))
    for i, (col, pat) in enumerate(zip(cols, patterns[:5])):
        with col:
            st.caption(f"**{pat.name}**\nvf={pat.volume_fraction*100:.0f}%")
            fig = _make_cross_section_fig(pat, Ly, Lz)
            fig.update_layout(height=150, margin=dict(l=5, r=5, t=5, b=20))
            st.plotly_chart(fig, use_container_width=True)
