"""
pareto_ui.py
============
パレート最適化タブの UI モジュール。

孔径 d・ピッチ p・ラミナ厚 t_lam・枚数 n_lam を設計変数として
耐火性能（CLT面温度↓）と断熱性能（断熱抵抗R↑）のパレートフロントを求める。
"""

from __future__ import annotations

import numpy as np
import streamlit as st


def render_pareto_tab() -> None:
    """パレート最適化タブを描画する。"""
    from clt_fire_sim.optimizer.pareto_optimizer import (
        get_default_d_list,
        get_default_p_list,
        get_pareto_state,
        start_pareto_optimization,
        stop_pareto,
        _compute_vf,
        _generate_candidates,
    )

    st.header("🎯 パレート最適化（孔パターン×ラミナ構成）")
    st.caption(
        "耐火性能（CLT面温度 最小化）と断熱性能（断熱抵抗R 最大化）を同時に満たす"
        " パレート最適解を探索します。"
    )

    # ──────────────────────────────────────────────────────────────
    # パラメータ設定 UI
    # ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ 探索パラメータ設定", expanded=True):
        col_d, col_p, col_t = st.columns(3)

        with col_d:
            st.markdown("**🔵 孔径 d [mm]**")
            d_presets = [0, 6, 12, 18, 24, 30, 36, 40]
            d_selected = st.multiselect(
                "孔径候補を選択",
                d_presets,
                default=[0, 6, 12, 18, 24, 30],
                key="pareto_d_list",
            )
            d_list = sorted(float(d) for d in d_selected) if d_selected else [0.0]
            st.caption(f"候補: {d_list} mm")

        with col_p:
            st.markdown("**📐 ピッチ p [mm]**")
            p_presets = [30, 40, 50, 60, 80, 100]
            p_selected = st.multiselect(
                "ピッチ候補を選択", p_presets,
                default=[30, 40, 50, 60, 80, 100],
                key="pareto_p_list",
            )
            p_list = sorted(float(p) for p in p_selected) if p_selected else [50.0]
            st.caption(f"候補: {p_list} mm")

        with col_t:
            st.markdown("**📏 ラミナ構成**")
            use_12 = st.checkbox("12mm ラミナ使用", value=True, key="pareto_use_12")
            use_24 = st.checkbox("24mm ラミナ使用", value=True, key="pareto_use_24")
            t_lam_list = []
            if use_12:
                t_lam_list.append(12.0)
            if use_24:
                t_lam_list.append(24.0)
            if not t_lam_list:
                t_lam_list = [12.0]

            n_lam_max = st.slider("最大層数", 1, 8, 8, key="pareto_n_lam_max")
            max_total = max(t * n_lam_max for t in t_lam_list)
            total_options = sorted(set(
                t * n
                for t in t_lam_list
                for n in range(1, n_lam_max + 1)
                if t * n <= 96
            ))
            st.caption(f"総厚の候補 [mm]: {[int(x) for x in total_options]}")

        # 候補数と所要時間のプレビュー
        try:
            preview_candidates = _generate_candidates(d_list, p_list, t_lam_list, n_lam_max)
            unique_sims = len(set(c.r_key for c in preview_candidates))
            est_sec = unique_sims * 1.5  # ~1.5秒/ケース（実測値）
            est_min = est_sec / 60
            time_str = f"{est_sec:.0f} 秒" if est_sec < 120 else f"約 {est_min:.1f} 分"
            st.info(
                f"設計変数の組み合わせ: **{len(preview_candidates)} 通り**  |  "
                f"実際のシミュレーション回数: **{unique_sims} 回**（重複除去後）  |  "
                f"推定所要時間: **{time_str}**"
            )
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    # CLT ベース設定
    # ──────────────────────────────────────────────────────────────
    with st.expander("🌲 CLT ベース設定", expanded=False):
        st.caption("保護層の下に敷かれる構造用 CLT の設定です。")
        base_mat = st.selectbox(
            "保護層材料",
            ["sugi", "hinoki", "lauan", "douglas_fir"],
            index=0,
            key="pareto_base_mat",
        )
        base_rho = st.number_input(
            "保護層密度 [kg/m³]", value=400.0, min_value=100.0, max_value=1000.0,
            step=10.0, key="pareto_base_rho",
        )
        t_end_min = st.number_input(
            "加熱時間 [分]", value=60.0, min_value=30.0, max_value=120.0,
            step=10.0, key="pareto_t_end",
        )

    # ──────────────────────────────────────────────────────────────
    # 実行・中断ボタン
    # ──────────────────────────────────────────────────────────────
    state = get_pareto_state()

    col_run, col_stop = st.columns([4, 1])
    with col_run:
        run_disabled = state.status == "running" or not t_lam_list or not p_list
        if st.button("🚀 パレート最適化を実行", disabled=run_disabled, use_container_width=True):
            start_pareto_optimization(
                d_list=d_list,
                p_list=p_list,
                t_lam_list=t_lam_list,
                n_lam_max=n_lam_max,
                base_mat=base_mat,
                base_rho=base_rho,
                n_cells=10,
                t_end_min=t_end_min,
            )
            st.rerun()
    with col_stop:
        if st.button("⏹ 中断", disabled=(state.status != "running"), use_container_width=True):
            stop_pareto()

    # ──────────────────────────────────────────────────────────────
    # 進捗表示
    # ──────────────────────────────────────────────────────────────
    if state.status == "running":
        prog = state.progress
        elapsed = state.elapsed_s
        est = state.est_remaining_s
        st.progress(prog, f"計算中: {state.done}/{state.total} ケース完了")
        col_e, col_r = st.columns(2)
        col_e.metric("経過時間", f"{elapsed:.0f} 秒")
        col_r.metric("残り推定", f"{est:.0f} 秒" if est > 0 else "計算中...")
        st.rerun()

    # ──────────────────────────────────────────────────────────────
    # 結果表示
    # ──────────────────────────────────────────────────────────────
    if state.status in ("done", "stopped") and state.candidates:
        _render_pareto_results(state)


# ---------------------------------------------------------------------------
# 結果描画
# ---------------------------------------------------------------------------

# 孔径 d → 色のマッピング（固定カラーパレット）
_D_COLORS: dict[float, str] = {
    0.0:  "#90A4AE",  # 無孔 = グレー
    6.0:  "#42A5F5",  # 水色
    12.0: "#1565C0",  # 青
    14.0: "#26C6DA",  # シアン
    18.0: "#66BB6A",  # 緑
    24.0: "#FFA726",  # オレンジ
    30.0: "#EF5350",  # 赤
    36.0: "#AB47BC",  # 紫
    40.0: "#8D6E63",  # ブラウン
}

# ラミナ厚 → Plotly マーカー形状
_T_SYMBOLS: dict[float, str] = {12.0: "circle", 24.0: "square"}


def _render_pareto_results(state) -> None:
    """フィルタ付きパレートフロントの可視化と解一覧テーブル。"""
    import plotly.graph_objects as go
    import pandas as pd

    all_cands = [c for c in state.candidates if c.T_clt_60 < float("inf") and not c.error]
    pareto_pts = state.pareto_front

    if not all_cands:
        st.warning("有効な候補がありません。")
        return

    st.success(
        f"最適化完了: {len(all_cands)} 候補を評価 → "
        f"パレート最適解 **{len(pareto_pts)} 点** を特定"
    )

    # ユニーク値を取得
    unique_d = sorted(set(c.d_mm for c in all_cands))
    unique_p = sorted(set(c.p_mm for c in all_cands))
    unique_t = sorted(set(c.t_lam_mm for c in all_cands))

    # ─── 表示フィルタ チェックボックス ─────────────────────────────
    with st.expander("🔍 グラフ表示フィルタ", expanded=True):
        col_fd, col_fp, col_ft = st.columns(3)

        with col_fd:
            st.markdown("**🔵 孔径 d [mm]**")
            # 全選択 / 全解除
            all_d_on = st.checkbox("すべて選択", value=True, key="filter_d_all")
            st.divider()
            sel_d: dict[float, bool] = {}
            for d in unique_d:
                label = f"d = {int(d)} mm" + ("  （無孔）" if d == 0 else "")
                color = _D_COLORS.get(d, "#999")
                sel_d[d] = st.checkbox(
                    label, value=all_d_on, key=f"filter_d_{int(d)}"
                )

        with col_fp:
            st.markdown("**📐 ピッチ p [mm]**")
            all_p_on = st.checkbox("すべて選択", value=True, key="filter_p_all")
            st.divider()
            sel_p: dict[float, bool] = {}
            for p in unique_p:
                sel_p[p] = st.checkbox(
                    f"p = {int(p)} mm", value=all_p_on, key=f"filter_p_{int(p)}"
                )

        with col_ft:
            st.markdown("**📏 ラミナ厚 [mm]**")
            all_t_on = st.checkbox("すべて選択", value=True, key="filter_t_all")
            st.divider()
            sel_t: dict[float, bool] = {}
            for t in unique_t:
                sym = _T_SYMBOLS.get(t, "circle")
                sym_icon = "●" if sym == "circle" else "■"
                sel_t[t] = st.checkbox(
                    f"{sym_icon}  {int(t)} mm ラミナ", value=all_t_on, key=f"filter_t_{int(t)}"
                )

    # フィルタ適用
    def _visible(c) -> bool:
        return (
            sel_d.get(c.d_mm, True)
            and sel_p.get(c.p_mm, True)
            and sel_t.get(c.t_lam_mm, True)
        )

    filtered_all = [c for c in all_cands if _visible(c)]
    filtered_pareto = [c for c in pareto_pts if _visible(c)]

    # ─── Plotlyグラフ ───────────────────────────────────────────────
    fig = go.Figure()

    # 全体パレートフロントの参照線（常時表示・グレー破線）
    if pareto_pts:
        pf_sorted = sorted(pareto_pts, key=lambda c: c.R_value)
        fig.add_trace(go.Scatter(
            x=[c.R_value for c in pf_sorted],
            y=[c.T_clt_60 for c in pf_sorted],
            mode="lines",
            line=dict(color="rgba(160,160,160,0.5)", width=1.5, dash="dash"),
            name="パレートフロント（全体参考）",
            hoverinfo="skip",
        ))

    # 孔径ごとにトレースを分けて描画（ラミナ厚でシンボルを変える）
    for d in unique_d:
        if not sel_d.get(d, True):
            continue
        d_cands = [c for c in filtered_all if c.d_mm == d]
        if not d_cands:
            continue

        color = _D_COLORS.get(d, "#888888")
        d_label = f"d={int(d)}mm" + ("（無孔）" if d == 0 else "")

        symbols = [_T_SYMBOLS.get(c.t_lam_mm, "circle") for c in d_cands]
        # 点のサイズ = 総厚に比例（最小6、最大16）
        sizes = [6 + (c.total_mm - 12) / (96 - 12) * 10 for c in d_cands]

        fig.add_trace(go.Scatter(
            x=[c.R_value for c in d_cands],
            y=[c.T_clt_60 for c in d_cands],
            mode="markers",
            marker=dict(
                color=color,
                symbol=symbols,
                size=sizes,
                opacity=0.70,
                line=dict(width=0.8, color="white"),
            ),
            name=d_label,
            legendgroup=f"d_{int(d)}",
            customdata=np.column_stack([
                [c.d_mm for c in d_cands],
                [c.p_mm for c in d_cands],
                [c.vf for c in d_cands],
                [c.total_mm for c in d_cands],
                [c.t_lam_mm for c in d_cands],
                [c.n_lam for c in d_cands],
            ]),
            hovertemplate=(
                f"<b>{d_label}</b><br>"
                "p=%{customdata[1]:.0f}mm | vf=%{customdata[2]:.3f}<br>"
                "総厚=%{customdata[3]:.0f}mm "
                "(%{customdata[4]:.0f}mm×%{customdata[5]:.0f}枚)<br>"
                "R=%{x:.3f} m²K/W | CLT面温度=%{y:.1f}°C"
                "<extra></extra>"
            ),
        ))

    # 表示中のパレート最適解（赤い星）
    if filtered_pareto:
        fp_sorted = sorted(filtered_pareto, key=lambda c: c.R_value)
        fig.add_trace(go.Scatter(
            x=[c.R_value for c in fp_sorted],
            y=[c.T_clt_60 for c in fp_sorted],
            mode="markers+lines",
            marker=dict(
                color="crimson", size=14, symbol="star",
                line=dict(width=1, color="darkred"),
            ),
            line=dict(color="crimson", width=1.5),
            name="★ パレート最適解（表示中）",
            customdata=np.column_stack([
                [c.d_mm for c in fp_sorted],
                [c.p_mm for c in fp_sorted],
                [c.vf for c in fp_sorted],
                [c.total_mm for c in fp_sorted],
                [c.t_lam_mm for c in fp_sorted],
                [c.n_lam for c in fp_sorted],
            ]),
            hovertemplate=(
                "<b>★ パレート最適解</b><br>"
                "d=%{customdata[0]:.0f}mm / p=%{customdata[1]:.0f}mm<br>"
                "vf=%{customdata[2]:.3f} | 総厚=%{customdata[3]:.0f}mm<br>"
                "ラミナ %{customdata[4]:.0f}mm×%{customdata[5]:.0f}枚<br>"
                "R=%{x:.3f} m²K/W | CLT面温度=%{y:.1f}°C"
                "<extra></extra>"
            ),
        ))

    # ラミナ厚のシンボル凡例（ダミートレース）
    for t in unique_t:
        sym = _T_SYMBOLS.get(t, "circle")
        icon = "●" if sym == "circle" else "■"
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color="gray", symbol=sym, size=9),
            name=f"{icon} {int(t)}mm ラミナ（シンボル）",
            showlegend=True,
            legendgroup=f"t_{int(t)}",
        ))

    # 100°C 基準線
    if all_cands:
        r_min = min(c.R_value for c in all_cands)
        r_max = max(c.R_value for c in all_cands)
        fig.add_shape(
            type="line", x0=r_min, x1=r_max, y0=100, y1=100,
            line=dict(color="darkorange", width=1.5, dash="dash"),
        )
        fig.add_annotation(
            x=r_max, y=100, text="100°C 基準（60分準耐火）",
            showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(color="darkorange", size=11),
        )

    fig.update_layout(
        title="パレートフロント：断熱性能 vs 耐火性能",
        xaxis_title="断熱抵抗 R [m²·K/W]（大きいほど断熱性能↑）",
        yaxis_title="CLT面温度@60分 [°C]（小さいほど耐火性能↑）",
        height=560,
        legend=dict(
            x=1.02, y=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgray", borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(r=220),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🎨 **色**: 孔径 d  |  **● 丸**: 12mmラミナ  |  **■ 四角**: 24mmラミナ  "
        "|  **点の大きさ**: 総厚（大 = 厚い）  |  **★**: パレート最適解"
    )

    # ─── パレート最適解テーブル ─────────────────────────────────────
    show_all_pareto = st.toggle("フィルタに関わらず全パレート解を表示", value=False)
    display_pareto = pareto_pts if show_all_pareto else filtered_pareto

    if display_pareto:
        st.subheader("✅ パレート最適解一覧")
        st.caption(
            "★ = 断熱性能・耐火性能の両方で他のどの候補にも劣らない設計。"
            "CLT面温度の低い順（耐火性能の高い順）に並んでいます。"
        )
        rows = []
        for c in sorted(display_pareto, key=lambda x: x.T_clt_60):
            rows.append({
                "孔径 d [mm]": int(c.d_mm),
                "ピッチ p [mm]": int(c.p_mm),
                "空洞率 vf": f"{c.vf:.3f}",
                "ラミナ厚 [mm]": int(c.t_lam_mm),
                "枚数": int(c.n_lam),
                "総厚 [mm]": int(c.total_mm),
                "CLT面温度@60分 [°C]": f"{c.T_clt_60:.1f}",
                "断熱抵抗 R [m²K/W]": f"{c.R_value:.3f}",
                "物性モデル": "池畑(2021)" if c.use_ikehata_model() else "並列混合則",
            })
        df_pareto = pd.DataFrame(rows)
        st.dataframe(df_pareto, hide_index=True, use_container_width=True)
        st.download_button(
            "💾 パレート解 CSV をダウンロード",
            data=df_pareto.to_csv(index=False).encode("utf-8-sig"),
            file_name="pareto_front.csv",
            mime="text/csv",
        )
    elif not show_all_pareto:
        st.info("表示フィルタに一致するパレート最適解がありません。フィルタ条件を緩めてください。")

    # ─── 全候補テーブル ─────────────────────────────────────────────
    with st.expander("📋 全候補一覧（フィルタ適用中）", expanded=False):
        rows_all = []
        for c in sorted(filtered_all, key=lambda x: (x.T_clt_60, -x.R_value)):
            rows_all.append({
                "孔径 d [mm]": int(c.d_mm),
                "ピッチ p [mm]": int(c.p_mm),
                "空洞率 vf": f"{c.vf:.3f}",
                "ラミナ厚 [mm]": int(c.t_lam_mm),
                "枚数": int(c.n_lam),
                "総厚 [mm]": int(c.total_mm),
                "CLT面温度@60分 [°C]": f"{c.T_clt_60:.1f}",
                "断熱抵抗 R [m²K/W]": f"{c.R_value:.3f}",
                "★パレート": "★" if c.is_pareto else "",
            })
        if rows_all:
            st.dataframe(pd.DataFrame(rows_all), hide_index=True, use_container_width=True)
        else:
            st.info("フィルタ条件に一致する候補がありません。")

    # ─── エラー一覧 ──────────────────────────────────────────────────
    errors = [c for c in state.candidates if c.error]
    if errors:
        with st.expander(f"⚠️ エラー ({len(errors)} 件)", expanded=False):
            for c in errors[:10]:
                st.text(f"d={c.d_mm}mm, p={c.p_mm}mm, {c.t_lam_mm}mm×{c.n_lam}枚: {c.error}")
