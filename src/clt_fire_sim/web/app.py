"""
app.py
======
【役割】
CLT 耐火シミュレータ Web UI の Streamlit エントリポイント。

【起動方法】
    uv run streamlit run src/clt_fire_sim/web/app.py

【タブ構成】
    📋 設定確認  : YAML プレビュー・設定サマリー（Phase 7.1）
    ▶️ 解析実行  : 解析開始・進捗表示・中断ボタン（Phase 7.2）
    📊 結果      : 評価結果・ダウンロード（Phase 7.2）
    🔬 ビューア  : Plotly インタラクティブ可視化（Phase 7.3）
    📂 履歴      : 比較テーブル・ビューア連携・削除（Phase 7.4）
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# パス設定（streamlit run で直接起動する場合に必要）
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent.parent.parent.parent  # → .../clt_fire_sim/src
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st

# ---------------------------------------------------------------------------
# ページ設定（必ず最初に呼ぶ）
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CLT 耐火シミュレータ",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**CLT 耐火シミュレータ v0.2**\n\n"
            "Eurocode 5 (EN 1995-1-2) 準拠の 1D/3D 非定常熱伝導シミュレーション。\n"
            "建築基準法 60/75/90/120 分耐火評価向け。"
        ),
    },
)

# ---------------------------------------------------------------------------
# 内部モジュールのインポート
# ---------------------------------------------------------------------------
from clt_fire_sim.materials import MATERIAL_DB
from clt_fire_sim.web import config_editor
from clt_fire_sim.web import runner as web_runner
from clt_fire_sim.web import system_info

# ---------------------------------------------------------------------------
# セッション状態の初期化
# ---------------------------------------------------------------------------
config_editor.init_session_state()

# ビューアで表示する結果（None = 最新の解析結果を使用）
if "viewer_result" not in st.session_state:
    st.session_state["viewer_result"] = None
if "viewer_label" not in st.session_state:
    st.session_state["viewer_label"] = None

# 削除確認ダイアログ用
if "confirm_delete_run_id" not in st.session_state:
    st.session_state["confirm_delete_run_id"] = None

# ---------------------------------------------------------------------------
# サイドバー：設定フォーム
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔥 CLT 耐火シミュレータ")
    st.caption("v0.4 — Phase 7.4")
    st.divider()
    config = config_editor.render_sidebar()

# ---------------------------------------------------------------------------
# メインエリア：タブ構成
# ---------------------------------------------------------------------------
st.title("CLT 耐火シミュレーション")

tab_config, tab_run, tab_result, tab_3d, tab_opt, tab_pareto, tab_history = st.tabs([
    "📋 設定確認",
    "▶️ 解析実行",
    "📊 結果",
    "🔬 ビューア",
    "🔍 孔最適化",
    "🎯 パレート最適化",
    "📂 履歴",
])

# ===========================================================================
# タブ1：設定確認
# ===========================================================================
with tab_config:
    _n_layers = len(config.specimen.layers)
    _total_mm = sum(layer.thickness_mm for layer in config.specimen.layers)

    # ── 試験体断面図 ────────────────────────────────────────────────────────
    st.markdown("### 試験体断面図")
    try:
        _fig_diagram = viz_plotly.make_specimen_diagram(config)
        st.plotly_chart(_fig_diagram, use_container_width=True)
    except Exception as _diag_err:
        st.warning(f"断面図の生成に失敗しました: {_diag_err}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 試験体")
        st.metric("試験体名", config.specimen.name)
        st.metric("総厚", f"{_total_mm:.0f} mm")
        st.metric("層数", f"{_n_layers} 層")

        import pandas as pd
        layer_rows = []
        for i, layer in enumerate(config.specimen.layers):
            mat_name = MATERIAL_DB.get(layer.material, {}).get("name", layer.material)
            face = "🔥 加熱面" if i == 0 else ("❄️ 非加熱面" if i == _n_layers - 1 else "")
            layer_rows.append({
                "層": f"第{i+1}層 {face}",
                "材料": mat_name,
                "厚さ [mm]": layer.thickness_mm,
                "密度 [kg/m³]": layer.rho_0_kg_m3,
                "含水率 [%]": f"{layer.moisture_content*100:.0f}",
            })
        st.dataframe(pd.DataFrame(layer_rows), hide_index=True)

    with col_right:
        st.markdown("### 解析条件")
        st.metric("解析モード", st.session_state.get("analysis_mode", "1D"))
        st.metric("加熱時間", f"{config.simulation.t_end_min:.0f} 分")
        st.metric("初期温度", f"{config.boundary.unheated.T_inf:.1f} °C")
        st.metric("メッシュ", f"{config.simulation.n_cells_per_layer} セル/層")
        eval_badges = " ".join(f"`{t:.0f} 分`" for t in config.evaluation.eval_times_min)
        st.markdown(f"**評価時刻**: {eval_badges}")

    st.markdown("### YAML プレビュー")
    yaml_str = config_editor.config_to_yaml_str(config)
    st.code(yaml_str, language="yaml", line_numbers=True)
    st.download_button(
        "💾 YAML をダウンロード",
        data=yaml_str.encode("utf-8"),
        file_name=f"{config.specimen.name or 'clt_config'}.yaml",
        mime="text/yaml",
    )

# ===========================================================================
# タブ2：解析実行
# ===========================================================================
with tab_run:
    state = web_runner.get_state()

    # ── PC スペック表示 ──────────────────────────────────────────────
    with st.expander("💻 PC スペック", expanded=False):
        try:
            info = system_info.get_system_info()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("総メモリ", f"{info['total_ram_gb']:.1f} GB")
            c2.metric("空きメモリ", f"{info['available_ram_gb']:.1f} GB")
            c3.metric("CPU コア数", f"{info['cpu_count']} コア")
            c4.metric("CPU 使用率", f"{info['cpu_pct']:.0f} %")
        except Exception:
            st.caption("psutil によるスペック取得に失敗しました。")

    # ── メモリ警告 ────────────────────────────────────────────────────
    try:
        avail_gb = system_info.get_system_info()["available_ram_gb"]
        est_mem = system_info.estimate_memory_gb(
            n_cells_per_layer=config.simulation.n_cells_per_layer,
            n_layers=len(config.specimen.layers),
            t_end_min=config.simulation.t_end_min,
            mode=st.session_state.get("analysis_mode", "1D"),
        )
        warnings = system_info.check_resources_warning(
            estimated_mem_gb=est_mem,
            available_ram_gb=avail_gb,
            mode=st.session_state.get("analysis_mode", "1D"),
            mesh_option=st.session_state.get("mesh_option", "標準（推奨）"),
        )
        for w in warnings:
            st.warning(w)
        if not warnings:
            st.success(f"✅ 推定メモリ使用量 {est_mem*1000:.0f} MB — 問題ありません。")
    except Exception:
        pass

    # ── 解析設定サマリー ─────────────────────────────────────────────
    st.markdown(
        f"**{config.specimen.name}** | "
        f"{_n_layers} 層 {_total_mm:.0f}mm | "
        f"{config.simulation.t_end_min:.0f} 分解析 | "
        f"1D モード"
    )

    # ── 解析開始 / 中断ボタン ─────────────────────────────────────────
    if state.status in ("idle", "done", "stopped", "error"):
        if st.button(
            "🚀 解析開始",
            type="primary",
            use_container_width=True,
            disabled=(state.status == "running"),
        ):
            outputs_base = Path("outputs")
            web_runner.start_simulation(config, outputs_base=outputs_base)
            st.rerun()

    if state.status == "running":
        if st.button("⏹️ 中断", use_container_width=True, type="secondary"):
            web_runner.stop_simulation()
            st.rerun()

    # ── 進捗表示 ─────────────────────────────────────────────────────
    if state.status == "running":
        st.divider()
        st.markdown("#### ⏳ 解析中...")

        # プログレスバー
        progress_pct = int(state.progress * 100)
        st.progress(state.progress, text=f"{progress_pct}%  ({state.current_time_min:.1f} / {state.t_end_min:.0f} 分)")

        # メトリクス
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("経過時間", f"{state.current_time_min:.1f} 分")
        mc2.metric("最高温度", f"{state.max_temp_C:.0f} °C")
        mc3.metric("炭化深さ", f"{state.char_depth_mm:.1f} mm")
        mc4.metric("残り（目安）", f"{max(0, state.est_remaining_s):.0f} 秒")

        # CPU / メモリ（ライブ）
        try:
            live = system_info.get_live_usage()
            lc1, lc2 = st.columns(2)
            lc1.metric("CPU 使用率", f"{live['cpu_pct']:.0f} %")
            lc2.metric("メモリ使用率", f"{live['ram_used_pct']:.0f} %")
        except Exception:
            pass

        # 自動リフレッシュ（0.5 秒ごとにポーリング）
        import time
        time.sleep(0.5)
        st.rerun()

    elif state.status == "done":
        st.success(f"✅ 解析完了！  計算時間: {state.elapsed_s:.1f} 秒")
        st.info("「📊 結果」タブで結果を確認してください。")

    elif state.status == "stopped":
        st.warning(f"⏹️ 解析を中断しました。（{state.current_time_min:.1f} 分時点）")

    elif state.status == "error":
        st.error(f"❌ エラーが発生しました:\n\n{state.error_msg}")
        st.markdown(
            "**よくある原因**:\n"
            "- メモリ不足 → メッシュを「粗い」にする\n"
            "- 数値発散 → `dt_base` を小さくする（YAML で調整）\n"
        )

    if state.status in ("done", "stopped", "error"):
        if st.button("🔄 リセット（新しい解析を始める）", use_container_width=True):
            web_runner.reset_state()
            st.rerun()

# ===========================================================================
# タブ3：結果
# ===========================================================================
with tab_result:
    state = web_runner.get_state()

    if state.status != "done" or state.result is None:
        st.info("解析が完了するとここに結果が表示されます。「▶️ 解析実行」タブで解析を開始してください。")
    else:
        result = state.result
        evaluation = result.get("evaluation", {})

        # ── 判定結果テーブル ──────────────────────────────────────────
        st.markdown("### 🏆 耐火性能評価結果")
        rows = []
        for t_key, jdg in sorted(evaluation.items()):
            ok = jdg["insulation_ok"]
            rows.append({
                "評価時刻": t_key,
                "遮熱性": "✅ 合格" if ok else "❌ 不合格",
                "炭化深さ [mm]": f"{jdg['char_depth_mm']:.1f}",
                "非加熱面温度 [°C]": f"{jdg['unheated_face_temp_C']:.1f}",
                "温度上昇 [K]": f"{jdg['unheated_face_rise_K']:.1f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        # ── 炭化速度 ──────────────────────────────────────────────────
        from clt_fire_sim.report import compute_charring_rate
        beta = compute_charring_rate(result)
        col_beta, col_ref = st.columns(2)
        col_beta.metric(
            "炭化速度 β",
            f"{beta:.3f} mm/min" if not __import__("math").isnan(beta) else "算出不可",
        )
        col_ref.metric("Eurocode 5 目標", "0.65 mm/min")

        # ── 断熱性能（室温・稳常状態）────────────────────────────────────
        st.markdown("### 🌡️ 断熱性能（室温・稳常状態）")
        _cfg = result.get("config", config)
        if _cfg is not None and hasattr(_cfg, "specimen"):
            try:
                import pandas as _pd_th
                from clt_fire_sim.materials import MATERIAL_DB as _MDB
                _AIR_K_RT = 0.026
                _RSI_APP, _RSE_APP = 0.13, 0.04

                def _lam_rt(lyr) -> float:
                    """LayerConfig から室温熱伝導率 [W/mK] を返す。"""
                    mt = getattr(lyr, "material_type", "wood")
                    if mt == "custom" and getattr(lyr, "k_W_mK", None) is not None:
                        return float(lyr.k_W_mK)
                    db = _MDB.get(getattr(lyr, "material", "sugi"), {})
                    pt = db.get("properties_type", "wood")
                    if pt == "constant":
                        bk = float(db.get("k", 0.12))
                    elif pt == "charred_cork":
                        bk = float(db.get("k", 0.041))
                    else:
                        bk = float(db.get("k_measured", 0.12))
                    if mt in ("perforated_wood", "perforated_wood_advanced", "slitted_wood"):
                        vf = float(getattr(lyr, "void_fraction", 0.0))
                        return (1.0 - vf) * bk + vf * _AIR_K_RT
                    return bk

                th_rows = []
                r_sum = 0.0
                t_sum_m = 0.0
                for i, lyr in enumerate(_cfg.specimen.layers):
                    lam = _lam_rt(lyr)
                    r_lyr = lyr.thickness_mm / 1000.0 / lam
                    r_contact = getattr(lyr, "contact_resistance_m2KW", 0.0)
                    r_sum += r_lyr + r_contact
                    t_sum_m += lyr.thickness_mm / 1000.0
                    mat_disp = getattr(lyr, "custom_name", "") or lyr.material
                    mt = getattr(lyr, "material_type", "wood")
                    if mt in ("perforated_wood", "perforated_wood_advanced", "slitted_wood"):
                        vf_disp = getattr(lyr, "void_fraction", 0.0)
                        mat_disp += f" (vf={vf_disp:.3f})"
                    th_rows.append({
                        "層": i + 1,
                        "材料": mat_disp,
                        "厚さ [mm]": f"{lyr.thickness_mm:.1f}",
                        "λ [W/mK]": f"{lam:.4f}",
                    })
                U_app = 1.0 / (_RSE_APP + r_sum + _RSI_APP)
                lam_eq = t_sum_m / r_sum if r_sum > 0 else 0.0
                st.dataframe(_pd_th.DataFrame(th_rows), hide_index=True, use_container_width=True)
                col_u, col_lam = st.columns(2)
                col_u.metric("熱貫流率 U（表面抵抗含む）", f"{U_app:.3f} W/m²K")
                col_lam.metric("等価熱伝導率 λ_eq（全層）", f"{lam_eq:.4f} W/mK")
                st.caption("室温 (20°C) での稳常値。λ_eq = 総厚 ÷ ΣR（表面抵抗除く）。U の表面抵抗: Rsi=0.13, Rse=0.04 m²K/W（ISO 6946）")
            except Exception as _e_th:
                st.caption(f"断熱性能の計算を省略しました: {_e_th}")

        # ── グラフ（静止画）───────────────────────────────────────────
        st.markdown("### 📈 グラフ")
        if state.out_dir and (state.out_dir / "summary.png").exists():
            st.image(str(state.out_dir / "summary.png"), width="stretch")

            col_dl1, col_dl2, col_dl3 = st.columns(3)
            for col, fname, label in [
                (col_dl1, "charring.png", "炭化深さ"),
                (col_dl2, "profiles.png", "温度分布"),
                (col_dl3, "surface.png", "表面温度"),
            ]:
                p = state.out_dir / fname
                if p.exists():
                    col.download_button(
                        f"⬇️ {label}グラフ",
                        data=p.read_bytes(),
                        file_name=fname,
                        mime="image/png",
                    )

        # ── ダウンロード ─────────────────────────────────────────────
        st.markdown("### 💾 ファイルダウンロード")
        dl1, dl2, dl3 = st.columns(3)

        if state.h5_path and state.h5_path.exists():
            dl1.download_button(
                "⬇️ HDF5 ダウンロード",
                data=state.h5_path.read_bytes(),
                file_name="result.h5",
                mime="application/x-hdf",
                help="他のメンバーと共有できる結果ファイル",
            )

        csv_path = state.out_dir / "results.csv" if state.out_dir else None
        if csv_path and csv_path.exists():
            dl2.download_button(
                "⬇️ CSV ダウンロード",
                data=csv_path.read_bytes(),
                file_name="results.csv",
                mime="text/csv",
            )

        txt_path = state.out_dir / "summary.txt" if state.out_dir else None
        if txt_path and txt_path.exists():
            txt_content = txt_path.read_text(encoding="utf-8")
            dl3.download_button(
                "⬇️ サマリー TXT",
                data=txt_content.encode("utf-8"),
                file_name="summary.txt",
                mime="text/plain",
            )

# ===========================================================================
# タブ4：インタラクティブ ビューア（Phase 7.3）
# ===========================================================================
with tab_3d:
    from clt_fire_sim.web import viz_plotly

    _state3d = web_runner.get_state()

    # 表示対象：履歴からロードされた結果 > 最新の解析結果
    _viewer_result = st.session_state.get("viewer_result")
    _viewer_label = st.session_state.get("viewer_label")

    if _viewer_result is None and (_state3d.status == "done" and _state3d.result is not None):
        _viewer_result = _state3d.result
        _viewer_label = None  # 最新解析

    if _viewer_result is None:
        st.info(
            "📊 解析が完了するとここにインタラクティブグラフが表示されます。\n\n"
            "「▶️ 解析実行」タブで解析を開始するか、\n"
            "「📂 履歴」タブで過去の結果を選択してください。"
        )
    else:
        # 表示中の結果ラベル
        if _viewer_label:
            st.info(f"📂 表示中: **{_viewer_label}**（過去の解析結果）")
            if st.button("🔄 最新の解析結果に戻る", key="viewer_reset"):
                st.session_state["viewer_result"] = None
                st.session_state["viewer_label"] = None
                st.rerun()
        else:
            st.success("📊 最新の解析結果を表示中")

        _result3d = _viewer_result
        _config3d = _result3d.get("config", config)
        _T_init3d = getattr(
            getattr(_config3d, "boundary", None),
            "unheated",
            None,
        )
        _T_init3d = getattr(_T_init3d, "T_inf", 20.0) if _T_init3d else 20.0

        st.markdown("### 🔬 インタラクティブ可視化")

        # ── サブタブ ──────────────────────────────────────────────────
        vtab_anim, vtab_heatmap, vtab_char, vtab_surf = st.tabs([
            "🌡️ 温度プロファイル",
            "🗺️ 温度場ヒートマップ",
            "🔥 炭化深さ",
            "📈 表面温度",
        ])

        with vtab_anim:
            st.caption(
                "スライダーで時刻を選択できます。"
                "▶ 再生ボタンでアニメーション再生。"
            )
            try:
                _n_frames = st.slider(
                    "フレーム数（細かいほど滑らか・重い）",
                    min_value=10, max_value=80, value=40, step=5,
                    key="viz_n_frames",
                )
                fig_anim = viz_plotly.make_temp_profile_animation(
                    _result3d, n_frames=_n_frames
                )
                st.plotly_chart(fig_anim, width="stretch")
            except Exception as _e:
                st.error(f"プロファイルアニメーションの生成に失敗しました: {_e}")

        with vtab_heatmap:
            st.caption(
                "横軸: 時間、縦軸: 加熱面からの深さ（上が加熱面）。"
                "白い破線は炭化前線（300°C 等温面）。"
            )
            try:
                fig_hm = viz_plotly.make_temp_heatmap(_result3d)
                st.plotly_chart(fig_hm, width="stretch")
            except Exception as _e:
                st.error(f"ヒートマップの生成に失敗しました: {_e}")

        with vtab_char:
            try:
                import numpy as _np
                import pandas as _pd
                fig_char = viz_plotly.make_charring_chart(_result3d)
                st.plotly_chart(fig_char, width="stretch")
                _t_char = _result3d["times"] / 60.0
                _char_mm = _result3d["char_depths"] * 1000.0
                _df_char = _pd.DataFrame({"時刻[分]": _t_char, "炭化深さ[mm]": _char_mm})
                st.download_button(
                    "📥 炭化深さ CSV",
                    data=_df_char.to_csv(index=False).encode("utf-8-sig"),
                    file_name="charring.csv",
                    mime="text/csv",
                )
            except Exception as _e:
                st.error(f"炭化深さグラフの生成に失敗しました: {_e}")

        with vtab_surf:
            try:
                import numpy as _np
                import pandas as _pd
                fig_surf = viz_plotly.make_surface_temp_chart(
                    _result3d, T_init=_T_init3d
                )
                st.plotly_chart(fig_surf, width="stretch")
                _t_surf = _result3d["times"] / 60.0
                _T_mat = _result3d["temperatures"]
                _mesh3d = _result3d.get("mesh")
                _is_3d = _mesh3d is not None and hasattr(_mesh3d, "ny")
                if _is_3d:
                    _nx3, _ny3, _nz3 = _mesh3d.nx, _mesh3d.ny, _mesh3d.nz
                    _T_h = _np.array([_T_mat[_ti].reshape(_nx3, _ny3, _nz3)[0, :, :].mean() for _ti in range(len(_T_mat))])
                    _T_u = _np.array([_T_mat[_ti].reshape(_nx3, _ny3, _nz3)[-1, :, :].mean() for _ti in range(len(_T_mat))])
                else:
                    _T_h = _T_mat[:, 0]
                    _T_u = _T_mat[:, -1]
                # 構造層表面温度
                _T_struct_csv = None
                _struct_col = "構造層表面温度[°C]"
                try:
                    _x_c3 = _result3d.get("x_centers")
                    _cfg3 = _result3d.get("config", config)
                    if _x_c3 is not None and _cfg3 is not None:
                        _spec3 = _cfg3.specimen
                        _li3 = _cfg3.evaluation.structural_layer_index
                        _nl3 = len(_spec3.layers)
                        if _li3 < 0:
                            _li3 = _nl3 + _li3
                        _li3 = int(_np.clip(_li3, 0, _nl3 - 1))
                        _ls3 = _np.cumsum([0.0] + [_la.thickness_mm / 1000.0 for _la in _spec3.layers])
                        _sci3 = int(_np.argmin(_np.abs(_x_c3 - _ls3[_li3])))
                        if _is_3d:
                            _T_struct_csv = _np.array([_T_mat[_ti].reshape(_nx3, _ny3, _nz3)[_sci3, :, :].mean() for _ti in range(len(_T_mat))])
                        else:
                            _T_struct_csv = _T_mat[:, _sci3]
                        _struct_col = f"構造層表面温度（第{_li3+1}層）[°C]"
                except Exception:
                    pass
                _surf_dict: dict = {"時刻[分]": _t_surf, "加熱面温度[°C]": _T_h}
                if _T_struct_csv is not None:
                    _surf_dict[_struct_col] = _T_struct_csv
                _surf_dict["非加熱面温度[°C]"] = _T_u
                _df_surf = _pd.DataFrame(_surf_dict)
                st.download_button(
                    "📥 表面温度 CSV",
                    data=_df_surf.to_csv(index=False).encode("utf-8-sig"),
                    file_name="surface_temp.csv",
                    mime="text/csv",
                )
            except Exception as _e:
                st.error(f"表面温度グラフの生成に失敗しました: {_e}")

# ===========================================================================
# タブ5：孔配置最適化（Phase 8）
# ===========================================================================
with tab_opt:
    from clt_fire_sim.web.optimizer_ui import render_optimizer_tab
    render_optimizer_tab(config)

# ===========================================================================
# タブ6：パレート最適化
# ===========================================================================
with tab_pareto:
    from clt_fire_sim.web.pareto_ui import render_pareto_tab
    render_pareto_tab()

# ===========================================================================
# タブ7：履歴（Phase 7.4）
# ===========================================================================
with tab_history:
    from clt_fire_sim.web.result_store import load_result_hdf5, scan_results

    outputs_dir = Path("outputs")
    history = scan_results(outputs_dir)

    if not history:
        st.info(
            "📂 まだ解析履歴がありません。\n"
            "「▶️ 解析実行」タブで解析を実行すると、ここに一覧が表示されます。"
        )
    else:
        st.markdown(f"### 📂 解析履歴（{len(history)} 件）")

        # ── 比較テーブル ──────────────────────────────────────────────
        with st.expander("📊 全件比較テーブル", expanded=True):
            _cmp_rows = []
            for _m in history:
                _ev = _m.get("evaluation", {})
                _row = {
                    "試験体名": _m.get("specimen_name", "?"),
                    "解析日時": _m.get("timestamp", "?")[:19].replace("T", " "),
                    "総厚 [mm]": f"{_m.get('total_thickness_mm', 0):.0f}",
                    "加熱時間 [分]": f"{_m.get('t_end_min', 0):.0f}",
                    "β [mm/min]": (
                        f"{_m.get('char_rate_beta', 0):.3f}"
                        if _m.get("char_rate_beta") is not None
                        else "—"
                    ),
                }
                # 評価時刻ごとの遮熱性を列追加
                for _t_key, _jdg in sorted(_ev.items()):
                    _row[_t_key] = "✅" if _jdg.get("insulation_ok") else "❌"
                _cmp_rows.append(_row)

            if _cmp_rows:
                st.dataframe(
                    pd.DataFrame(_cmp_rows),
                    hide_index=True,
                )

        st.divider()

        # ── 個別カード ────────────────────────────────────────────────
        for _meta in history:
            _run_id = _meta.get("run_id", "")
            _h5_path = Path(_meta.get("h5_path", ""))
            _run_dir = Path(_meta.get("run_dir", ""))
            _specimen = _meta.get("specimen_name", "?")
            _ts = _meta.get("timestamp", "?")[:19].replace("T", " ")
            _thick = _meta.get("total_thickness_mm", 0)
            _t_end = _meta.get("t_end_min", 0)
            _beta = _meta.get("char_rate_beta")

            with st.expander(
                f"**{_specimen}** — {_ts} | "
                f"総厚 {_thick:.0f}mm | {_t_end:.0f}分解析",
            ):
                # 評価テーブル
                _ev = _meta.get("evaluation", {})
                if _ev:
                    _rows = [
                        {
                            "評価時刻": k,
                            "遮熱性": "✅ 合格" if v.get("insulation_ok") else "❌ 不合格",
                            "炭化深さ [mm]": f"{v.get('char_depth_mm', 0):.1f}",
                            "非加熱面 [°C]": f"{v.get('unheated_face_temp_C', 0):.1f}",
                            "温度上昇 [K]": f"{v.get('unheated_face_rise_K', 0):.1f}",
                        }
                        for k, v in sorted(_ev.items())
                    ]
                    st.dataframe(
                        pd.DataFrame(_rows), hide_index=True
                    )

                if _beta is not None:
                    st.metric("炭化速度 β", f"{_beta:.3f} mm/min")

                # ── ボタン行 ──
                _bc1, _bc2, _bc3, _bc4 = st.columns(4)

                # 📊 ビューアで再表示
                if _h5_path.exists():
                    if _bc1.button(
                        "🔬 ビューアで見る",
                        key=f"view_{_run_id}",
                        help="このデータをビューアタブで表示します",
                    ):
                        try:
                            _loaded, _loaded_meta = load_result_hdf5(_h5_path)
                            st.session_state["viewer_result"] = _loaded
                            st.session_state["viewer_label"] = (
                                f"{_specimen}（{_ts}）"
                            )
                            st.success(
                                f"✅ 読み込み完了！「🔬 ビューア」タブで確認してください。"
                            )
                        except Exception as _exc:
                            st.error(f"読み込みエラー: {_exc}")

                # ⬇️ HDF5 ダウンロード
                if _h5_path.exists():
                    _bc2.download_button(
                        "⬇️ HDF5",
                        data=_h5_path.read_bytes(),
                        file_name=f"{_run_id}.h5",
                        mime="application/x-hdf",
                        key=f"dl_{_run_id}",
                    )

                # ⬇️ CSV ダウンロード
                _csv_p = _run_dir / "results.csv"
                if _csv_p.exists():
                    _bc3.download_button(
                        "⬇️ CSV",
                        data=_csv_p.read_bytes(),
                        file_name=f"{_run_id}.csv",
                        mime="text/csv",
                        key=f"csv_{_run_id}",
                    )

                # 🗑️ 削除ボタン
                _confirm_id = st.session_state.get("confirm_delete_run_id")
                if _confirm_id == _run_id:
                    # 確認ダイアログ
                    st.warning(
                        f"⚠️ **{_specimen}** のデータを完全に削除します。元に戻せません。"
                    )
                    _dc1, _dc2 = st.columns(2)
                    if _dc1.button(
                        "✅ 削除する", key=f"del_ok_{_run_id}", type="primary"
                    ):
                        import shutil
                        try:
                            shutil.rmtree(_run_dir)
                            st.session_state["confirm_delete_run_id"] = None
                            st.success("削除しました。")
                            st.rerun()
                        except Exception as _exc:
                            st.error(f"削除エラー: {_exc}")
                    if _dc2.button("❌ キャンセル", key=f"del_cancel_{_run_id}"):
                        st.session_state["confirm_delete_run_id"] = None
                        st.rerun()
                else:
                    if _bc4.button(
                        "🗑️ 削除",
                        key=f"del_{_run_id}",
                        help="この解析結果を削除します",
                    ):
                        st.session_state["confirm_delete_run_id"] = _run_id
                        st.rerun()

# ---------------------------------------------------------------------------
# フッター
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "CLT 耐火シミュレータ v0.4 | Eurocode 5 (EN 1995-1-2) 準拠 | "
    "Phase 7.4 — 履歴・比較・ビューア連携"
)

