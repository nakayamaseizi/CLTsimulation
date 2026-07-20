"""
config_editor.py
================
【役割】
Streamlit サイドバーの設定フォームをレンダリングし、
現在の入力から CLTConfig オブジェクトを生成する。

【公開関数】
- init_session_state()       : セッション状態を初期化する（app.py の最初に呼ぶ）
- render_sidebar()           : サイドバー全体を描画し、現在の CLTConfig を返す
- build_config()             : 現在のセッション状態から CLTConfig を構築する
- config_to_yaml_str()       : CLTConfig を YAML 文字列に変換する
- load_config_from_yaml_str(): YAML 文字列から CLTConfig を復元する

【設計方針】
- 各レイヤーに UUID を割り当て、ウィジェットキーを安定化させる
  （並び替え・削除後も値がずれない）
- セッション状態 st.session_state.layers がレイヤーリストの正本
- プリセット読込時は新しい UUID を発行し、ウィジェットを完全リフレッシュ
"""

from __future__ import annotations

import uuid
from io import StringIO
from typing import Any

import streamlit as st
import yaml

from clt_fire_sim.config import (
    BoundaryConfig,
    CLTConfig,
    EvaluationConfig,
    HeatedBCConfig,
    LayerConfig,
    SimulationConfig,
    SpecimenConfig,
    UnheatedBCConfig,
)
from clt_fire_sim.materials import MATERIAL_DB

from .presets import PRESET_NAMES, PRESETS

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

# 材料プルダウンの選択肢（キー: DB キー、値: 日本語名）
MATERIAL_KEYS: list[str] = list(MATERIAL_DB.keys())
MATERIAL_LABELS: dict[str, str] = {
    k: f"{v['name']} （密度 {v['rho_0']:.0f} kg/m³）"
    for k, v in MATERIAL_DB.items()
}


def _all_material_keys() -> list[str]:
    """DB + カスタム材料のキー一覧を返す（カスタムは "custom:名前" 形式）。"""
    keys = list(MATERIAL_DB.keys())
    for c in st.session_state.get("custom_materials", []):
        keys.append(f"custom:{c['name']}")
    return keys


def _all_material_labels() -> dict[str, str]:
    """DB + カスタム材料のラベル辞書を返す。"""
    # 特定材料にアイコンプレフィックスを付けて識別しやすくする
    _PREFIX = {
        "charred_cork":    "🔶",   # 遅燃断熱材
        "cork":            "🟠",
        "glass_wool":      "⬜",   # 無機断熱材
        "fr_sugi":         "🔴",   # 不燃処理スギ（燃え止まり層用）
        "insulation_board":"🟤",   # 木質繊維断熱板
        "momigara":        "🌾",   # 籾殻（ばら充填）
        "kuntan":          "⬛",   # 籾殻くん炭（炭化籾殻）
    }
    labels = {}
    for k, v in MATERIAL_DB.items():
        prefix = _PREFIX.get(k, "")
        labels[k] = f"{prefix} {v['name']} （密度 {v['rho_0']:.0f} kg/m³）".strip()
    for c in st.session_state.get("custom_materials", []):
        key = f"custom:{c['name']}"
        labels[key] = f"⚙️ {c['name']}（k={c['k']:.3f} W/m·K, ρ={c['rho']:.0f} kg/m³）"
    return labels

# メッシュ粗さのプリセット（n_cells_per_layer）
MESH_OPTIONS: dict[str, int] = {
    "粗い（計算速い・精度低め）": 6,
    "標準（推奨）": 12,
    "細かい（精度高・計算遅め）": 20,
}

# 非加熱面 BC の選択肢
UNHEATED_BC_OPTIONS: list[str] = [
    "対流＋輻射冷却（Eurocode 5 標準）",
    "断熱（保守側評価）",
]

# ---------------------------------------------------------------------------
# セッション状態の初期化
# ---------------------------------------------------------------------------

def _make_layer(
    name: str,
    material: str = "sugi",
    thickness_mm: float = 30.0,
    rho_0_kg_m3: float = 400.0,
    moisture_content: float = 0.12,
    material_type: str = "wood",
    void_fraction: float = 0.0,
    k_W_mK: float | None = None,
    cp_J_kgK: float | None = None,
    custom_name: str = "",
    hole_diameter_mm: float = 10.0,
    hole_pitch_mm: float = 30.0,
    hole_depth_mm: float = 20.0,
    slit_width_mm: float = 15.0,
    slit_depth_mm: float = 3.0,
    slit_pitch_mm: float = 30.0,
    k_char_factor: float = 1.0,
    hole_filler: str = "air",
    hole_filler_density_kg_m3: float | None = None,
) -> dict[str, Any]:
    """新しいレイヤー辞書を生成する（UUID 付き）。"""
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "material": material,
        "thickness_mm": thickness_mm,
        "rho_0_kg_m3": rho_0_kg_m3,
        "moisture_content": moisture_content,
        "material_type": material_type,
        "void_fraction": void_fraction,
        "k_W_mK": k_W_mK,
        "cp_J_kgK": cp_J_kgK,
        "custom_name": custom_name,
        "hole_diameter_mm": hole_diameter_mm,
        "hole_pitch_mm": hole_pitch_mm,
        "hole_depth_mm": hole_depth_mm,
        "slit_width_mm": slit_width_mm,
        "slit_depth_mm": slit_depth_mm,
        "slit_pitch_mm": slit_pitch_mm,
        "k_char_factor": k_char_factor,
        "hole_filler": hole_filler,
        "hole_filler_density_kg_m3": hole_filler_density_kg_m3,
    }


# 孔・スリット充填材の選択肢（キー: MATERIAL_DB キー or "air"）
HOLE_FILLER_OPTIONS: dict[str, str] = {
    "air": "🌬️ 空気（充填なし）",
    "kuntan": "⬛ 籾殻くん炭（燻炭）",
    "momigara": "🌾 籾殻",
}


def void_fraction_3d(vf_area: float, depth_mm: float, thickness_mm: float) -> float:
    """加工深さを考慮した体積空洞率（3D）を返す。

    断面空洞率 φ_A（加工面を正面から見た面積比）に対し、
    加工が板厚を貫通しない場合は深さ比 (depth/thickness) の分だけ
    実際の空洞体積が小さくなる。

        φ_V = φ_A × min(depth / thickness, 1.0)

    【導出（有孔加工）】
        孔1本の体積     = π(d/2)² × H
        単位セル体積     = P² × t
        φ_V = π(d/2)²·H / (P²·t) = φ_A × (H/t)

    【導出（スリット加工）】
        スリット体積/長さ = W × D
        単位セル体積/長さ = P × t
        φ_V = W·D / (P·t) = φ_A × (D/t)

    Parameters
    ----------
    vf_area : float
        断面空洞率 φ_A（0〜1）。
    depth_mm : float
        加工深さ（孔深さ H またはスリット深さ D）[mm]。
    thickness_mm : float
        層の板厚 t [mm]。

    Returns
    -------
    float
        体積空洞率 φ_V（0〜1）。貫通加工（depth ≥ thickness）では φ_A に一致する。
    """
    if thickness_mm <= 0:
        return 0.0
    return float(vf_area) * min(float(depth_mm) / float(thickness_mm), 1.0)


def init_session_state() -> None:
    """セッション状態を初期化する。

    Streamlit は毎回スクリプトを再実行するため、
    session_state に一度だけ初期値を設定する。
    """
    if "layers" not in st.session_state:
        # デフォルト：5層スギCLT 150mm
        st.session_state.layers = [
            _make_layer(f"第{i+1}層", "sugi", 30.0, 400.0, 0.12)
            for i in range(5)
        ]

    # 解析モード（"1D" / "3D"）
    if "analysis_mode" not in st.session_state:
        st.session_state.analysis_mode = "1D"

    # 試験体寸法（3D モード時に使用）
    if "specimen_width_mm" not in st.session_state:
        st.session_state.specimen_width_mm = 300.0
    if "specimen_height_mm" not in st.session_state:
        st.session_state.specimen_height_mm = 300.0

    # メッシュ分割数（実行タブで直接指定。サイドバーの mesh_option より優先）
    if "n_cells_x_run" not in st.session_state:
        st.session_state["n_cells_x_run"] = 12
    if "n_cells_y" not in st.session_state:
        st.session_state["n_cells_y"] = 6
    if "n_cells_z" not in st.session_state:
        st.session_state["n_cells_z"] = 6

    # 解析条件
    if "t_end_min" not in st.session_state:
        st.session_state.t_end_min = 90.0
    if "mesh_option" not in st.session_state:
        st.session_state.mesh_option = "標準（推奨）"
    if "T_init" not in st.session_state:
        st.session_state.T_init = 20.0
    if "unheated_bc_option" not in st.session_state:
        st.session_state.unheated_bc_option = UNHEATED_BC_OPTIONS[0]
    if "alpha_c_unheated" not in st.session_state:
        st.session_state.alpha_c_unheated = 9.0
    if "eps_m_unheated" not in st.session_state:
        st.session_state.eps_m_unheated = 0.8
    if "specimen_name" not in st.session_state:
        st.session_state.specimen_name = "CLT試験体"

    # チュートリアル（初回のみ表示）
    if "tutorial_done" not in st.session_state:
        st.session_state.tutorial_done = False

    # カスタム材料レジストリ（ユーザーが追加した材料）
    if "custom_materials" not in st.session_state:
        st.session_state.custom_materials = []  # list of dict

    # 放冷フェーズ設定
    if "cooling_time_h" not in st.session_state:
        st.session_state.cooling_time_h = 0.0
    if "cooling_tau_min" not in st.session_state:
        st.session_state.cooling_tau_min = 75.0
    if "char_stop_enabled" not in st.session_state:
        st.session_state.char_stop_enabled = False
    if "structural_layer_index" not in st.session_state:
        st.session_state.structural_layer_index = -1


# ---------------------------------------------------------------------------
# レイヤー操作コールバック
# ---------------------------------------------------------------------------

def _add_layer() -> None:
    """新しいレイヤーをリストの末尾に追加する。"""
    n = len(st.session_state.layers) + 1
    st.session_state.layers.append(
        _make_layer(f"第{n}層", "sugi", 30.0, 400.0, 0.12)
    )


def _delete_layer(lid: str) -> None:
    """指定 ID のレイヤーをリストから削除する。"""
    st.session_state.layers = [
        layer for layer in st.session_state.layers if layer["id"] != lid
    ]


def _move_layer_up(lid: str) -> None:
    """指定 ID のレイヤーを 1 つ上（加熱面側）に移動する。"""
    layers = st.session_state.layers
    for i, layer in enumerate(layers):
        if layer["id"] == lid and i > 0:
            layers[i - 1], layers[i] = layers[i], layers[i - 1]
            break


def _move_layer_down(lid: str) -> None:
    """指定 ID のレイヤーを 1 つ下（非加熱面側）に移動する。"""
    layers = st.session_state.layers
    for i, layer in enumerate(layers):
        if layer["id"] == lid and i < len(layers) - 1:
            layers[i + 1], layers[i] = layers[i], layers[i + 1]
            break


def _load_preset(preset_name: str) -> None:
    """プリセット設定をセッション状態に読み込む。

    新しい UUID を発行することで、ウィジェットを完全リフレッシュする。
    """
    preset = PRESETS[preset_name]
    spec = preset.get("specimen", {})
    sim = preset.get("simulation", {})

    # レイヤーリストを新 UUID で再構築（新フィールドも引き継ぐ）
    st.session_state.layers = [
        _make_layer(
            name=layer.get("name", f"第{i+1}層"),
            material=layer.get("material", "sugi"),
            thickness_mm=layer.get("thickness_mm", 30.0),
            rho_0_kg_m3=layer.get("rho_0_kg_m3", 400.0),
            moisture_content=layer.get("moisture_content", 0.12),
            material_type=layer.get("material_type", "wood"),
            void_fraction=layer.get("void_fraction", 0.0),
            k_W_mK=layer.get("k_W_mK"),
            cp_J_kgK=layer.get("cp_J_kgK"),
            custom_name=layer.get("custom_name", ""),
            k_char_factor=layer.get("k_char_factor", 1.0),
            hole_filler=layer.get("hole_filler", "air"),
            hole_filler_density_kg_m3=layer.get("hole_filler_density_kg_m3"),
        )
        for i, layer in enumerate(spec.get("layers", []))
    ]

    st.session_state.specimen_name = spec.get("name", "CLT試験体")
    st.session_state.t_end_min = sim.get("t_end_min", 90.0)

    # 放冷フェーズ設定（プリセットに記載がある場合のみ反映）
    if "cooling_time_h" in sim:
        st.session_state.cooling_time_h = sim["cooling_time_h"]
    if "cooling_tau_min" in sim:
        st.session_state.cooling_tau_min = sim["cooling_tau_min"]

    # 燃え止まり評価設定
    ev = preset.get("evaluation", {})
    if "char_stop_enabled" in ev:
        st.session_state.char_stop_enabled = ev["char_stop_enabled"]
    if "structural_layer_index" in ev:
        st.session_state.structural_layer_index = ev["structural_layer_index"]


# ---------------------------------------------------------------------------
# サイドバーの各セクション描画
# ---------------------------------------------------------------------------

def _render_preset_section() -> None:
    """プリセット読込セクションを描画する。"""
    st.subheader("📂 プリセット読込")

    selected = st.selectbox(
        "プリセットを選択",
        options=["─ 選択してください ─"] + PRESET_NAMES,
        key="preset_selectbox",
        help="あらかじめ用意した CLT 設定を一発で読み込めます。",
    )

    if st.button(
        "このプリセットを読み込む",
        disabled=(selected == "─ 選択してください ─"),
        use_container_width=True,
    ):
        _load_preset(selected)
        st.success(f"「{selected}」を読み込みました。")
        st.rerun()

    # 選択中のプリセット説明を表示
    if selected != "─ 選択してください ─":
        desc = PRESETS[selected].get("description", "")
        if desc:
            st.caption(f"💡 {desc}")


def _render_mode_section() -> None:
    """解析モード選択セクションを描画する。"""
    st.subheader("⚙️ 解析モード")
    mode = st.radio(
        "計算の次元",
        options=["1D（推奨）", "3D"],
        index=0 if st.session_state.analysis_mode == "1D" else 1,
        horizontal=True,
        help="1D は厚み方向のみ計算。3D は幅・高さも考慮します（計算時間が大幅に増加）。",
        key="mode_radio",
    )
    st.session_state.analysis_mode = "1D" if mode == "1D（推奨）" else "3D"

    if st.session_state.analysis_mode == "3D":
        st.warning("⚠️ 3D モードは計算時間が大幅に増加します。まず 1D で試してください。")
        col_w, col_h = st.columns(2)
        st.session_state.specimen_width_mm = col_w.number_input(
            "幅 [mm]",
            min_value=100.0, max_value=2000.0,
            value=st.session_state.specimen_width_mm,
            step=50.0,
            help="試験体の幅（y 方向）",
            key="spec_width",
        )
        st.session_state.specimen_height_mm = col_h.number_input(
            "高さ [mm]",
            min_value=100.0, max_value=2000.0,
            value=st.session_state.specimen_height_mm,
            step=50.0,
            help="試験体の高さ（z 方向）",
            key="spec_height",
        )


def _render_layer_editor() -> None:
    """CLT 構成エディタセクションを描画する。

    各レイヤーを縦に並べ、▲▼削除ボタンで並び替え・削除できる。
    先頭（第1層）が加熱面側。
    """
    st.subheader("🏗️ CLT 構成エディタ")

    # 試験体名
    st.session_state.specimen_name = st.text_input(
        "試験体名",
        value=st.session_state.specimen_name,
        key="spec_name_input",
        help="レポートに表示される試験体の名前",
    )

    # 加熱面 → 非加熱面 の方向案内
    n_layers = len(st.session_state.layers)
    total_mm = sum(
        st.session_state.get(f"thick_{layer['id']}", layer["thickness_mm"])
        for layer in st.session_state.layers
    )
    col_label, col_total = st.columns([2, 1])
    col_label.markdown("🔥 **加熱面** ↓")
    col_total.metric("総厚", f"{total_mm:.0f} mm")

    layers = st.session_state.layers
    for i, layer in enumerate(layers):
        lid = layer["id"]
        is_first = i == 0
        is_last = i == n_layers - 1

        # レイヤーヘッダー（番号＋アイコン＋操作ボタン）
        face_icon = "🔥" if is_first else ("❄️" if is_last else "")
        with st.container():
            hcol1, hcol2, hcol3, hcol4 = st.columns([4, 1, 1, 1])
            hcol1.markdown(
                f"**第 {i+1} 層** {face_icon}",
                help="第1層が加熱面側（ISO 834 火炎に接する面）です。",
            )
            hcol2.button(
                "▲", key=f"up_{lid}",
                disabled=is_first,
                on_click=_move_layer_up, args=(lid,),
                help="1つ上（加熱面側）に移動",
            )
            hcol3.button(
                "▼", key=f"down_{lid}",
                disabled=is_last,
                on_click=_move_layer_down, args=(lid,),
                help="1つ下（非加熱面側）に移動",
            )
            hcol4.button(
                "✕", key=f"del_{lid}",
                disabled=(n_layers <= 1),
                on_click=_delete_layer, args=(lid,),
                help="このレイヤーを削除（最低1層必要）",
            )

            # 層名 + 厚さ
            c_name, c_thick = st.columns([3, 2])
            c_name.text_input(
                "層名",
                value=layer["name"],
                key=f"name_{lid}",
            )
            c_thick.number_input(
                "厚さ [mm]",
                min_value=5.0, max_value=200.0,
                value=layer["thickness_mm"],
                step=5.0,
                key=f"thick_{lid}",
            )

            # 材料タイプ選択
            MAT_TYPE_OPTIONS = [
                "🌲 木材",
                "◉ 有孔板（簡易）",
                "🔬 有孔板（精密・池畑式）",
                "〰 スリット加工",
                "⚙️ カスタム",
            ]
            _type_map = {
                "wood": 0,
                "perforated_wood": 1,
                "perforated_wood_advanced": 2,
                "slitted_wood": 3,
                "custom": 4,
            }
            _type_rev = {0: "wood", 1: "perforated_wood", 2: "perforated_wood_advanced",
                         3: "slitted_wood", 4: "custom"}
            cur_type = layer.get("material_type", "wood")
            type_idx = _type_map.get(cur_type, 0)
            sel_type_label = st.radio(
                "材料タイプ",
                options=MAT_TYPE_OPTIONS,
                index=type_idx,
                horizontal=False,
                key=f"mat_type_radio_{lid}",
                help=(
                    "🌲木材: Eurocode 5 温度依存物性値\n"
                    "◉有孔板(簡易): 並列混合則による等価λ\n"
                    "🔬有孔板(精密): 池畑(2021)実験式（Φ・H依存）\n"
                    "〰スリット: 柴田(2021)・池畑(2021)準拠のスリット加工モデル"
                ),
            )
            mat_type = _type_rev[MAT_TYPE_OPTIONS.index(sel_type_label)]

            def _render_base_material(lid=lid, layer=layer):
                """ベース樹種・密度・含水率の共通UI。"""
                _all_mat_keys = _all_material_keys()
                _all_mat_labels = _all_material_labels()
                cur_mat = layer.get("material", "sugi")
                if cur_mat not in _all_mat_keys:
                    cur_mat = "sugi"
                st.selectbox(
                    "ベース樹種",
                    options=_all_mat_keys,
                    format_func=lambda k: _all_mat_labels.get(k, k),
                    index=_all_mat_keys.index(cur_mat),
                    key=f"mat_{lid}",
                )
                with st.expander("▸ 詳細（密度・含水率）"):
                    st.number_input(
                        "乾燥密度 [kg/m³]",
                        min_value=100.0, max_value=1200.0,
                        value=layer["rho_0_kg_m3"],
                        step=10.0,
                        key=f"rho_{lid}",
                    )
                    st.slider(
                        "含水率 [%]",
                        min_value=0, max_value=30,
                        value=int(layer["moisture_content"] * 100),
                        key=f"mc_{lid}",
                    )
                    st.slider(
                        "炭化熱伝導係数 k_char_factor [-]",
                        min_value=1.0, max_value=5.0,
                        value=float(layer.get("k_char_factor", 1.0)),
                        step=0.1,
                        key=f"kcf_{lid}",
                        help="炭化域の熱伝導率倍率。木材:1.0〜1.5、有孔:1.5〜2.0、スリット:2.0〜3.0",
                    )

            def _render_filler_select(lid=lid, layer=layer):
                """孔・スリット充填材の選択UI（有孔・スリット加工共通）。"""
                _fill_keys = list(HOLE_FILLER_OPTIONS.keys())
                cur_fill = layer.get("hole_filler", "air")
                if cur_fill not in _fill_keys:
                    cur_fill = "air"
                st.selectbox(
                    "孔・スリット充填材",
                    options=_fill_keys,
                    format_func=lambda k: HOLE_FILLER_OPTIONS.get(k, k),
                    index=_fill_keys.index(cur_fill),
                    key=f"hfill_{lid}",
                    help=(
                        "孔・スリット内部の充填材。\n"
                        "空気: 従来モデル（池畑式・対流限界）\n"
                        "くん炭・籾殻: 充填材との並列混合則に切替。"
                        "充填時は孔内対流・燃え抜けが抑制される。"
                    ),
                )
                _f = st.session_state.get(f"hfill_{lid}", cur_fill)
                if _f != "air":
                    from clt_fire_sim.materials import MATERIAL_DB as _MDB2
                    _fdef = float(_MDB2.get(_f, {}).get("rho_0", 150.0))
                    st.number_input(
                        "充填密度（かさ密度）[kg/m³]",
                        min_value=90.0, max_value=300.0,
                        value=float(layer.get("hole_filler_density_kg_m3") or _fdef),
                        step=10.0,
                        key=f"hfdens_{lid}",
                        help=(
                            "詰め方の目安： 緩い 90〜110／標準 130〜150／圧縮 200〜220 kg/m³。\n"
                            "密度が高いほど空隙が減り、熱伝導率が上昇（断熱性能が低下）します。"
                        ),
                    )
                    try:
                        from clt_fire_sim.materials import loose_fill_k_rt as _lfk
                        _dv = float(st.session_state.get(f"hfdens_{lid}", _fdef))
                        st.caption(
                            f"→ 室温 λ ≈ **{_lfk(_dv):.4f} W/mK**（密度依存相関式）　"
                            "池畑式（空気孔の実験式）は適用外となり、"
                            "充填材との並列混合則で計算されます。"
                        )
                    except Exception:
                        pass

            # ── 木材 ──────────────────────────────────────────────
            if mat_type == "wood":
                _render_base_material()

            # ── 有孔板（簡易：並列混合則） ──────────────────────────
            elif mat_type == "perforated_wood":
                _render_base_material()
                import math as _math
                c_d, c_p = st.columns(2)
                c_d.number_input(
                    "孔径 d [mm]",
                    min_value=3.0, max_value=100.0,
                    value=float(layer.get("hole_diameter_mm", 10.0)),
                    step=1.0,
                    key=f"phi_{lid}",
                    help="円形貫通孔の直径",
                )
                c_p.number_input(
                    "ピッチ P [mm]",
                    min_value=5.0, max_value=200.0,
                    value=float(layer.get("hole_pitch_mm", 30.0)),
                    step=1.0,
                    key=f"pitch_{lid}",
                    help="孔の中心間距離（正方格子配置）",
                )
                _d_v = float(st.session_state.get(f"phi_{lid}", layer.get("hole_diameter_mm", 10.0)))
                _p_v = float(st.session_state.get(f"pitch_{lid}", layer.get("hole_pitch_mm", 30.0)))
                _vf_v = min(_math.pi * (_d_v / 2) ** 2 / max(_p_v, _d_v + 0.1) ** 2, 0.95)
                _rho_v = float(st.session_state.get(f"rho_{lid}", layer["rho_0_kg_m3"]))
                # 簡易モデルは貫通孔を仮定（板厚方向に一様）→ φ_V = φ_A
                st.caption(
                    f"**断面空洞率 φ_A** = π(d/2)²/P² ≈ **{_vf_v*100:.1f}%**　"
                    f"／　**体積空洞率 φ_V** ≈ **{_vf_v*100:.1f}%**"
                )
                st.caption(
                    f"↳ 貫通孔を仮定するモデルのため φ_V = φ_A。"
                    f"　等価密度 ≈ {_rho_v * (1 - _vf_v):.0f} kg/m³"
                )
                _render_filler_select()

            # ── 有孔板（精密：池畑 2021 実験式） ──────────────────
            elif mat_type == "perforated_wood_advanced":
                _render_base_material()
                import math as _math
                c_phi, c_p2, c_h = st.columns(3)
                c_phi.number_input(
                    "孔径 Φ [mm]",
                    min_value=3.0, max_value=18.0,
                    value=float(layer.get("hole_diameter_mm", 10.0)),
                    step=1.0,
                    key=f"phi_{lid}",
                    help="池畑(2021)実験式の適用範囲: 3〜18 mm",
                )
                c_p2.number_input(
                    "ピッチ P [mm]",
                    min_value=5.0, max_value=200.0,
                    value=float(layer.get("hole_pitch_mm", 30.0)),
                    step=1.0,
                    key=f"pitch_{lid}",
                    help="孔の中心間距離（正方格子配置）",
                )
                c_h.number_input(
                    "孔深さ H [mm]",
                    min_value=6.0, max_value=30.0,
                    value=float(layer.get("hole_depth_mm", 20.0)),
                    step=1.0,
                    key=f"hole_h_{lid}",
                    help="池畑(2021)実験式の適用範囲: 6〜30 mm。板厚と同じ場合は貫通孔。",
                )
                # 開孔率 & Ra1 表示
                try:
                    from clt_fire_sim.materials import _ra1_ikehata
                    phi_v = float(st.session_state.get(f"phi_{lid}", layer.get("hole_diameter_mm", 10.0)))
                    p2_v = float(st.session_state.get(f"pitch_{lid}", layer.get("hole_pitch_mm", 30.0)))
                    h_v = float(st.session_state.get(f"hole_h_{lid}", layer.get("hole_depth_mm", 20.0)))
                    t_v = float(st.session_state.get(f"thick_{lid}", layer["thickness_mm"]))
                    rho_v = float(st.session_state.get(f"rho_{lid}", layer["rho_0_kg_m3"]))
                    vf_adv = min(_math.pi * (phi_v / 2) ** 2 / max(p2_v, phi_v + 0.1) ** 2, 0.95)
                    vf3d = void_fraction_3d(vf_adv, h_v, t_v)
                    ra1 = _ra1_ikehata(phi_v, h_v)
                    st.caption(
                        f"**断面空洞率 φ_A** ≈ **{vf_adv*100:.1f}%**　"
                        f"／　**体積空洞率 φ_V** ≈ **{vf3d*100:.1f}%**"
                    )
                    if h_v >= t_v:
                        _depth_note = f"貫通孔（孔深 {h_v:.0f} ≥ 板厚 {t_v:.0f}mm）→ φ_V = φ_A"
                    else:
                        _depth_note = (
                            f"φ_V = φ_A × H/t = {vf_adv*100:.1f}% × {h_v:.0f}/{t_v:.0f}"
                        )
                    from clt_fire_sim.materials import (
                        _IKEHATA_REF_AREA_MM2, void_k_at_temperature,
                    )
                    _n_ref = (_IKEHATA_REF_AREA_MM2 * vf_adv
                              / max(_math.pi * (phi_v / 2) ** 2, 1e-9))
                    _ra = ra1 * _n_ref
                    _kv20 = (h_v * 1e-3 / _ra) if _ra > 1e-12 else 0.026
                    _kv500 = float(void_k_at_temperature(_kv20, 500.0))
                    st.caption(
                        f"↳ {_depth_note}　"
                        f"等価密度 ≈ {rho_v * (1 - vf3d):.0f} kg/m³"
                    )
                    st.caption(
                        f"📐 池畑式: Ra1 = {ra1*1e4:.2f}×10⁻⁴ m²K/W（孔1個）"
                        f"　×　N = {_n_ref:.0f} 個（300×300mm 換算）"
                        f"　→　**Ra = {_ra:.4f} m²K/W**"
                    )
                    st.caption(
                        f"🔥 孔の等価λ: 室温 **{_kv20:.4f}** → 500°C **{_kv500:.4f}** W/mK"
                        f"（輻射の T³ 則で外挿）"
                    )
                except Exception:
                    pass
                _render_filler_select()

            # ── スリット加工 ────────────────────────────────────
            elif mat_type == "slitted_wood":
                _render_base_material()
                c_w, c_d, c_p = st.columns(3)
                c_w.number_input(
                    "スリット幅 W [mm]",
                    min_value=5.0, max_value=30.0,
                    value=float(layer.get("slit_width_mm", 15.0)),
                    step=1.0,
                    key=f"sw_{lid}",
                    help="柴田(2021)推奨: 15mm（体積減少14%で耐火性能への影響小）",
                )
                c_d.number_input(
                    "スリット深さ D [mm]",
                    min_value=1.0, max_value=15.0,
                    value=float(layer.get("slit_depth_mm", 3.0)),
                    step=1.0,
                    key=f"sd_{lid}",
                    help="対流遷移深さ: 9〜12mm。D≤9mm では断熱効果あり。",
                )
                c_p.number_input(
                    "ピッチ P [mm]",
                    min_value=10.0, max_value=60.0,
                    value=float(layer.get("slit_pitch_mm", 30.0)),
                    step=5.0,
                    key=f"sp_{lid}",
                    help="スリット中心間距離",
                )
                try:
                    sw_v = float(st.session_state.get(f"sw_{lid}", layer.get("slit_width_mm", 15.0)))
                    sd_v = float(st.session_state.get(f"sd_{lid}", layer.get("slit_depth_mm", 3.0)))
                    sp_v = float(st.session_state.get(f"sp_{lid}", layer.get("slit_pitch_mm", 30.0)))
                    st_v = float(st.session_state.get(f"thick_{lid}", layer["thickness_mm"]))
                    srho_v = float(st.session_state.get(f"rho_{lid}", layer["rho_0_kg_m3"]))
                    from clt_fire_sim.materials import (
                        iso6946_air_gap_resistance as _iso_r,
                        _SLIT_CONV_ONSET_MM, _SLIT_VALIDATED_MAX_MM,
                    )
                    vf_slit = min(sw_v / max(sp_v, sw_v + 1.0), 0.95)
                    vf3d_slit = void_fraction_3d(vf_slit, sd_v, st_v)
                    rs_approx = _iso_r(sd_v, plateau_mm=_SLIT_CONV_ONSET_MM)
                    st.caption(
                        f"**断面空洞率 φ_A** = W/P ≈ **{vf_slit*100:.1f}%**　"
                        f"／　**体積空洞率 φ_V** ≈ **{vf3d_slit*100:.1f}%**"
                    )
                    st.caption(
                        f"↳ φ_V = φ_A × D/t = {vf_slit*100:.1f}% × {sd_v:.0f}/{st_v:.0f}　"
                        f"等価密度 ≈ {srho_v * (1 - vf3d_slit):.0f} kg/m³"
                    )
                    from clt_fire_sim.materials import void_k_at_temperature as _vkt
                    _skv20 = (sd_v * 1e-3 / rs_approx) if rs_approx > 1e-12 else 0.026
                    _skv500 = float(_vkt(_skv20, 500.0))
                    _clamp = (
                        f"（対流により {_SLIT_CONV_ONSET_MM:.0f}mm で頭打ち）"
                        if sd_v > _SLIT_CONV_ONSET_MM else ""
                    )
                    st.caption(
                        f"📐 スリット熱抵抗 Rs ≈ {rs_approx:.4f} m²K/W"
                        f"（ISO 6946 密閉空気層）{_clamp}"
                    )
                    st.caption(
                        f"🔥 スリットの等価λ: 室温 **{_skv20:.4f}** → 500°C **{_skv500:.4f}** W/mK"
                        f"（輻射の T³ 則で外挿）"
                    )
                    if sd_v > _SLIT_VALIDATED_MAX_MM:
                        st.warning(
                            f"⚠️ スリット深さ {sd_v:.1f}mm は実測の裏付け範囲"
                            f"（D ≤ {_SLIT_VALIDATED_MAX_MM:.0f}mm）を超えています。\n\n"
                            "池畑(2021) の D=21mm 実測では対流により熱抵抗が "
                            "D=3mm を下回りますが、本モデルは頭打ちまでしか再現しません。"
                            "**断熱性能を過大評価（危険側）**する恐れがあります。"
                        )
                except Exception:
                    pass
                _render_filler_select()
                # void_fraction のダミーキー（build_config で参照）
                if f"vf_{lid}" not in st.session_state:
                    st.session_state[f"vf_{lid}"] = 0

            # ── カスタム材料 ─────────────────────────────────────
            else:  # custom
                # 登録済みカスタム材料から選択 or 直接入力
                customs = st.session_state.get("custom_materials", [])
                if customs:
                    names = ["─ 直接入力 ─"] + [c["name"] for c in customs]
                    sel_custom = st.selectbox(
                        "登録済み材料から選択",
                        options=names,
                        key=f"custom_sel_{lid}",
                    )
                    if sel_custom != "─ 直接入力 ─":
                        found = next((c for c in customs if c["name"] == sel_custom), None)
                        if found:
                            layer["k_W_mK"] = found["k"]
                            layer["cp_J_kgK"] = found["cp"]
                            layer["rho_0_kg_m3"] = found["rho"]
                            layer["custom_name"] = found["name"]

                st.text_input(
                    "材料名",
                    value=layer.get("custom_name", ""),
                    key=f"custom_name_{lid}",
                    placeholder="例：断熱ボード",
                )
                col_k, col_cp = st.columns(2)
                col_k.number_input(
                    "熱伝導率 k [W/m·K]",
                    min_value=0.001, max_value=50.0,
                    value=float(layer.get("k_W_mK") or 0.12),
                    format="%.4f",
                    step=0.01,
                    key=f"k_custom_{lid}",
                    help="木材: 0.12 / 石膏ボード: 0.19 / コンクリート: 1.4",
                )
                col_cp.number_input(
                    "比熱 cp [J/kg·K]",
                    min_value=100.0, max_value=5000.0,
                    value=float(layer.get("cp_J_kgK") or 1600.0),
                    step=50.0,
                    key=f"cp_custom_{lid}",
                    help="木材: 1600 / 石膏ボード: 1050 / コンクリート: 880",
                )
                st.number_input(
                    "密度 ρ [kg/m³]",
                    min_value=10.0, max_value=3000.0,
                    value=layer["rho_0_kg_m3"],
                    step=10.0,
                    key=f"rho_{lid}",
                )
                # セッション状態にダミーキーを設定（build_config で参照）
                if f"mat_{lid}" not in st.session_state:
                    st.session_state[f"mat_{lid}"] = layer.get("material", "sugi")
                if f"mc_{lid}" not in st.session_state:
                    st.session_state[f"mc_{lid}"] = int(layer["moisture_content"] * 100)

            st.divider()

    # 層追加ボタン
    st.button(
        "＋ 層を追加",
        on_click=_add_layer,
        use_container_width=True,
        help="非加熱面側に新しい層を追加します。",
    )
    st.markdown("❄️ **非加熱面**")


def _render_analysis_settings() -> None:
    """解析条件セクションを描画する。"""
    st.subheader("🕐 解析条件")

    st.session_state.t_end_min = st.slider(
        "加熱時間 [分]",
        min_value=30, max_value=180,
        value=int(st.session_state.t_end_min),
        step=15,
        key="t_end_slider",
        help="ISO 834 加熱曲線を適用する時間。60・75・90・120 分が主な評価基準。",
    )

    # 評価時刻の自動表示
    eval_times = [t for t in [60, 75, 90, 120] if t <= st.session_state.t_end_min]
    if eval_times:
        st.caption(f"評価時刻：{', '.join(str(t) for t in eval_times)} 分")

    st.session_state.T_init = st.number_input(
        "初期温度 [°C]",
        min_value=0.0, max_value=40.0,
        value=st.session_state.T_init,
        step=1.0,
        key="T_init_input",
        help="シミュレーション開始時の試験体温度（室温：20°C が標準）",
    )

    st.session_state.unheated_bc_option = st.selectbox(
        "非加熱面 境界条件",
        options=UNHEATED_BC_OPTIONS,
        index=UNHEATED_BC_OPTIONS.index(st.session_state.unheated_bc_option),
        key="unheated_bc_select",
        help="非加熱面（裏面）の熱的境界条件。Eurocode 5 標準は対流＋輻射冷却。",
    )
    if "断熱" not in st.session_state.unheated_bc_option:
        col_ac, col_em = st.columns(2)
        st.session_state.alpha_c_unheated = col_ac.slider(
            "α_c [W/m²K]",
            min_value=1.0, max_value=30.0,
            value=float(st.session_state.alpha_c_unheated),
            step=1.0,
            key="alpha_c_slider",
            help="非加熱面の対流熱伝達率。Eurocode 5 規定値: 9 W/m²K。実験環境に強制対流があると 15〜25 程度。",
        )
        st.session_state.eps_m_unheated = col_em.slider(
            "ε_m [-]",
            min_value=0.1, max_value=1.0,
            value=float(st.session_state.eps_m_unheated),
            step=0.05,
            key="eps_m_slider",
            help="非加熱面の放射率。Eurocode 5 規定値: 0.8。値を大きくすると輻射放熱が増加する。",
        )

    st.session_state.mesh_option = st.selectbox(
        "メッシュの細かさ",
        options=list(MESH_OPTIONS.keys()),
        index=list(MESH_OPTIONS.keys()).index(st.session_state.mesh_option),
        key="mesh_select",
        help="細かいほど精度が上がりますが計算時間も増えます。まずは「標準」で。",
    )

    n_cells = MESH_OPTIONS[st.session_state.mesh_option]
    n_layers = len(st.session_state.layers)
    st.caption(f"総セル数：{n_cells * n_layers} セル（{n_layers} 層 × {n_cells} セル/層）")

    # ---- 放冷フェーズ設定 ----
    with st.expander("🧊 放冷フェーズ（燃え止まり型CLT向け）", expanded=False):
        st.caption(
            "炭化コルクやスリット加工ラミナを含む**燃え止まり型CLT**の評価に使用します。"
            "加熱終了後の炉内自然冷却をシミュレートし、構造層への損傷有無を判定します。"
        )
        cooling_h = st.number_input(
            "放冷時間 [時間]",
            min_value=0.0, max_value=6.0,
            value=st.session_state.cooling_time_h,
            step=0.5,
            key="cooling_time_input",
            help=(
                "0 = 放冷なし（従来動作）。\n"
                "防耐火性能基準・評価業務方法書: 加熱時間の3倍 + α = 通常4時間推奨。"
            ),
        )
        st.session_state.cooling_time_h = cooling_h

        if cooling_h > 0.0:
            tau_min = st.number_input(
                "炉内冷却時定数 τ [分]",
                min_value=1.0, max_value=200.0,
                value=st.session_state.cooling_tau_min,
                step=1.0,
                key="cooling_tau_input",
                help=(
                    "指数減衰モデルの時定数。小型マッフル炉（FUW210PB）実測推定値: 75分。\n"
                    "大型炉では時定数が大きくなります（より緩やかに冷却）。"
                ),
            )
            st.session_state.cooling_tau_min = tau_min

            # 燃え止まり判定オプション
            st.divider()
            char_stop = st.checkbox(
                "🛑 燃え止まり判定を行う",
                value=st.session_state.char_stop_enabled,
                key="char_stop_checkbox",
                help=(
                    "放冷終了時に構造層表面温度と試験体内最高温度を評価し、"
                    "自消（燃え止まり）の可否を判定します。"
                ),
            )
            st.session_state.char_stop_enabled = char_stop

            if char_stop:
                n_layers_now = len(st.session_state.layers)
                struct_idx = st.selectbox(
                    "構造層インデックス",
                    options=list(range(n_layers_now)),
                    index=min(max(st.session_state.structural_layer_index, 0), n_layers_now - 1),
                    format_func=lambda i: f"第{i+1}層（{'🔥加熱面側' if i == 0 else ('❄️非加熱面側' if i == n_layers_now-1 else '中間層')}）",
                    key="struct_layer_sel",
                    help="燃え止まり判定で「構造層表面温度」を評価する層。通常は炭化コルクの背後の層。",
                )
                st.session_state.structural_layer_index = struct_idx
                st.caption(
                    "判定基準：放冷終了時に構造層表面温度 < 100°C かつ "
                    "試験体内最高温度 < 250°C → 燃え止まり OK"
                )


def _render_yaml_io_section() -> None:
    """設定 YAML のエクスポート・インポートセクションを描画する。"""
    st.subheader("📄 設定ファイル")

    # YAML ダウンロード
    config = build_config()
    yaml_str = config_to_yaml_str(config)
    st.download_button(
        label="💾 YAML として保存",
        data=yaml_str.encode("utf-8"),
        file_name=f"{config.specimen.name or 'clt_config'}.yaml",
        mime="text/yaml",
        use_container_width=True,
        help="現在の設定を YAML ファイルとして保存します。既存 CLI からも実行できます。",
    )

    # YAML アップロード
    uploaded = st.file_uploader(
        "📂 YAML を読み込む",
        type=["yaml", "yml"],
        key="yaml_uploader",
        help="以前に保存した設定ファイル、または手書きの YAML を読み込みます。",
    )
    if uploaded is not None:
        _load_yaml_from_upload(uploaded)


def _load_yaml_from_upload(uploaded_file: Any) -> None:
    """アップロードされた YAML ファイルをセッション状態に反映する。

    Pydantic で検証し、エラーがあれば日本語でメッセージを表示する。
    """
    try:
        raw = yaml.safe_load(uploaded_file.read().decode("utf-8"))
        config = CLTConfig(**raw)
    except Exception as exc:
        st.error(f"❌ YAML の読み込みに失敗しました：{exc}")
        return

    # レイヤーを新 UUID で再構築（LayerConfig に name フィールドは無いため番号で命名）
    st.session_state.layers = [
        _make_layer(
            name=f"第{i+1}層",
            material=layer.material,
            thickness_mm=layer.thickness_mm,
            rho_0_kg_m3=layer.rho_0_kg_m3,
            moisture_content=layer.moisture_content,
            material_type=getattr(layer, "material_type", "wood"),
            void_fraction=getattr(layer, "void_fraction", 0.0),
            k_W_mK=getattr(layer, "k_W_mK", None),
            cp_J_kgK=getattr(layer, "cp_J_kgK", None),
            custom_name=getattr(layer, "custom_name", ""),
            hole_diameter_mm=getattr(layer, "hole_diameter_mm", 10.0),
            hole_pitch_mm=getattr(layer, "hole_pitch_mm", 30.0),
            hole_depth_mm=getattr(layer, "hole_depth_mm", 20.0),
            slit_width_mm=getattr(layer, "slit_width_mm", 15.0),
            slit_depth_mm=getattr(layer, "slit_depth_mm", 3.0),
            slit_pitch_mm=getattr(layer, "slit_pitch_mm", 30.0),
            k_char_factor=getattr(layer, "k_char_factor", 1.0),
            hole_filler=getattr(layer, "hole_filler", "air"),
            hole_filler_density_kg_m3=getattr(layer, "hole_filler_density_kg_m3", None),
        )
        for i, layer in enumerate(config.specimen.layers)
    ]
    st.session_state.specimen_name = config.specimen.name
    st.session_state.t_end_min = config.simulation.t_end_min
    st.session_state.T_init = config.boundary.unheated.T_inf
    st.session_state.mesh_option = _n_cells_to_option(config.simulation.n_cells_per_layer)
    st.success("✅ YAML を読み込みました。")
    st.rerun()


def _n_cells_to_option(n: int) -> str:
    """n_cells_per_layer の値をメッシュ選択肢名に変換する。"""
    if n <= 8:
        return "粗い（計算速い・精度低め）"
    if n <= 15:
        return "標準（推奨）"
    return "細かい（精度高・計算遅め）"


# ---------------------------------------------------------------------------
# 設定オブジェクト構築
# ---------------------------------------------------------------------------

def build_config() -> CLTConfig:
    """現在のセッション状態から CLTConfig を構築して返す。

    ウィジェットのセッション状態キー（例: `f"mat_{lid}"`）から値を読み取り、
    Pydantic モデルに変換する。
    """
    import math as _math
    layers: list[LayerConfig] = []
    for layer in st.session_state.layers:
        lid = layer["id"]

        # 材料タイプ
        _type_label_to_key = {
            "🌲 木材": "wood",
            "◉ 有孔板（簡易）": "perforated_wood",
            "🔬 有孔板（精密・池畑式）": "perforated_wood_advanced",
            "〰 スリット加工": "slitted_wood",
            "⚙️ カスタム": "custom",
            # 後方互換
            "◉ 有孔板": "perforated_wood",
        }
        type_label = st.session_state.get(f"mat_type_radio_{lid}", "🌲 木材")
        mat_type = _type_label_to_key.get(type_label, layer.get("material_type", "wood"))

        material = st.session_state.get(f"mat_{lid}", layer["material"])
        # "custom:名前" キーの場合は元の DB キーに戻す（物性値は custom フィールドで指定）
        if material.startswith("custom:"):
            material = layer.get("material", "sugi")

        thickness = st.session_state.get(f"thick_{lid}", layer["thickness_mm"])
        rho_0 = st.session_state.get(f"rho_{lid}", layer["rho_0_kg_m3"])
        mc_pct = st.session_state.get(f"mc_{lid}", int(layer["moisture_content"] * 100))

        # 有孔板用：孔径 + ピッチ → 開孔率を自動計算
        hole_diameter_mm = float(st.session_state.get(f"phi_{lid}", layer.get("hole_diameter_mm", 10.0)))
        hole_pitch_mm = float(st.session_state.get(f"pitch_{lid}", layer.get("hole_pitch_mm", 30.0)))
        hole_depth_mm = float(st.session_state.get(f"hole_h_{lid}", layer.get("hole_depth_mm", 20.0)))

        if mat_type in ("perforated_wood", "perforated_wood_advanced"):
            _p = max(hole_pitch_mm, hole_diameter_mm + 0.1)
            void_fraction = min(_math.pi * (hole_diameter_mm / 2) ** 2 / _p ** 2, 0.95)
        else:
            void_fraction = 0.0

        # スリット加工用
        slit_width_mm = float(st.session_state.get(f"sw_{lid}", layer.get("slit_width_mm", 15.0)))
        slit_depth_mm = float(st.session_state.get(f"sd_{lid}", layer.get("slit_depth_mm", 3.0)))
        slit_pitch_mm = float(st.session_state.get(f"sp_{lid}", layer.get("slit_pitch_mm", 30.0)))

        # カスタム材料
        k_val = st.session_state.get(f"k_custom_{lid}", layer.get("k_W_mK"))
        cp_val = st.session_state.get(f"cp_custom_{lid}", layer.get("cp_J_kgK"))
        custom_name = st.session_state.get(f"custom_name_{lid}", layer.get("custom_name", ""))
        k_char_factor = float(st.session_state.get(f"kcf_{lid}", layer.get("k_char_factor", 1.0)))
        hole_filler = str(st.session_state.get(f"hfill_{lid}", layer.get("hole_filler", "air")))
        if hole_filler != "air":
            _hf_d = st.session_state.get(
                f"hfdens_{lid}", layer.get("hole_filler_density_kg_m3")
            )
            hole_filler_density = float(_hf_d) if _hf_d else None
        else:
            hole_filler_density = None

        layers.append(LayerConfig(
            material=material,
            thickness_mm=float(thickness),
            rho_0_kg_m3=float(rho_0),
            moisture_content=float(mc_pct) / 100.0,
            material_type=mat_type,
            void_fraction=void_fraction,
            k_W_mK=float(k_val) if k_val is not None else None,
            cp_J_kgK=float(cp_val) if cp_val is not None else None,
            custom_name=custom_name,
            hole_diameter_mm=hole_diameter_mm,
            hole_pitch_mm=hole_pitch_mm,
            hole_depth_mm=hole_depth_mm,
            slit_width_mm=slit_width_mm,
            slit_depth_mm=slit_depth_mm,
            slit_pitch_mm=slit_pitch_mm,
            k_char_factor=k_char_factor,
            hole_filler=hole_filler,
            hole_filler_density_kg_m3=hole_filler_density,
        ))

    n_cells = int(st.session_state.get(
        "n_cells_x_run",
        MESH_OPTIONS.get(st.session_state.get("mesh_option", "標準（推奨）"), 12),
    ))
    t_end = float(st.session_state.get("t_end_min", 90.0))
    T_init = float(st.session_state.get("T_init", 20.0))
    spec_name = st.session_state.get("specimen_name", "CLT試験体")

    # 非加熱面 BC（断熱の場合は alpha_c=0 で近似）
    bc_option = st.session_state.get("unheated_bc_option", UNHEATED_BC_OPTIONS[0])
    if "断熱" in bc_option:
        unheated_bc = UnheatedBCConfig(alpha_c=0.0, eps_m=0.0, T_inf=T_init)
    else:
        alpha_c_val = float(st.session_state.get("alpha_c_unheated", 9.0))
        eps_m_val = float(st.session_state.get("eps_m_unheated", 0.8))
        unheated_bc = UnheatedBCConfig(alpha_c=alpha_c_val, eps_m=eps_m_val, T_inf=T_init)

    # 評価時刻（60 分以下の加熱時間には 60 分評価をスキップ）
    eval_times = [t for t in [60.0, 75.0, 90.0, 120.0] if t <= t_end]
    if not eval_times:
        eval_times = [t_end]

    # 放冷フェーズ設定
    cooling_time_h = float(st.session_state.get("cooling_time_h", 0.0))
    cooling_tau_min = float(st.session_state.get("cooling_tau_min", 75.0))
    char_stop_enabled = bool(st.session_state.get("char_stop_enabled", False))
    structural_layer_index = int(st.session_state.get("structural_layer_index", -1))

    return CLTConfig(
        specimen=SpecimenConfig(name=spec_name, layers=layers),
        boundary=BoundaryConfig(
            heated=HeatedBCConfig(alpha_c=25.0, eps_m=0.8, eps_f=1.0),
            unheated=unheated_bc,
        ),
        simulation=SimulationConfig(
            t_end_min=t_end,
            dt_base_s=5.0,
            dt_min_s=1.0,
            dt_max_s=10.0,
            n_picard=3,
            n_cells_per_layer=n_cells,
            mesh_ratio=1.05,
            record_interval_s=30.0,
            cooling_time_h=cooling_time_h,
            cooling_tau_min=cooling_tau_min,
        ),
        evaluation=EvaluationConfig(
            char_temp=300.0,
            eval_times_min=eval_times,
            unheated_face_temp_limit=160.0,
            char_stop_enabled=char_stop_enabled,
            structural_layer_index=structural_layer_index,
        ),
    )


def load_config_from_yaml_str(yaml_str: str) -> CLTConfig:
    """YAML 文字列から CLTConfig を復元する。

    HDF5 に保存された設定 YAML を読み込む際に使用する。
    YAML 文字列は config_to_yaml_str() で生成されたものを想定する。

    Parameters
    ----------
    yaml_str : str
        YAML 文字列（または bytes）。

    Returns
    -------
    CLTConfig
        復元した設定オブジェクト。パースに失敗した場合は例外を送出する。
    """
    if isinstance(yaml_str, (bytes, bytearray)):
        yaml_str = yaml_str.decode("utf-8")
    data = yaml.safe_load(yaml_str)
    return CLTConfig.model_validate(data)


def _layer_to_dict(layer: "LayerConfig") -> dict:
    """LayerConfig を YAML 用辞書に変換する（新フィールド対応）。"""
    d: dict = {
        "material": layer.material,
        "thickness_mm": layer.thickness_mm,
        "rho_0_kg_m3": layer.rho_0_kg_m3,
        "moisture_content": layer.moisture_content,
    }
    if layer.material_type != "wood":
        d["material_type"] = layer.material_type
    if layer.void_fraction > 0:
        d["void_fraction"] = round(layer.void_fraction, 4)
    if layer.k_W_mK is not None:
        d["k_W_mK"] = layer.k_W_mK
    if layer.cp_J_kgK is not None:
        d["cp_J_kgK"] = layer.cp_J_kgK
    if layer.custom_name:
        d["custom_name"] = layer.custom_name
    # 有孔板用（簡易・精密共通）
    if layer.material_type in ("perforated_wood", "perforated_wood_advanced"):
        d["hole_diameter_mm"] = layer.hole_diameter_mm
        d["hole_pitch_mm"] = layer.hole_pitch_mm
    if layer.material_type == "perforated_wood_advanced":
        d["hole_depth_mm"] = layer.hole_depth_mm
    # スリット加工用
    if layer.material_type == "slitted_wood":
        d["slit_width_mm"] = layer.slit_width_mm
        d["slit_depth_mm"] = layer.slit_depth_mm
        d["slit_pitch_mm"] = layer.slit_pitch_mm
    if getattr(layer, "k_char_factor", 1.0) != 1.0:
        d["k_char_factor"] = round(layer.k_char_factor, 2)
    # 孔・スリット充填材（"air" 以外のみ保存）
    if getattr(layer, "hole_filler", "air") != "air":
        d["hole_filler"] = layer.hole_filler
        if getattr(layer, "hole_filler_density_kg_m3", None):
            d["hole_filler_density_kg_m3"] = round(layer.hole_filler_density_kg_m3, 1)
    return d


def config_to_yaml_str(config: "CLTConfig") -> str:
    """CLTConfig を YAML 文字列に変換する。

    既存 CLI（`python -m clt_fire_sim.runner config.yaml`）で
    そのまま使える形式で出力する。
    """
    data: dict = {
        "specimen": {
            "name": config.specimen.name,
            "layers": [
                _layer_to_dict(layer)
                for layer in config.specimen.layers
            ],
        },
        "boundary": {
            "heated": {
                "type": "iso834",
                "alpha_c": config.boundary.heated.alpha_c,
                "eps_m": config.boundary.heated.eps_m,
                "eps_f": config.boundary.heated.eps_f,
            },
            "unheated": {
                "type": "conv_rad_cooling",
                "alpha_c": config.boundary.unheated.alpha_c,
                "eps_m": config.boundary.unheated.eps_m,
                "T_inf": config.boundary.unheated.T_inf,
            },
        },
        "simulation": {
            "t_end_min": config.simulation.t_end_min,
            "dt_base_s": config.simulation.dt_base_s,
            "dt_min_s": config.simulation.dt_min_s,
            "dt_max_s": config.simulation.dt_max_s,
            "n_picard": config.simulation.n_picard,
            "n_cells_per_layer": config.simulation.n_cells_per_layer,
            "mesh_ratio": config.simulation.mesh_ratio,
            "record_interval_s": config.simulation.record_interval_s,
            **( {"cooling_time_h": config.simulation.cooling_time_h,
                 "cooling_tau_min": config.simulation.cooling_tau_min}
               if config.simulation.cooling_time_h > 0 else {} ),
        },
        "evaluation": {
            "char_temp": config.evaluation.char_temp,
            "eval_times_min": config.evaluation.eval_times_min,
            "unheated_face_temp_limit": config.evaluation.unheated_face_temp_limit,
            **( {"char_stop_enabled": config.evaluation.char_stop_enabled,
                 "structural_layer_index": config.evaluation.structural_layer_index}
               if config.evaluation.char_stop_enabled else {} ),
        },
    }
    buf = StringIO()
    yaml.dump(
        data, buf,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# サイドバー全体のレンダリング
# ---------------------------------------------------------------------------

def render_sidebar() -> CLTConfig:
    """サイドバー全体を描画し、現在の設定を CLTConfig として返す。

    app.py の `with st.sidebar:` ブロック内から呼び出す。

    Returns
    -------
    CLTConfig
        現在のフォーム入力から構築した設定オブジェクト。
    """
    _render_preset_section()
    st.divider()
    _render_mode_section()
    st.divider()
    _render_layer_editor()
    st.divider()
    _render_custom_material_manager()
    st.divider()
    _render_analysis_settings()
    st.divider()
    _render_yaml_io_section()

    return build_config()


# ---------------------------------------------------------------------------
# カスタム材料管理セクション
# ---------------------------------------------------------------------------

def _render_custom_material_manager() -> None:
    """カスタム材料の登録・削除セクションを描画する。"""
    st.subheader("🧪 カスタム材料の管理")

    customs = st.session_state.get("custom_materials", [])

    # 登録済み一覧
    if customs:
        for idx, c in enumerate(customs):
            col_info, col_del = st.columns([5, 1])
            col_info.markdown(
                f"**{c['name']}**  "
                f"k={c['k']:.3f} W/m·K、ρ={c['rho']:.0f} kg/m³、cp={c['cp']:.0f} J/kg·K"
            )
            if col_del.button("🗑️", key=f"del_custom_{idx}", help="削除"):
                st.session_state.custom_materials.pop(idx)
                st.rerun()
    else:
        st.caption("登録済みのカスタム材料はありません。")

    # 新規登録フォーム
    with st.expander("＋ 新しいカスタム材料を登録", expanded=False):
        new_name = st.text_input("材料名", key="new_custom_name", placeholder="例：石膏ボード")
        col_k2, col_cp2, col_rho2 = st.columns(3)
        new_k = col_k2.number_input(
            "k [W/m·K]", min_value=0.001, max_value=50.0,
            value=0.19, format="%.3f", step=0.01, key="new_custom_k",
            help="石膏: 0.19 / 木材: 0.12 / コンクリ: 1.4",
        )
        new_cp = col_cp2.number_input(
            "cp [J/kg·K]", min_value=100.0, max_value=5000.0,
            value=1050.0, step=50.0, key="new_custom_cp",
            help="石膏: 1050 / 木材: 1600 / コンクリ: 880",
        )
        new_rho = col_rho2.number_input(
            "ρ [kg/m³]", min_value=10.0, max_value=3000.0,
            value=800.0, step=10.0, key="new_custom_rho",
            help="石膏: 800 / 木材: 400 / コンクリ: 2300",
        )

        if st.button("登録", key="add_custom_material", use_container_width=True):
            if not new_name.strip():
                st.error("材料名を入力してください。")
            elif any(c["name"] == new_name.strip() for c in customs):
                st.error(f"「{new_name}」はすでに登録されています。")
            else:
                st.session_state.custom_materials.append({
                    "name": new_name.strip(),
                    "k": float(new_k),
                    "cp": float(new_cp),
                    "rho": float(new_rho),
                })
                st.success(f"「{new_name}」を登録しました。レイヤーの材料タイプで「木材」を選ぶと使えます。")
                st.rerun()
