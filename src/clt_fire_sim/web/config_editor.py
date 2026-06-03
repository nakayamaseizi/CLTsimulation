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
) -> dict[str, Any]:
    """新しいレイヤー辞書を生成する（UUID 付き）。"""
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "material": material,
        "thickness_mm": thickness_mm,
        "rho_0_kg_m3": rho_0_kg_m3,
        "moisture_content": moisture_content,
    }


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

    # 解析条件
    if "t_end_min" not in st.session_state:
        st.session_state.t_end_min = 90.0
    if "mesh_option" not in st.session_state:
        st.session_state.mesh_option = "標準（推奨）"
    if "T_init" not in st.session_state:
        st.session_state.T_init = 20.0
    if "unheated_bc_option" not in st.session_state:
        st.session_state.unheated_bc_option = UNHEATED_BC_OPTIONS[0]
    if "specimen_name" not in st.session_state:
        st.session_state.specimen_name = "CLT試験体"

    # チュートリアル（初回のみ表示）
    if "tutorial_done" not in st.session_state:
        st.session_state.tutorial_done = False


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

    # レイヤーリストを新 UUID で再構築
    st.session_state.layers = [
        _make_layer(
            name=layer.get("name", f"第{i+1}層"),
            material=layer.get("material", "sugi"),
            thickness_mm=layer.get("thickness_mm", 30.0),
            rho_0_kg_m3=layer.get("rho_0_kg_m3", 400.0),
            moisture_content=layer.get("moisture_content", 0.12),
        )
        for i, layer in enumerate(spec.get("layers", []))
    ]

    st.session_state.specimen_name = spec.get("name", "CLT試験体")
    st.session_state.t_end_min = sim.get("t_end_min", 90.0)


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

            # レイヤー詳細入力
            st.text_input(
                "層名",
                value=layer["name"],
                key=f"name_{lid}",
                help="管理用の名前（例：'第1層（加熱面）'）",
            )
            mat_idx = MATERIAL_KEYS.index(layer["material"]) if layer["material"] in MATERIAL_KEYS else 0
            st.selectbox(
                "材料",
                options=MATERIAL_KEYS,
                format_func=lambda k: MATERIAL_LABELS[k],
                index=mat_idx,
                key=f"mat_{lid}",
                help="木材の樹種を選択してください。",
            )
            st.number_input(
                "厚さ [mm]",
                min_value=5.0, max_value=200.0,
                value=layer["thickness_mm"],
                step=5.0,
                key=f"thick_{lid}",
                help="1 層あたりの厚さ（一般的な CLT 規格は 12〜45mm）",
            )

            # 詳細設定（折りたたみ）
            with st.expander("▸ 詳細設定（密度・含水率）"):
                st.number_input(
                    "初期乾燥密度 [kg/m³]",
                    min_value=200.0, max_value=900.0,
                    value=layer["rho_0_kg_m3"],
                    step=10.0,
                    key=f"rho_{lid}",
                    help="木材の乾燥密度。カタログ値があれば変更してください。",
                )
                st.slider(
                    "含水率 [%]",
                    min_value=0, max_value=30,
                    value=int(layer["moisture_content"] * 100),
                    key=f"mc_{lid}",
                    help="含水率が高いほど蒸発帯で炭化が遅くなります（建築部材標準：12%）",
                )

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
    layers: list[LayerConfig] = []
    for layer in st.session_state.layers:
        lid = layer["id"]
        material = st.session_state.get(f"mat_{lid}", layer["material"])
        thickness = st.session_state.get(f"thick_{lid}", layer["thickness_mm"])
        rho_0 = st.session_state.get(f"rho_{lid}", layer["rho_0_kg_m3"])
        mc_pct = st.session_state.get(f"mc_{lid}", int(layer["moisture_content"] * 100))
        layers.append(LayerConfig(
            material=material,
            thickness_mm=float(thickness),
            rho_0_kg_m3=float(rho_0),
            moisture_content=float(mc_pct) / 100.0,
        ))

    n_cells = MESH_OPTIONS.get(
        st.session_state.get("mesh_option", "標準（推奨）"), 12
    )
    t_end = float(st.session_state.get("t_end_min", 90.0))
    T_init = float(st.session_state.get("T_init", 20.0))
    spec_name = st.session_state.get("specimen_name", "CLT試験体")

    # 非加熱面 BC（断熱の場合は alpha_c=0 で近似）
    bc_option = st.session_state.get("unheated_bc_option", UNHEATED_BC_OPTIONS[0])
    if "断熱" in bc_option:
        unheated_bc = UnheatedBCConfig(alpha_c=0.0, eps_m=0.0, T_inf=T_init)
    else:
        unheated_bc = UnheatedBCConfig(alpha_c=9.0, eps_m=0.8, T_inf=T_init)

    # 評価時刻（60 分以下の加熱時間には 60 分評価をスキップ）
    eval_times = [t for t in [60.0, 75.0, 90.0, 120.0] if t <= t_end]
    if not eval_times:
        eval_times = [t_end]

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
        ),
        evaluation=EvaluationConfig(
            char_temp=300.0,
            eval_times_min=eval_times,
            unheated_face_temp_limit=160.0,
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


def config_to_yaml_str(config: CLTConfig) -> str:
    """CLTConfig を YAML 文字列に変換する。

    既存 CLI（`python -m clt_fire_sim.runner config.yaml`）で
    そのまま使える形式で出力する。
    """
    data: dict = {
        "specimen": {
            "name": config.specimen.name,
            "layers": [
                {
                    "material": layer.material,
                    "thickness_mm": layer.thickness_mm,
                    "rho_0_kg_m3": layer.rho_0_kg_m3,
                    "moisture_content": layer.moisture_content,
                }
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
        },
        "evaluation": {
            "char_temp": config.evaluation.char_temp,
            "eval_times_min": config.evaluation.eval_times_min,
            "unheated_face_temp_limit": config.evaluation.unheated_face_temp_limit,
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
    _render_analysis_settings()
    st.divider()
    _render_yaml_io_section()

    return build_config()
