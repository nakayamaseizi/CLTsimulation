"""
config.py
=========
【役割】
YAML 設定ファイルの読み込みと検証。pydantic BaseModel を使って
各パラメータの型・範囲チェックを自動化する。

【設定ファイルの階層構造】
    specimen:       試験体の層構成・材料
    boundary:       加熱面・非加熱面の境界条件
    simulation:     ソルバーの数値パラメータ
    evaluation:     性能評価基準（耐火・遮熱性の判定閾値）

【使い方】
    from clt_fire_sim.config import load_config
    config = load_config("configs/clt_5layer_sugi.yaml")
    config.specimen.layers[0].thickness_mm  # → 30.0
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 試験体（Specimen）設定
# ---------------------------------------------------------------------------

class LayerConfig(BaseModel):
    """CLT 1 層分の設定。

    Parameters
    ----------
    material : str
        材料名。materials.MATERIAL_DB のキー（"sugi", "hinoki" 等）。
    thickness_mm : float
        層の厚み [mm]。正の値のみ許容。
    rho_0_kg_m3 : float
        初期乾燥密度 [kg/m³]。
    moisture_content : float
        含水率（質量比、0〜1）。
    material_type : str
        材料タイプ。
        - "wood"            : Eurocode 5 木材モデル（デフォルト）
        - "perforated_wood" : 等間隔孔・スリット板（等価均質物性値）
        - "custom"          : ユーザー定義（k・ρ・cp を直接入力）
    void_fraction : float
        空洞率（0〜0.95）。material_type="perforated_wood" のときのみ有効。
    k_W_mK : float or None
        熱伝導率 [W/m·K]。material_type="custom" のときに必須。
    cp_J_kgK : float or None
        比熱 [J/kg·K]。material_type="custom" のときに必須。
    custom_name : str
        カスタム材料名（表示・レポート用）。
    """

    material: str = "sugi"
    thickness_mm: float = Field(gt=0, description="層厚み [mm]（正の値）")
    rho_0_kg_m3: float = Field(default=400.0, gt=0, description="初期乾燥密度 [kg/m³]")
    moisture_content: float = Field(
        default=0.12, ge=0.0, le=1.0, description="含水率（質量比 0〜1）"
    )
    # 拡張フィールド（デフォルト値あり → 既存 YAML との後方互換性を維持）
    material_type: str = Field(
        default="wood",
        description="材料タイプ: wood|perforated_wood|perforated_wood_advanced|slitted_wood|custom",
    )
    void_fraction: float = Field(default=0.0, ge=0.0, lt=1.0, description="空洞率（有孔板用）")
    k_W_mK: float | None = Field(default=None, description="熱伝導率 [W/m·K]（カスタム用）")
    cp_J_kgK: float | None = Field(default=None, description="比熱 [J/kg·K]（カスタム用）")
    custom_name: str = Field(default="", description="カスタム材料名")
    # ---- 炭化速度補正係数 ----
    k_char_factor: float = Field(
        default=1.0,
        ge=1.0,
        le=3.0,
        description=(
            "炭化域（T>300°C）の熱伝導率補正係数。燃焼発熱による炭化促進を近似。\n"
            "1.0=補正なし（Eurocode 5 保守値）。\n"
            "実験再現用推奨値: 無加工スギ→1.3、有孔加工→1.5、スリット幅10深9→2.0。"
        ),
    )
    # ---- 有孔加工精密モデル（perforated_wood_advanced）用 ----
    hole_diameter_mm: float = Field(default=10.0, gt=0, description="孔径 [mm]")
    hole_pitch_mm: float = Field(default=30.0, gt=0, description="孔ピッチ（中心間距離）[mm]")
    hole_depth_mm: float = Field(default=20.0, gt=0, description="孔深さ [mm]（池畑式: 6〜30mm）")
    # ---- スリット加工モデル（slitted_wood）用 ----
    slit_width_mm: float = Field(default=15.0, gt=0, description="スリット幅 [mm]")
    slit_depth_mm: float = Field(default=3.0, gt=0, description="スリット深さ [mm]")
    slit_pitch_mm: float = Field(default=30.0, gt=0, description="スリットピッチ（中心間距離）[mm]")
    # ---- 接触熱抵抗（この層の非加熱面側界面） ----
    contact_resistance_m2KW: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "この層の非加熱面側界面の接触熱抵抗 R_contact [m²K/W]。\n"
            "CLT の接着層（レゾルシノール・フェノール共縮合樹脂など）を近似。\n"
            "典型値: 0.0001〜0.0005 m²K/W（接着剤厚 0.1-0.3mm, λ≈0.2-0.3 W/mK）。\n"
            "デフォルト 0.0 = 接触熱抵抗なし（従来動作）。"
        ),
    )


class SpecimenConfig(BaseModel):
    """試験体全体の設定。

    Parameters
    ----------
    name : str
        試験体の名称（グラフ・レポートのタイトルに使用）。
    layers : list[LayerConfig]
        各層の設定リスト。加熱面側（左）から非加熱面側（右）の順。
    """

    name: str = "CLT試験体"
    layers: list[LayerConfig]


# ---------------------------------------------------------------------------
# 境界条件（Boundary）設定
# ---------------------------------------------------------------------------

class HeatedBCConfig(BaseModel):
    """加熱面（ISO 834 標準加熱）の境界条件設定。

    Parameters
    ----------
    type : str
        BC の種類識別子。現在は "iso834" のみ対応。
    alpha_c : float
        対流熱伝達率 [W/(m²·K)]。Eurocode 5 規定値は 25 W/(m²·K)。
    eps_m : float
        材料表面放射率（0〜1）。Eurocode 5 規定値は 0.8。
    eps_f : float
        火炎放射率（0〜1）。Eurocode 5 規定値は 1.0。
    """

    type: str = "iso834"
    alpha_c: float = Field(default=25.0, ge=0.0, description="対流熱伝達率 [W/(m²·K)]")
    eps_m: float = Field(default=0.8, ge=0.0, le=1.0, description="材料表面放射率")
    eps_f: float = Field(default=1.0, ge=0.0, le=1.0, description="火炎放射率")


class UnheatedBCConfig(BaseModel):
    """非加熱面（自然対流＋輻射）の境界条件設定。

    Parameters
    ----------
    type : str
        BC の種類識別子。現在は "conv_rad_cooling" のみ対応。
    alpha_c : float
        対流熱伝達率 [W/(m²·K)]。Eurocode 5 規定値は 9 W/(m²·K)。
    eps_m : float
        材料表面放射率（0〜1）。Eurocode 5 規定値は 0.8。
    T_inf : float
        外気温度 [°C]。通常 20°C。
    """

    type: str = "conv_rad_cooling"
    alpha_c: float = Field(default=9.0, ge=0.0, description="対流熱伝達率 [W/(m²·K)]")
    eps_m: float = Field(default=0.8, ge=0.0, le=1.0, description="材料表面放射率")
    T_inf: float = Field(default=20.0, description="外気温度 [°C]")


class BoundaryConfig(BaseModel):
    """境界条件全体の設定（加熱面＋非加熱面）。"""

    heated: HeatedBCConfig = HeatedBCConfig()
    unheated: UnheatedBCConfig = UnheatedBCConfig()


# ---------------------------------------------------------------------------
# ソルバー（Simulation）設定
# ---------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    """数値ソルバーのパラメータ設定。

    Parameters
    ----------
    t_end_min : float
        解析終了時刻 [分]。
    dt_base_s : float
        通常時の時間刻み幅 [s]。
    dt_min_s : float
        最小時間刻み幅 [s]（水蒸発帯通過中などに使用）。
    dt_max_s : float
        最大時間刻み幅 [s]。
    n_picard : int
        ピカード反復回数（非線形物性値の収束反復）。
    n_cells_per_layer : int
        層あたりのセル数。細かいほど精度が上がるが計算時間も増加。
    mesh_ratio : float
        メッシュの幾何公比（隣接セル幅の比）。1.05〜1.15 を推奨。
    record_interval_s : float
        温度場の記録間隔 [s]。
    """

    t_end_min: float = Field(default=90.0, gt=0, description="加熱終了時刻 [分]")
    dt_base_s: float = Field(default=5.0, gt=0, description="基本時間刻み [s]")
    dt_min_s: float = Field(default=1.0, gt=0, description="最小時間刻み [s]")
    dt_max_s: float = Field(default=10.0, gt=0, description="最大時間刻み [s]")
    n_picard: int = Field(default=3, ge=1, description="ピカード反復回数")
    n_cells_per_layer: int = Field(default=12, ge=3, description="層あたりのセル数")
    mesh_ratio: float = Field(default=1.05, ge=1.0, description="メッシュ幾何公比")
    record_interval_s: float = Field(default=30.0, gt=0, description="記録間隔 [s]")
    # ---- 放冷フェーズ設定 ----
    cooling_time_h: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "加熱終了後の放冷シミュレーション時間 [時間]。"
            "0 = 放冷なし（従来動作）。"
            "燃え止まり型CLTの評価には 4.0（=240分）を推奨（防耐火評価業務方法書より）。"
        ),
    )
    cooling_tau_min: float = Field(
        default=45.0,
        gt=0,
        description=(
            "炉内冷却時定数 τ [分]。"
            "指数減衰モデル T_furnace = T_amb + (T_end - T_amb)·exp(-t'/τ)。"
            "小型電気マッフル炉（FUW210PB）実験データの再解析結果:\n"
            "  τ ≈ 45 分（朱 2023 図3-8 の炉内温度曲線からフィット）。\n"
            "  60 分後炉内温度: τ=45 → 264°C（実験値 200-300°C と整合）。\n"
            "  旧デフォルト 75 分では 436°C と過大推定されていた。\n"
            "大型炉や換気量の少ない条件では τ を大きく設定すること。"
        ),
    )


# ---------------------------------------------------------------------------
# 性能評価（Evaluation）設定
# ---------------------------------------------------------------------------

class EvaluationConfig(BaseModel):
    """耐火性能評価の基準値設定。

    Parameters
    ----------
    char_temp : float
        炭化温度 [°C]。Eurocode 5 では 300°C を使用。
    eval_times_min : list[float]
        性能評価時刻のリスト [分]。
        日本建築基準法: 60分（耐火）、60/75/90分（準耐火）。
    unheated_face_temp_limit : float
        非加熱面温度の上限 [°C]。
        遮熱性基準: 初期温度 + 140K = 160°C（初期 20°C の場合）。
    """

    char_temp: float = Field(default=300.0, description="炭化温度 [°C]")
    eval_times_min: list[float] = Field(
        default=[60.0, 75.0, 90.0],
        description="性能評価時刻リスト [分]",
    )
    unheated_face_temp_limit: float = Field(
        default=160.0,
        description="非加熱面温度上限 [°C]（遮熱性基準: 20 + 140 = 160°C）",
    )
    # ---- 燃え止まり評価設定 ----
    char_stop_enabled: bool = Field(
        default=False,
        description=(
            "燃え止まり（自消）評価を行うか否か。"
            "True にするには simulation.cooling_time_h > 0 が必要。"
            "燃え止まり型CLTや炭化コルクを使用する場合に True にする。"
        ),
    )
    char_stop_struct_temp_limit: float = Field(
        default=100.0,
        description=(
            "燃え止まり判定の構造層表面温度上限 [°C]。"
            "放冷終了時にこの温度以下であれば燃え止まりと判定。"
            "鷹野研究室の実験では 100°C を基準とする（柴田 2021, 中村 2022）。"
        ),
    )
    char_stop_max_temp_limit: float = Field(
        default=250.0,
        description=(
            "燃え止まり判定の試験体内最高温度上限 [°C]。"
            "放冷終了時に試験体内のどのセルもこの温度以下であれば燃え止まりと判定。"
        ),
    )
    structural_layer_index: int = Field(
        default=-1,
        description=(
            "構造層の層インデックス（0-indexed）。"
            "-1 = 最終層（非加熱面側）。"
            "燃え止まり判定で構造層表面温度を評価する際に使用する。"
        ),
    )


# ---------------------------------------------------------------------------
# トップレベル設定クラス
# ---------------------------------------------------------------------------

class CLTConfig(BaseModel):
    """CLT 耐火シミュレーション全体の設定。

    YAML ファイルから load_config() でロードする。
    各セクションはデフォルト値を持つため、YAML に書かれていない
    セクションは自動的にデフォルト値が適用される。

    Examples
    --------
    >>> config = load_config("configs/clt_5layer_sugi.yaml")
    >>> config.specimen.name
    '5層スギCLT 150mm'
    >>> config.boundary.heated.alpha_c
    25.0
    """

    specimen: SpecimenConfig
    boundary: BoundaryConfig = BoundaryConfig()
    simulation: SimulationConfig = SimulationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


# ---------------------------------------------------------------------------
# YAML ローダー
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> CLTConfig:
    """YAML ファイルから CLT 設定を読み込み、検証して返す。

    pydantic が型チェックと範囲チェックを自動で行う。
    不正な値（負の厚み、範囲外の放射率など）は ValidationError を送出する。

    Parameters
    ----------
    path : str or Path
        YAML 設定ファイルのパス。

    Returns
    -------
    CLTConfig
        検証済みの設定オブジェクト。

    Raises
    ------
    FileNotFoundError
        指定したパスに YAML ファイルが存在しない場合。
    pydantic.ValidationError
        設定値が制約条件（型・範囲）を満たさない場合。
    yaml.YAMLError
        YAML の構文エラーがある場合。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return CLTConfig(**data)
