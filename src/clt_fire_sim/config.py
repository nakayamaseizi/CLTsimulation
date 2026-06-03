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
        材料名。materials.MATERIAL_DB のキー。
    thickness_mm : float
        層の厚み [mm]。正の値のみ許容。
    rho_0_kg_m3 : float
        初期乾燥密度 [kg/m³]。
    moisture_content : float
        含水率（質量比、0〜1）。
    """

    material: str = "sugi"
    thickness_mm: float = Field(gt=0, description="層厚み [mm]（正の値）")
    rho_0_kg_m3: float = Field(default=400.0, gt=0, description="初期乾燥密度 [kg/m³]")
    moisture_content: float = Field(
        default=0.12, ge=0.0, le=1.0, description="含水率（質量比 0〜1）"
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

    t_end_min: float = Field(default=90.0, gt=0, description="解析終了時刻 [分]")
    dt_base_s: float = Field(default=5.0, gt=0, description="基本時間刻み [s]")
    dt_min_s: float = Field(default=1.0, gt=0, description="最小時間刻み [s]")
    dt_max_s: float = Field(default=10.0, gt=0, description="最大時間刻み [s]")
    n_picard: int = Field(default=3, ge=1, description="ピカード反復回数")
    n_cells_per_layer: int = Field(default=12, ge=3, description="層あたりのセル数")
    mesh_ratio: float = Field(default=1.05, ge=1.0, description="メッシュ幾何公比")
    record_interval_s: float = Field(default=30.0, gt=0, description="記録間隔 [s]")


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
