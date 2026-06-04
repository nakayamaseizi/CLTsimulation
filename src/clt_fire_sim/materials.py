"""
materials.py
============
【役割】
温度依存物性値の管理と補間。Eurocode 5 (EN 1995-1-2) Annex B の
データテーブルを実装し、任意温度での物性値を線形補間で返す。

【実装する物性値】
- 密度比  ρ(T)/ρ₀       （含水率依存）
- 比熱    cp(T)  [J/(kg·K)]（含水率依存）
- 熱伝導率 k(T)  [W/(m·K)] （含水率によらず一定）

【99°C・120°C の段差の扱い】
水の蒸発に伴う密度・比熱の不連続変化を
±smooth_half_width (デフォルト 2.5°C) の幅で滑らかに繋ぐ。
例: (99, val_low), (99, val_high) → (96.5, val_low), (101.5, val_high)
"""

from __future__ import annotations

from typing import Union

import numpy as np

# テーブルの型エイリアス
# [(温度 [°C], 物性値), ...]
ThermalTable = list[tuple[float, float]]


# ---------------------------------------------------------------------------
# Eurocode 5 Annex B 物性値テーブル（定数）
# ---------------------------------------------------------------------------

# ---- 熱伝導率 k(T) [W/(m·K)] ----
# 注意：高温（800°C以上）で k が上昇するのは、炭素層の熱放射による
# 「見かけ上の」熱伝導率の増大を Eurocode がモデル化しているため。
WOOD_THERMAL_CONDUCTIVITY: ThermalTable = [
    (20,    0.12),
    (200,   0.15),
    (350,   0.07),   # 炭化収縮で一時的に低下
    (500,   0.09),
    (800,   0.35),   # 炭素層の輻射伝熱
    (1200,  1.50),   # 高温の炭素層
]

# 比熱テーブルのピーク値（含水率 12% 換算、Eurocode 5 規定値）
# 99〜120°C の高い cp は水の蒸発潜熱をエンタルピー法で表現
# 蒸発潜熱換算: 0.12 kg_水 × 2260 kJ/kg ÷ 21°C ≈ 12914 J/(kg·K)
# Eurocode 5 では 13600 J/(kg·K) を採用（保守側の切り上げ）
_CP_PEAK_99_REF = 13600.0   # 含水率 12% 時の 99°C ピーク値 [J/(kg·K)]
_CP_PEAK_120_REF = 13500.0  # 含水率 12% 時の 120°C 終端値 [J/(kg·K)]
_W_REFERENCE = 0.12         # ピーク値の基準含水率（Eurocode 5 の規定値）


def _make_density_ratio_table(moisture_content: float) -> ThermalTable:
    """含水率から密度比テーブルを生成する。

    Parameters
    ----------
    moisture_content : float
        含水率（質量比）。例: 0.12 = 12%。

    Returns
    -------
    ThermalTable
        [(温度[°C], ρ/ρ₀), ...] のリスト。
        99°C で段差（水の蒸発）→ 後で smooth_table_jumps() で滑らかにする。
    """
    w = moisture_content
    return [
        (20,    1.0 + w),   # 常温：自由水を含む
        (99,    1.0 + w),   # 99°C 直前：まだ水分を保持
        (99,    1.0),       # 99°C：瞬時に水が蒸発（段差）
        (120,   1.0),
        (200,   1.0),
        (250,   0.93),
        (300,   0.76),      # 炭化が進行し始める
        (350,   0.52),
        (400,   0.38),
        (600,   0.28),
        (800,   0.26),      # 完全炭化後も質量が残る（炭素＋灰分）
        (1200,  0.0),
    ]


def _make_specific_heat_table(moisture_content: float) -> ThermalTable:
    """含水率から比熱テーブルを生成する。

    Parameters
    ----------
    moisture_content : float
        含水率（質量比）。

    Returns
    -------
    ThermalTable
        [(温度[°C], cp [J/(kg·K)]), ...] のリスト。
        99°C と 120°C に段差あり → 後で smooth_table_jumps() で処理。

    Notes
    -----
    含水率 0% の場合は蒸発ピークなし（cp_peak = 0）。
    w ≠ 0.12 の場合はピーク値を線形スケーリングする。
    """
    if moisture_content > 1e-6:
        # 含水率に応じてピーク値をスケーリング（Eurocode 5 は 12% が基準）
        scale = moisture_content / _W_REFERENCE
        cp_peak_99 = _CP_PEAK_99_REF * scale
        cp_peak_120 = _CP_PEAK_120_REF * scale
    else:
        # 完全乾燥材：蒸発ピークなし
        cp_peak_99 = 0.0
        cp_peak_120 = 0.0

    # 99°C 直前の比熱（水なしの木材の値 + 水分の熱容量を一部加算）
    cp_just_below_99 = 1770.0

    return [
        (20,    1530.0),
        (99,    cp_just_below_99),              # 99°C 直前
        (99,    cp_just_below_99 + cp_peak_99), # 99°C 直後（蒸発ピーク開始、段差）
        (120,   cp_just_below_99 + cp_peak_120),# 120°C（蒸発ピーク終端）
        (120,   2120.0),                        # 120°C 直後（乾燥材、段差）
        (200,   2000.0),
        (250,   1620.0),
        (300,   710.0),    # 急激な熱分解で cp が一時低下
        (350,   850.0),
        (400,   1000.0),
        (600,   1400.0),
        (800,   1650.0),
        (1200,  1650.0),
    ]


# ---------------------------------------------------------------------------
# 補間ユーティリティ
# ---------------------------------------------------------------------------

def smooth_table_jumps(
    table: ThermalTable,
    smooth_half_width: float = 2.5,
) -> ThermalTable:
    """テーブル内の同一温度の段差を滑らかなランプに変換する。

    Eurocode の表には同じ温度に 2 つの値が存在する（段差）。
    これをそのまま numpy.interp に渡すと予期しない結果になるため、
    段差を ±smooth_half_width [°C] の範囲でスムージングする。

    例：
        (99, 1770), (99, 13600)
        → (96.5, 1770), (101.5, 13600)   （smooth_half_width=2.5 の場合）

    Parameters
    ----------
    table : ThermalTable
        入力テーブル（温度の重複エントリを含んでよい）。
    smooth_half_width : float
        段差を滑らかにする幅の半分 [°C]。デフォルト 2.5°C。

    Returns
    -------
    ThermalTable
        段差をスムージングしたテーブル。
    """
    result: ThermalTable = []
    i = 0
    while i < len(table):
        if (
            i + 1 < len(table)
            and abs(table[i][0] - table[i + 1][0]) < 1e-6
        ):
            # 同一温度の連続エントリ = 段差
            T_jump = table[i][0]
            v_before = table[i][1]   # 段差前の値
            v_after = table[i + 1][1]  # 段差後の値
            # 段差を ±smooth_half_width に広げる
            result.append((T_jump - smooth_half_width, v_before))
            result.append((T_jump + smooth_half_width, v_after))
            i += 2
        else:
            result.append(table[i])
            i += 1
    return result


def table_interp(table: ThermalTable, T: float | np.ndarray) -> np.ndarray:
    """温度テーブルから線形補間で物性値を返す。

    テーブル範囲外の温度は端点値でクランプする（外挿なし）。

    Parameters
    ----------
    table : ThermalTable
        平滑化済みの物性値テーブル。
    T : float or np.ndarray
        補間したい温度 [°C]。スカラーまたは配列。

    Returns
    -------
    np.ndarray
        補間された物性値。

    Notes
    -----
    numpy.interp は左端は x[0] 以下でも fp[0] を返し（クランプ）、
    右端は x[-1] 以上でも fp[-1] を返す（クランプ）。
    """
    T_arr = np.asarray(T, dtype=float)
    xp = np.array([t for t, _ in table], dtype=float)
    fp = np.array([v for _, v in table], dtype=float)
    return np.interp(T_arr, xp, fp)


# ---------------------------------------------------------------------------
# 材料クラス
# ---------------------------------------------------------------------------

class WoodProperties:
    """Eurocode 5 Annex B の温度依存物性値を持つ木材クラス。

    FVM1DSolver の props 引数として使用する。
    get_k_array(T) と get_rho_cp_array(T) の両メソッドを実装している。

    Parameters
    ----------
    rho_0 : float
        初期乾燥密度 [kg/m³]。スギ材の場合 350〜450 kg/m³ 程度。
    moisture_content : float
        含水率（質量比）。0.12 = 12%。
    smooth_half_width : float
        段差のスムージング幅の半分 [°C]。

    Examples
    --------
    >>> props = WoodProperties(rho_0=400.0, moisture_content=0.12)
    >>> T = np.array([20.0, 100.0, 300.0, 600.0])
    >>> k = props.get_k_array(T)
    >>> rho_cp = props.get_rho_cp_array(T)
    """

    def __init__(
        self,
        rho_0: float = 400.0,
        moisture_content: float = 0.12,
        smooth_half_width: float = 2.5,
    ) -> None:
        self.rho_0 = rho_0
        self.moisture_content = moisture_content

        # 各テーブルを生成してスムージング
        self._rho_ratio_table = smooth_table_jumps(
            _make_density_ratio_table(moisture_content),
            smooth_half_width,
        )
        self._cp_table = smooth_table_jumps(
            _make_specific_heat_table(moisture_content),
            smooth_half_width,
        )
        self._k_table = smooth_table_jumps(
            WOOD_THERMAL_CONDUCTIVITY,
            smooth_half_width,
        )

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """熱伝導率 k(T) [W/(m·K)] を返す。

        Parameters
        ----------
        T : np.ndarray
            各セルの温度 [°C]。

        Returns
        -------
        np.ndarray
            各セルの熱伝導率 [W/(m·K)]。
        """
        return table_interp(self._k_table, T)

    def get_rho_cp_array(self, T: np.ndarray) -> np.ndarray:
        """体積熱容量 ρ(T)·cp(T) [J/(m³·K)] を返す。

        密度は ρ(T) = ρ₀ × (ρ/ρ₀)(T) で計算する。
        このうち ρ₀ は初期乾燥密度（常数）。

        Parameters
        ----------
        T : np.ndarray
            各セルの温度 [°C]。

        Returns
        -------
        np.ndarray
            各セルの体積熱容量 [J/(m³·K)]。
        """
        rho_ratio = table_interp(self._rho_ratio_table, T)
        cp = table_interp(self._cp_table, T)
        rho = self.rho_0 * rho_ratio
        # ゼロ除算防止：炭化後の密度が 0 に近づいても最低値を設ける
        rho = np.maximum(rho, 1.0)  # [kg/m³]
        return rho * cp

    def get_density_array(self, T: np.ndarray) -> np.ndarray:
        """密度 ρ(T) [kg/m³] を返す（後処理で炭化量計算に使用）。

        Parameters
        ----------
        T : np.ndarray
            各セルの温度 [°C]。

        Returns
        -------
        np.ndarray
            各セルの密度 [kg/m³]。
        """
        rho_ratio = table_interp(self._rho_ratio_table, T)
        return self.rho_0 * rho_ratio


# ---------------------------------------------------------------------------
# 代表的な材料プリセット
# ---------------------------------------------------------------------------

def make_sugi_properties(
    rho_0: float = 400.0,
    moisture_content: float = 0.12,
) -> WoodProperties:
    """スギ（杉、Cryptomeria japonica）の物性値を返す。

    Parameters
    ----------
    rho_0 : float
        初期乾燥密度 [kg/m³]。スギの標準値は 350〜450 kg/m³。
    moisture_content : float
        含水率（質量比）。建築部材の標準は 0.12（12%）。

    Returns
    -------
    WoodProperties
        Eurocode 5 Annex B のスギ材物性値オブジェクト。
    """
    return WoodProperties(rho_0=rho_0, moisture_content=moisture_content)


# ---------------------------------------------------------------------------
# 材料データベース（Phase 3）
# ---------------------------------------------------------------------------

# 各材料のデフォルトパラメータ（Eurocode 5 Annex B 準拠）
# すべての材料に同じ k(T) テーブルを使用（針葉樹材の代表値）
# 密度のみ材料ごとに異なる
MATERIAL_DB: dict[str, dict] = {

    # ── 針葉樹（Eurocode 5 木材モデル）────────────────────────────────
    "sugi": {
        "name": "スギ（杉）Cryptomeria japonica",
        "rho_0": 400.0,
        "moisture_content": 0.12,
        "standard": "EN 1995-1-2 Annex B",
        "properties_type": "wood",
    },
    "hinoki": {
        "name": "ヒノキ（檜）Chamaecyparis obtusa",
        "rho_0": 430.0,
        "moisture_content": 0.12,
        "standard": "EN 1995-1-2 Annex B",
        "properties_type": "wood",
    },
    "lauan": {
        "name": "ラワン Shorea spp.（広葉樹）",
        "rho_0": 530.0,
        "moisture_content": 0.12,
        "standard": "EN 1995-1-2 Annex B",
        "properties_type": "wood",
    },
    "douglas_fir": {
        "name": "ベイマツ Pseudotsuga menziesii",
        "rho_0": 500.0,
        "moisture_content": 0.12,
        "standard": "EN 1995-1-2 Annex B",
        "properties_type": "wood",
    },
    "spruce": {
        "name": "エゾマツ / トウヒ Picea spp.",
        "rho_0": 380.0,
        "moisture_content": 0.12,
        "standard": "EN 1995-1-2 Annex B",
        "properties_type": "wood",
    },

    # ── 研究室論文から追加した木質系（Eurocode 5 スケール）────────────
    "falcata": {
        "name": "ファルカタ Paraserianthes falcataria（軽量広葉樹）",
        "rho_0": 280.0,        # 林田2018・田村2019 実測 196〜446 kg/m³ 中央値
        "moisture_content": 0.12,
        "k_measured": 0.080,   # 林田2018 λ=0.090, 田村2019 λ=0.070 の平均
        "standard": "鷹野研究室 実測値（2017〜2019年）",
        "properties_type": "wood",
    },
    "kiri": {
        "name": "キリ（桐）Paulownia tomentosa（超軽量広葉樹）",
        "rho_0": 296.0,        # 林田2018 実測
        "moisture_content": 0.12,
        "k_measured": 0.091,   # 吉原2017 実測
        "standard": "鷹野研究室 実測値（2017〜2018年）",
        "properties_type": "wood",
    },
    "akagashi": {
        "name": "アカガシ Quercus acuta（高比重広葉樹）",
        "rho_0": 850.0,        # 林田2018 実測 815〜885 kg/m³
        "moisture_content": 0.12,
        "k_measured": 0.186,   # 柴田2021 使用値
        "standard": "鷹野研究室 実測値（2018〜2021年）",
        "properties_type": "wood",
    },
    "bamboo_glulam": {
        "name": "竹集成材 Moso bamboo GLT",
        "rho_0": 600.0,        # 林田2018 実測 584〜620 kg/m³
        "moisture_content": 0.12,
        "standard": "鷹野研究室 実測値（2018年）",
        "properties_type": "wood",
    },
    "insulation_board": {
        "name": "インシュレーションボード（木質繊維断熱板）",
        "rho_0": 244.0,        # 林田2018 実測 230〜259 kg/m³
        "moisture_content": 0.12,
        "k_measured": 0.058,   # 林田2018 実測
        "standard": "鷹野研究室 実測値（2018〜2021年）",
        "properties_type": "wood",   # 木質系なので炭化モデルを適用
    },

    # ── 非木質系・定数物性値材料（ConstantProperties）────────────────
    "charred_cork": {
        "name": "炭化コルク（遅燃断熱材）",
        "rho_0": 130.0,        # 吉原2017・柴田2021 実測 108〜162 kg/m³
        "moisture_content": 0.0,
        "k": 0.041,            # 複数論文共通実測値
        "cp": 2000.0,          # コルク典型値
        "standard": "鷹野研究室 実測値（2017〜2023年）λ=0.041 W/mK",
        "properties_type": "constant",
        "notes": "有炎燃焼なし（遅燃断熱型）。75mm厚で自消（燃え止まり）を確認。",
    },
    "cork": {
        "name": "コルク（無垢）",
        "rho_0": 127.0,
        "moisture_content": 0.0,
        "k": 0.074,            # 林田2018 実測
        "cp": 2000.0,
        "standard": "林田2018 実測値 λ=0.074 W/mK",
        "properties_type": "constant",
    },
    "glass_wool": {
        "name": "グラスウール（断熱材）",
        "rho_0": 24.0,         # 標準品 24 kg/m³
        "moisture_content": 0.0,
        "k": 0.051,            # 林田2018 文献値（田中俊六ら「最新建築環境工学」2014）
        "cp": 840.0,
        "standard": "文献値（田中俊六ら2014）λ=0.051 W/mK",
        "properties_type": "constant",
    },
}


# ---------------------------------------------------------------------------
# 定数物性値クラス（カスタム材料用）
# ---------------------------------------------------------------------------

class ConstantProperties:
    """温度依存なしの定数物性値クラス。ユーザー定義のカスタム材料に使用する。

    Parameters
    ----------
    k : float
        熱伝導率 [W/m·K]。
    rho : float
        密度 [kg/m³]。
    cp : float
        比熱 [J/kg·K]。
    """

    def __init__(self, k: float, rho: float, cp: float) -> None:
        self.k_val = float(k)
        self.rho_val = float(rho)
        self.cp_val = float(cp)

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """温度に依らず一定の熱伝導率配列を返す。"""
        return np.full(T.shape, self.k_val)

    def get_rho_cp_array(self, T: np.ndarray) -> np.ndarray:
        """温度に依らず一定の ρcp 配列を返す。"""
        return np.full(T.shape, self.rho_val * self.cp_val)


# ---------------------------------------------------------------------------
# 有孔板の等価物性値クラス
# ---------------------------------------------------------------------------

# 空気の物性値（孔内の静止空気）
_AIR_K: float = 0.026        # W/m·K
_AIR_RHO_CP: float = 1206.0  # J/m³·K


class PerforatedWoodProperties:
    """等間隔孔・スリット板の等価均質物性値クラス。

    木材の物性値に空洞率 φ を考慮した等価物性値を計算する（並列混合則）:
        k_eff     = (1-φ) * k_wood(T)     + φ * k_air
        ρcp_eff   = (1-φ) * ρcp_wood(T)  + φ * ρcp_air

    この近似は孔が加熱面と平行に配置（板厚方向に垂直）された場合に妥当。
    実際の 3D 効果（孔端部の温度集中など）は無視される。

    Parameters
    ----------
    base_props : WoodProperties
        ベースとなる木材の物性値オブジェクト。
    void_fraction : float
        空洞率（0〜0.95）。断面積に占める孔・スリットの割合。
    """

    def __init__(self, base_props: WoodProperties, void_fraction: float) -> None:
        self.base = base_props
        self.vf = float(max(0.0, min(0.95, void_fraction)))

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """等価熱伝導率を返す（並列混合則）。"""
        k_wood = self.base.get_k_array(T)
        return (1.0 - self.vf) * k_wood + self.vf * _AIR_K

    def get_rho_cp_array(self, T: np.ndarray) -> np.ndarray:
        """等価体積熱容量を返す（並列混合則）。"""
        rho_cp_wood = self.base.get_rho_cp_array(T)
        return (1.0 - self.vf) * rho_cp_wood + self.vf * _AIR_RHO_CP


# ---------------------------------------------------------------------------
# 既存の make_properties 関数
# ---------------------------------------------------------------------------

def make_properties(
    material: str = "sugi",
    rho_0: float | None = None,
    moisture_content: float | None = None,
) -> WoodProperties:
    """材料名から WoodProperties オブジェクトを生成する。

    MATERIAL_DB のデフォルト値を使用しつつ、
    rho_0 や moisture_content で個別に上書きできる。

    Parameters
    ----------
    material : str
        材料名。MATERIAL_DB のキー（"sugi", "hinoki", "lauan", "douglas_fir", "spruce"）。
    rho_0 : float or None
        初期乾燥密度 [kg/m³]。None の場合はデータベースのデフォルト値を使用。
    moisture_content : float or None
        含水率（質量比）。None の場合はデータベースのデフォルト値を使用。

    Returns
    -------
    WoodProperties
        指定された材料の物性値オブジェクト。

    Raises
    ------
    ValueError
        材料名が MATERIAL_DB に存在しない場合。

    Examples
    --------
    >>> props = make_properties("sugi")
    >>> props = make_properties("hinoki", rho_0=450.0)
    >>> props = make_properties("sugi", moisture_content=0.0)  # 完全乾燥
    """
    if material not in MATERIAL_DB:
        available = ", ".join(MATERIAL_DB.keys())
        raise ValueError(
            f"材料 '{material}' はデータベースにありません。"
            f"使用可能な材料: {available}"
        )

    defaults = MATERIAL_DB[material]
    props_type = defaults.get("properties_type", "wood")

    # 定数物性値材料（非木質系：炭化コルク・グラスウールなど）
    if props_type == "constant":
        return ConstantProperties(
            k=defaults["k"],
            rho=rho_0 if rho_0 is not None else defaults["rho_0"],
            cp=defaults["cp"],
        )

    # 木質系（Eurocode 5 温度依存モデル）
    return WoodProperties(
        rho_0=rho_0 if rho_0 is not None else defaults["rho_0"],
        moisture_content=(
            moisture_content
            if moisture_content is not None
            else defaults["moisture_content"]
        ),
    )
