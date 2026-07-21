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
# 物理定数・較正温度
# ---------------------------------------------------------------------------

#: ステファン-ボルツマン定数 [W/(m²·K⁴)]
_SIGMA_SB: float = 5.670374419e-8

#: 空隙・ばら材の熱伝導率の較正温度 [°C]
#: 池畑(2021)・柴田(2021) の実験（炉内 13°C ↔ 33°C）および
#: 鷹野研 HFM 実測（13→33°C, 平均 23.02°C）の平均温度。
#: これらの実測値は伝導・輻射・対流をすべて含む合計値である。
_VOID_CAL_T_C: float = 23.0


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
        k_scale: float = 1.0,
        k_char_factor: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        k_scale : float
            熱伝導率スケール係数（デフォルト 1.0 = Eurocode 5 そのまま）。
            実測 λ が Eurocode 5 の基準値（20°C で 0.12 W/m·K）と異なる樹種に対して
            k_scale = λ_measured / 0.12 として補正する。
            例: ファルカタ λ=0.080 → k_scale=0.080/0.12≈0.667
                アカガシ   λ=0.186 → k_scale=0.186/0.12≈1.550
        k_char_factor : float
            炭化域（T > 300°C）の熱伝導率補正係数（デフォルト 1.0 = 補正なし）。

            【物理的背景】
            純熱伝導モデルは燃焼時の発熱（木材の燃焼熱 ≈ 16 MJ/kg）を陽に扱わない。
            この発熱が炭化フロントを加速させる効果を、炭化域のλを等価的に増大させて
            近似する経験的補正である。

            【推奨値（鷹野研実験値との整合）】
            - 無加工スギ (ρ≈400): k_char_factor = 1.3〜1.5
              (実測 0.81-0.86 mm/min vs Eurocode5 0.676 mm/min → 比率 ≈ 1.20-1.27)
              ただし安全側評価では 1.0 のままが保守的
            - 有孔加工スギ: k_char_factor = 1.5〜1.7（酸素供給による促進）
            - スリット加工スギ（幅10深9）: k_char_factor = 1.9〜2.0

            【注意】
            この係数はキャリブレーション用の経験的補正であり、
            材料・試験炉条件によって変化する。デフォルト 1.0（補正なし）は
            Eurocode 5 の設計用炭化速度 β₀=0.65 mm/min と整合する保守的な値。
            実験再現を目的とする場合のみ 1.0 より大きい値を使用すること。
        """
        self.rho_0 = rho_0
        self.moisture_content = moisture_content
        self.k_scale = float(k_scale)
        self.k_char_factor = float(k_char_factor)

        self._smooth_half_width = smooth_half_width
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
        # 乾燥後（含水率0%）テーブル: 放冷フェーズで蒸発ピークを再適用しないため
        self._cp_table_dry = smooth_table_jumps(
            _make_specific_heat_table(0.0),
            smooth_half_width,
        )
        self._rho_ratio_table_dry = smooth_table_jumps(
            _make_density_ratio_table(0.0),
            smooth_half_width,
        )

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """熱伝導率 k(T) [W/(m·K)] を返す。

        Eurocode 5 の標準曲線に k_scale を乗じて実測値補正済みの値を返す。
        炭化域（T > 300°C）には k_char_factor を追加適用し、
        燃焼発熱による炭化フロント促進効果を近似する。

        Parameters
        ----------
        T : np.ndarray
            各セルの温度 [°C]。

        Returns
        -------
        np.ndarray
            各セルの熱伝導率 [W/(m·K)]。
        """
        T_arr = np.asarray(T, dtype=float)
        k = table_interp(self._k_table, T_arr) * self.k_scale
        if self.k_char_factor != 1.0:
            # 炭化域（300°C 以上）に補正係数を適用
            # 300°C を遷移点として線形ブレンド（±50°C）し数値的滑らかさを確保
            blend = np.clip((T_arr - 250.0) / 100.0, 0.0, 1.0)
            k = k * (1.0 + blend * (self.k_char_factor - 1.0))
        return k

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

    def get_rho_cp_dry_array(self, T: np.ndarray) -> np.ndarray:
        """乾燥後の体積熱容量 [J/(m³·K)] を返す（放冷フェーズ用）。

        120°C を超えて一度乾燥したセルは水分が蒸発済みのため、
        冷却時に 99〜120°C の cp スパイク（蒸発潜熱）を再適用しない。
        含水率 0% のテーブルを使用することでスパイクを除去する。
        """
        rho_ratio = table_interp(self._rho_ratio_table_dry, T)
        cp = table_interp(self._cp_table_dry, T)
        rho = np.maximum(self.rho_0 * rho_ratio, 1.0)
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
        "k_measured": 0.058,   # 林田2018 実測 λ=0.058 W/mK
        # Eurocode 5 室温基準値(0.12 W/mK)に対する k_scale = 0.058/0.12 ≈ 0.483
        # → 全温度域で実測比率を適用した温度依存λ曲線を使用
        "standard": "鷹野研究室 実測値（2018〜2021年）λ=0.058 W/mK",
        "properties_type": "wood",
        "notes": (
            "木質繊維系。温度依存 WoodProperties（k_scale=0.483）を適用。"
            "炭化コルクより高い λ を持つが、同様に遅燃断熱層として機能する。"
        ),
    },

    # ── 不燃木（炭化なし簡略モデル）─────────────────────────────────
    "funen_ki": {
        "name": "不燃木（不燃処理木材・炭化抑制モデル）",
        "rho_0": 400.0,
        "moisture_content": 0.12,
        "standard": "簡略モデル（炭化なし仮定・高温 k 上昇なし）",
        "properties_type": "funen_ki",
        "notes": (
            "火側表面パネル用の不燃処理木材モデル。\n"
            "不燃処理により自消・炭化が抑制されるため、高温でも熱伝導率の急激な上昇なし。\n"
            "k(T): Eurocode 5 木材と同じ低温挙動、350°C 以上は緩やかな上昇に留まる。\n"
            "ρ・cp は未処理スギと同等（熱容量は変わらない）。\n"
            "（注）純熱伝導モデルのため、不燃処理の吸熱反応は再現されない。"
        ),
    },

    # ── 難燃処理木材 ──────────────────────────────────────────────
    "fr_sugi": {
        "name": "不燃処理スギ（難燃薬剤注入 180 kg/m³）",
        "rho_0": 400.0,        # スギと同等（薬剤注入で密度は微増するが熱計算上は同じ）
        "moisture_content": 0.12,
        "standard": "伯耆原ら2019・鷹野研究室（中村2022, 中尾2024）",
        "properties_type": "wood",
        "notes": (
            "燃え止まり層（燃え止まり型木質耐火部材）に用いる難燃処理スギ材。\n"
            "薬剤（アンモニウム系難燃剤等）を目標 180 kg/m³ 注入。\n"
            "薬剤分解温度 195°C で吸熱反応が起こり、燃焼を自消させる（燃え止まり機序）。\n"
            "【重要】熱伝導率・比熱・密度は無処理スギと同等とした（文献データなし）。\n"
            "難燃薬剤の吸熱反応（化学的燃え止まり）は純熱伝導モデルでは再現されないため、\n"
            "シミュレーターの燃え止まり判定は保守側（NG方向）に評価される。\n"
            "参考文献: 伯耆原ら「燃え止まり型木質耐火構造部材」日本建築学会環境系論文集 2019"
        ),
    },

    # ── 非木質系・温度依存物性値材料 ────────────────────────────────
    "charred_cork": {
        "name": "炭化コルク（遅燃断熱材）",
        "rho_0": 130.0,        # 吉原2017・柴田2021 実測 108〜162 kg/m³
        "moisture_content": 0.0,
        "k": 0.041,            # 室温実測値（複数論文共通）
        "cp": 2000.0,          # 室温比熱（コルク典型値）
        "standard": "鷹野研究室 実測値（2017〜2023年）λ=0.041 W/mK",
        "properties_type": "charred_cork",   # ← 温度依存モデルに変更
        "notes": (
            "有炎燃焼なし（遅燃断熱型）。75mm厚で自消（燃え止まり）を確認。"
            "温度依存λ(T)モデルを使用（CharredCorkProperties）。"
        ),
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

    # ── 農業系副産物（籾殻・籾殻くん炭）───────────────────────────────
    "momigara": {
        "name": "籾殻（もみ殻・ばら充填）",
        "rho_0": 125.0,        # 実測かさ密度（2026-07-21）
        "moisture_content": 0.10,  # 平衡含水率 8〜12% の代表値
        # k_measured は指定しない：かさ密度から loose_fill_k_rt() で自動算出
        "density_dependent_k": True,
        "standard": "鷹野研究室 実測値（2026-07-21 HFM法）λ=0.0645 W/mK @ρ125",
        "properties_type": "loose_fill",
        "notes": (
            "籾殻ばら充填層。\n"
            "【熱伝導率】実測 λ=0.06450 W/mK（ρ=125 kg/m³、平均温度23.0°C）。\n"
            "  熱流計法(HFM) Wintherm32v3/F314、校正 NIST SRM 1450d、"
            "13→33°C、厚75.2mm、上下差 1.18%。\n"
            "  密度を変えた場合は文献ベースの傾き dλ/dρ=7.3e-5 で外挿する。\n"
            "  緩い 90 kg/m³ → λ≈0.062　実測 125 → λ=0.0645　圧縮 216 → λ≈0.071\n"
            "【傾きの不確かさ】実測は 1 密度のみのため密度依存は文献由来。"
            "充填密度を変数にする場合は複数密度での実測が望ましい。\n"
            "比熱は Eurocode 5 木材テーブル（20°C: 1530 J/kgK）を流用"
            "（Marques et al. 2020 の籾殻実測 1599 J/kgK @40°C とほぼ一致）。\n"
            "籾殻はシリカ（SiO₂）を 15〜20% 含み、燃焼後も断熱性の高い灰骨格が残る。"
            "Eurocode 5 の残存密度比（800°C で 0.26）は籾殻の灰分残存率と近く、"
            "木材モデルの流用は近似として妥当。\n"
            "【注意】ばら材のため自重沈下・対流の影響は本モデルでは考慮されない。"
        ),
    },
    "kuntan": {
        "name": "籾殻くん炭（燻炭・炭化籾殻）",
        "rho_0": 114.35,       # 実測かさ密度（2026-07-21）
        "moisture_content": 0.0,
        # k / k_measured は指定しない：かさ密度から loose_fill_k_rt() で自動算出
        "density_dependent_k": True,
        "standard": "鷹野研究室 実測値（2026-07-21 HFM法）λ=0.0540 W/mK @ρ114",
        "properties_type": "loose_fill",
        "notes": (
            "籾殻を燻焼炭化させた多孔質炭素材（ばら充填）。\n"
            "既に炭化済みのため熱分解・水分蒸発ピークなし（炭化コルクと同型のモデル）。\n"
            "【熱伝導率】実測 λ=0.05404 W/mK（ρ=114.35 kg/m³、平均温度23.0°C）。\n"
            "  熱流計法(HFM) Wintherm32v3/F314、校正 NIST SRM 1450d、"
            "13→33°C、厚75.3mm、上下差 5.53%。\n"
            "  同時測定の籾殻(0.0645)より 16% 低く、炭化による断熱性能の向上を確認。\n"
            "  緩い 90 kg/m³ → λ≈0.052　実測 114 → λ=0.0540　圧縮 216 → λ≈0.062\n"
            "【傾きの不確かさ】実測は 1 密度のみのため密度依存は文献由来。\n"
            "【高温域】室温実測値を基準に λ(T) 曲線全体を同比率でスケール。"
            "多孔質炭素の輻射伝熱による上昇（炭化コルクモデルと同比率）。"
            "高温側は実測がなく推定であることに注意。\n"
            "関野ら(2018, 岩手大)は炭化による断熱性能維持と 450°C 炭化での λ 低下最大を報告。\n"
            "シリカ骨格（灰分 30〜40%）により高温でも形状保持性が高い。\n"
            "【注意】ばら材のため沈下・対流は考慮されない。"
        ),
    },
}


# 不燃木の熱伝導率テーブル
# Eurocode 5 木材と同じ低温挙動、炭化による高温上昇なし（350°C 以上は緩やかな上昇のみ）
_FUNEN_KI_K_TABLE: ThermalTable = [
    (20,    0.12),
    (200,   0.15),
    (350,   0.07),   # Eurocode 木材と同じ低点（炭化が起きないのでそのまま維持）
    (500,   0.08),   # 炭化なし：緩やかな上昇のみ
    (800,   0.10),   # 炭化層の輻射なし
    (1200,  0.13),   # 高温でも Eurocode 木材（1.50）と比べ大幅に低い
]


class FunenKiProperties(WoodProperties):
    """不燃木（不燃処理木材）の物性値モデル。

    通常の木材と同じ密度・比熱を持つが、不燃処理により炭化が抑制されるため
    高温での熱伝導率上昇が大幅に小さい。

    k(T): `_FUNEN_KI_K_TABLE` を使用（炭化域の急上昇なし）
    ρ(T), cp(T): WoodProperties と同じ（Eurocode 5 木材モデル）
    """

    def __init__(
        self,
        rho_0: float = 400.0,
        moisture_content: float = 0.12,
        smooth_half_width: float = 2.5,
    ) -> None:
        super().__init__(
            rho_0=rho_0,
            moisture_content=moisture_content,
            smooth_half_width=smooth_half_width,
            k_scale=1.0,
            k_char_factor=1.0,
        )
        # 炭化なし専用の k(T) テーブルに置き換え
        self._k_table = smooth_table_jumps(_FUNEN_KI_K_TABLE, smooth_half_width)


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
# 炭化コルク温度依存物性値クラス
# ---------------------------------------------------------------------------

# 炭化コルクの熱伝導率 λ(T) [W/(m·K)] 推定テーブル
# 【根拠】
# - 室温（20°C）: 鷹野研実測 λ=0.041 W/mK（複数論文共通）
# - 高温域: 多孔質炭素系材料の文献値およびコルク断熱材の温度依存性から推定
#   (Budaiwi et al., 1999; Dias et al., 2004; ISO/TR4115 等)
# - 200°C 以上では輻射熱伝達が増大し、見かけのλが上昇する（多孔質材料の一般的挙動）
# - 400°C 以上: 細孔構造の収縮・炭素化が進み、急激なλ上昇
# - 朱(2023)の実験では炭化コルク内部が 450-500°C に達することが報告されており、
#   この温度域のλ精度が燃え止まり判定に直接影響する。
_CHARRED_CORK_K_TABLE: ThermalTable = [
    (20,   0.041),   # 鷹野研実測（複数論文共通値）
    (100,  0.047),   # 軽微な上昇（水分放出効果も含む）
    (200,  0.057),   # 輻射項の発現
    (300,  0.068),   # 輻射が顕在化
    (400,  0.085),   # 細孔内輻射が支配的に
    # ↑ キャリブレーション済み（朱 2023: S15+CC75→OK, S15+CC50→NG の境界条件を再現）
    (500,  0.110),   # 炭化進行後の炭素層主導の伝熱
    (600,  0.135),
    (800,  0.190),   # 高温炭素骨格の高輻射伝熱
    # λ(T)/λ(20°C): 室温=1.0x, 200°C=1.4x, 400°C=2.2x, 600°C=3.3x, 800°C=4.6x
    # （多孔質炭素系材料の文献的な温度依存性と整合）
]

# 炭化コルクの比熱 cp(T) [J/(kg·K)] 推定テーブル
# - 室温: ~2000 J/kg·K（コルクの典型値、既存モデルと一致）
# - 高温: 炭化・ガス化により低下。炭素の cp は 700-1000 J/kg·K 程度
_CHARRED_CORK_CP_TABLE: ThermalTable = [
    (20,   2000.0),
    (100,  1900.0),
    (200,  1700.0),
    (300,  1400.0),
    (400,  1150.0),
    (500,  1000.0),
    (600,   900.0),
    (800,   800.0),
]

# 炭化コルクの密度比 ρ(T)/ρ₀ 推定テーブル
# 既に炭化済みのため、ほぼ不変。ただし 600°C 以上では若干の収縮・揮発
_CHARRED_CORK_RHO_RATIO_TABLE: ThermalTable = [
    (20,   1.00),
    (300,  1.00),
    (500,  0.95),
    (600,  0.90),
    (800,  0.85),
]


class CharredCorkProperties:
    """炭化コルク（遅燃断熱材）の温度依存熱物性クラス。

    既存の `ConstantProperties` による炭化コルクモデルを置き換える精密版。
    室温では実測値 λ=0.041 W/mK を再現し、高温域では多孔質炭素材料の
    文献的挙動（輻射伝熱の増大）を反映した温度依存 λ を使用する。

    【精度改善の動機（朱 2023）】
    炭化コルクを遅燃断熱層として用いた燃え止まり型 CLT 試験では、
    炭化コルク内部温度が加熱中に 450-500°C に達することが報告されている。
    この温度域で λ=0.041 W/mK（室温値）を使い続けると、熱伝達を
    過小評価し、実験より構造層への入熱が遅くなる方向に誤差が生じる。

    【参考文献】
    - Budaiwi et al. (1999), Constr. Build. Mater. 13:149-158
    - Dias et al. (2004), J. Mater. Process. Technol. 155:1555-1560
    - 鷹野研究室実測値 λ=0.041 W/mK（室温、複数論文共通）

    Parameters
    ----------
    rho_0 : float
        初期密度 [kg/m³]。デフォルト 130 kg/m³（鷹野研実測値）。
    smooth_half_width : float
        テーブル段差のスムージング幅の半分 [°C]。
    """

    def __init__(self, rho_0: float = 130.0, smooth_half_width: float = 5.0) -> None:
        self.rho_0 = float(rho_0)
        self._k_table = smooth_table_jumps(_CHARRED_CORK_K_TABLE, smooth_half_width)
        self._cp_table = smooth_table_jumps(_CHARRED_CORK_CP_TABLE, smooth_half_width)
        self._rho_ratio_table = smooth_table_jumps(
            _CHARRED_CORK_RHO_RATIO_TABLE, smooth_half_width
        )

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """温度依存熱伝導率 λ(T) [W/(m·K)] を返す。"""
        return table_interp(self._k_table, np.asarray(T, dtype=float))

    def get_rho_cp_array(self, T: np.ndarray) -> np.ndarray:
        """温度依存体積熱容量 ρ(T)·cp(T) [J/(m³·K)] を返す。"""
        T_arr = np.asarray(T, dtype=float)
        rho_ratio = table_interp(self._rho_ratio_table, T_arr)
        cp = table_interp(self._cp_table, T_arr)
        rho = np.maximum(self.rho_0 * rho_ratio, 1.0)
        return rho * cp


# ---------------------------------------------------------------------------
# ばら充填材（籾殻・くん炭）の密度依存熱伝導率
# ---------------------------------------------------------------------------

# 【背景】
# 籾殻・籾殻くん炭のようなばら材は「詰め方」でかさ密度が大きく変わり、
# 圧縮すると空隙が減って固体伝導経路が増えるため λ が上昇する。
# 同一研究内で ばら 90 kg/m³ → 圧縮 216 kg/m³（2.4倍）の変動と、
# それに伴う λ の有意な増加が報告されている。
#
# 【相関式】以下の 2 点を結ぶ線形近似：
#   ρ =  97 kg/m³ → λ = 0.037 W/mK（最適プロトタイプ実測, MDPI Buildings 2024）
#   ρ = 275 kg/m³ → λ = 0.050 W/mK（「275 kg/m³ 以下で λ≤0.05」の上限点）
#
# 【不確かさ】文献間のばらつきは大きい。同じ ~154 kg/m³ でも
# Yarbrough et al.(2005) は 0.046〜0.057 W/mK と本相関式（≈0.041）より高い。
# 測定法・含水率・品種・炭化温度の差に起因する。本相関式は中央推定であり、
# ±30% 程度の不確かさを持つものとして扱うこと。
# ---- 実測アンカー（鷹野研究室 2026-07-21 熱流計法 HFM）----
# 装置: Wintherm32v3 / F314 (S/N 2256)、校正 NIST SRM 1450d
# 試験条件: 上面 13.0°C / 下面 33.0°C、平均温度 23.02°C、試験体厚 75mm
#   籾殻   : ρ=125.00 kg/m³ → λ=0.06450 W/mK（上下差 1.18%）
#   燻炭   : ρ=114.35 kg/m³ → λ=0.05404 W/mK（上下差 5.53%）
# 平均温度 23°C は空隙モデルの較正温度 _VOID_CAL_T_C と一致する。
_LOOSE_FILL_MEASURED: dict[str, tuple[float, float]] = {
    "momigara": (125.00, 0.06450),
    "kuntan": (114.35, 0.05404),
}

# 実測が無い材料に用いる文献ベースの基準点（MDPI Buildings 2024 ほか）
_LOOSE_FILL_RHO_REF: float = 97.0      # 基準かさ密度 [kg/m³]
_LOOSE_FILL_K_REF: float = 0.037       # 基準密度での λ [W/m·K]

# dλ/dρ [(W/m·K)/(kg/m³)]
# 【重要】実測は各材料 1 密度のみのため、密度依存の傾きは実測から決められない。
# 籾殻と燻炭は別材料なので 2 点を結んで傾きにするのは誤り。
# 傾きは文献の同一材料の圧縮試験（ρ=97→0.037, ρ=275→0.050）から取り、
# 絶対値のみ実測で再アンカーする。傾きの不確かさは残る。
_LOOSE_FILL_SLOPE: float = 7.3e-5
_LOOSE_FILL_RHO_MIN: float = 90.0      # 相関式の適用下限 [kg/m³]
_LOOSE_FILL_RHO_MAX: float = 300.0     # 相関式の適用上限 [kg/m³]


def loose_fill_k_rt(rho_0: float, material: str | None = None) -> float:
    """ばら充填材（籾殻・くん炭）の室温熱伝導率を，かさ密度から算出する。

    充填時の押し込み具合（かさ密度）で断熱性能が変わる効果を表現する。
    密度が高いほど空隙が減り λ が上昇する（＝断熱性能が低下する）。

    material に実測アンカーがある場合はその測定点を基準とし、
    密度依存の傾きだけ文献値を用いる::

        λ(ρ) = λ_measured + slope × (ρ − ρ_measured)

    Parameters
    ----------
    rho_0 : float
        かさ密度 [kg/m³]。適用範囲 90〜300 kg/m³（範囲外はクランプ）。
    material : str or None
        材料キー（"momigara" / "kuntan"）。実測アンカーの選択に使う。
        None または未測定の材料では文献ベースの基準点を用いる。

    Returns
    -------
    float
        室温（≈23°C）熱伝導率 λ [W/(m·K)]。

    Examples
    --------
    >>> round(loose_fill_k_rt(125.0, "momigara"), 5)   # 実測点そのもの
    0.0645
    >>> round(loose_fill_k_rt(114.35, "kuntan"), 5)    # 実測点そのもの
    0.05404
    >>> round(loose_fill_k_rt(97), 4)                  # 実測なし（文献ベース）
    0.037
    """
    rho_ref, k_ref = _LOOSE_FILL_MEASURED.get(
        material or "", (_LOOSE_FILL_RHO_REF, _LOOSE_FILL_K_REF)
    )
    rho = float(np.clip(rho_0, _LOOSE_FILL_RHO_MIN, _LOOSE_FILL_RHO_MAX))
    return k_ref + _LOOSE_FILL_SLOPE * (rho - rho_ref)


# ---------------------------------------------------------------------------
# 籾殻くん炭（燻炭）温度依存物性値クラス
# ---------------------------------------------------------------------------

# 籾殻くん炭の熱伝導率 λ(T) [W/(m·K)] 推定テーブル
# 【根拠】
# - 室温（20°C）: 炭化籾殻ばら材の文献値 λ=0.040〜0.05 W/mK
#   （MDPI Buildings 2024: 密度275 kg/m³以下で λ≤0.05 / 関野ら2018 岩手大）
# - 高温域: 多孔質炭素系材料の輻射伝熱増大（炭化コルク _CHARRED_CORK_K_TABLE
#   と同じ温度依存比率でスケール: 0.050/0.041 ≈ 1.22倍）
# - くん炭は既に炭化済みのため熱分解による構造変化は小さいが、
#   細孔内輻射により 400°C 以上で見かけλが上昇する
_KUNTAN_K_TABLE: ThermalTable = [
    (20,   0.050),   # 文献値（炭化籾殻ばら材）
    (100,  0.057),
    (200,  0.070),   # 輻射項の発現
    (300,  0.083),
    (400,  0.104),   # 細孔内輻射が支配的に
    (500,  0.134),
    (600,  0.165),
    (800,  0.232),   # 高温炭素骨格の輻射伝熱
]

# 籾殻くん炭の比熱 cp(T) [J/(kg·K)] 推定テーブル
# - 炭素（チャー）+ シリカ灰分（30〜40%）の混合材
# - 室温: 炭素 ~710-1000 J/kgK とシリカ ~740 J/kgK の中間 ~900
# - 高温: 炭素の cp は温度と共に上昇（グラファイト: 500°C で ~1400 J/kgK）
_KUNTAN_CP_TABLE: ThermalTable = [
    (20,    900.0),
    (200,  1100.0),
    (400,  1250.0),
    (600,  1350.0),
    (800,  1450.0),
]

# 籾殻くん炭の密度比 ρ(T)/ρ₀ 推定テーブル
# 既に炭化済み + シリカ骨格のためほぼ不変。
# 400°C 以上で残留揮発分の放出による軽微な質量減少
_KUNTAN_RHO_RATIO_TABLE: ThermalTable = [
    (20,   1.00),
    (300,  1.00),
    (500,  0.95),
    (800,  0.88),
]


# ---- 籾殻（未炭化）の比熱・密度比テーブル ----
# 籾殻は未炭化のため、燻炭と違って加熱中に水分蒸発と熱分解を経る。
# 比熱: 室温は Marques et al.(2020) の実測 1599 J/kgK @40°C に整合させ、
# 99〜120°C に含水率 10% 相当の蒸発潜熱ピークを設ける
# （0.10 kg_水 × 2260 kJ/kg ÷ 21°C ≈ 10800 J/kgK を上乗せ）。
# 熱分解後は炭素質となるため燻炭の比熱に漸近させる。
_MOMIGARA_CP_TABLE: ThermalTable = [
    (20,    1550.0),
    (99,    1600.0),
    (99,   12400.0),   # 水分蒸発ピーク開始（段差）
    (120,  12300.0),   # 蒸発ピーク終端
    (120,   1800.0),   # 乾燥後（段差）
    (200,   1900.0),
    (300,   1500.0),   # 熱分解の吸熱を経て炭素質へ
    (400,   1250.0),
    (600,   1350.0),
    (800,   1450.0),
]

# 籾殻の密度比 ρ(T)/ρ₀ 推定テーブル
# 99°C で含水率 10% 分の水分が蒸発し、200〜400°C の熱分解で有機分が揮発。
# シリカ灰分 15〜20% が残るため、完全には失われない。
_MOMIGARA_RHO_RATIO_TABLE: ThermalTable = [
    (20,   1.10),      # 含水率 10% を含む
    (99,   1.10),
    (99,   1.00),      # 水分蒸発（段差）
    (200,  0.98),
    (300,  0.80),      # 熱分解開始
    (400,  0.55),
    (500,  0.45),
    (600,  0.40),
    (800,  0.36),      # 炭素＋シリカ灰分が残存
]


class KuntanProperties:
    """【旧モデル・保守用】籾殻くん炭の一括スケール型 λ(T) モデル。

    .. deprecated::
        `LooseFillPorousProperties`（成分分離モデル）に置き換えられた。
        UI からは選択できない。過去結果の再現・比較用に残している。

    炭化コルクの λ(T) テーブルを室温実測値の比率で一律スケールする方式。
    以下の問題があるため既定では使用しない:

    1. 空気の伝導率を定数（0.026）扱いし、温度上昇（500°C で約2倍）を無視。
       籾殻・燻炭の細孔（~0.5mm）では 600°C 付近まで気体伝導が輻射より
       大きいため、この温度域で系統的に誤差が出る。
    2. 輻射・気体・固体を分離せず一括で T³ スケールするため、
       物理的な内訳が追えず感度解析ができない。
    3. 高温側の形状が炭化コルク由来で、燻炭の実測に基づかない。

    Parameters
    ----------
    rho_0 : float
        かさ密度 [kg/m³]。デフォルト 114.35 kg/m³（実測値）。
    smooth_half_width : float
        テーブル段差のスムージング幅の半分 [°C]。
    """

    #: `_KUNTAN_K_TABLE` が想定する基準室温 λ [W/m·K]（20°C の表値）
    _K_TABLE_RT_REF: float = 0.050

    def __init__(
        self, rho_0: float = 114.35, smooth_half_width: float = 5.0,
    ) -> None:
        self.rho_0 = float(rho_0)
        # かさ密度に応じて λ(T) 曲線全体をスケール
        # （室温値を実測アンカーに合わせ、高温側の温度依存比率は維持する）
        self.k_rt = loose_fill_k_rt(self.rho_0, "kuntan")
        k_scale = self.k_rt / self._K_TABLE_RT_REF
        _k_scaled = [(T, k * k_scale) for T, k in _KUNTAN_K_TABLE]
        self._k_table = smooth_table_jumps(_k_scaled, smooth_half_width)
        self._cp_table = smooth_table_jumps(_KUNTAN_CP_TABLE, smooth_half_width)
        self._rho_ratio_table = smooth_table_jumps(
            _KUNTAN_RHO_RATIO_TABLE, smooth_half_width
        )

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """温度依存熱伝導率 λ(T) [W/(m·K)] を返す。"""
        return table_interp(self._k_table, np.asarray(T, dtype=float))

    def get_rho_cp_array(self, T: np.ndarray) -> np.ndarray:
        """温度依存体積熱容量 ρ(T)·cp(T) [J/(m³·K)] を返す。"""
        T_arr = np.asarray(T, dtype=float)
        rho_ratio = table_interp(self._rho_ratio_table, T_arr)
        cp = table_interp(self._cp_table, T_arr)
        rho = np.maximum(self.rho_0 * rho_ratio, 1.0)
        return rho * cp


# ---------------------------------------------------------------------------
# ばら充填多孔質材の成分分離モデル
# ---------------------------------------------------------------------------

# 乾燥空気の熱伝導率 λ_air(T) [W/(m·K)]（標準大気圧）
# 気体の伝導率は温度とともに上昇する（500°C で室温の約2倍）。
# 旧モデルはこれを定数 0.026 としていたため、細孔径が小さく輻射が
# 効きにくい材料（籾殻・燻炭）で系統誤差が生じていた。
_AIR_K_TABLE: ThermalTable = [
    (20,   0.0257), (100,  0.0314), (200,  0.0386), (300,  0.0454),
    (400,  0.0521), (500,  0.0574), (600,  0.0621), (700,  0.0668),
    (800,  0.0715), (900,  0.0763), (1000, 0.0807), (1200, 0.0890),
]


def air_thermal_conductivity(T: np.ndarray | float) -> np.ndarray:
    """乾燥空気の温度依存熱伝導率 [W/(m·K)] を返す。"""
    return table_interp(_AIR_K_TABLE, np.asarray(T, dtype=float))


class LooseFillPorousProperties:
    """ばら充填多孔質材（籾殻・燻炭）の成分分離型 温度依存熱物性クラス。

    見かけの熱伝導率を物理的な3成分に分解して扱う::

        λ_eff(T) = λ_solid + λ_gas(T) + λ_rad(T)

        λ_gas(T) = λ_air(T)                 気体伝導（実測物性・温度依存）
        λ_rad(T) = 4·ε·σ·T³·d_pore          細孔内輻射（T³則）
        λ_solid  = λ_meas − λ_gas(T_cal) − λ_rad(T_cal)    残差として決定

    固体骨格成分を「室温実測値から他成分を差し引いた残差」として決めるため、
    **較正温度において実測値を厳密に再現する**。そのうえで各成分が固有の
    温度依存性を持つので、一括スケールより物理的整合性が高い。

    【旧モデル（KuntanProperties）からの主な改善】
    - 空気の伝導率が温度依存になる。籾殻・燻炭の細孔（~0.5mm）では
      600°C 付近まで気体伝導が輻射より大きく、この寄与は無視できない。
    - 籾殻に Eurocode 5 木材曲線を流用しなくなる。旧実装では炭化の谷により
      400°C で室温より低い λ（0.041 < 0.065）となり、
      「燃焼中に断熱性能が向上する」という非物理的な挙動を示していた。
    - d_pore・ε が明示的パラメータとなり、感度解析が可能になる。

    【不確かさ】
    高温域の実測がないため d_pore が最大の不確かさ要因である。
    d_pore=0.1〜2.0mm の範囲で 500°C の λ は約3倍変動する
    （室温 λ は実測で固定されるため、ほぼ影響を受けない）。

    Parameters
    ----------
    k_meas : float
        較正温度における実測熱伝導率 [W/(m·K)]。
    rho_0 : float
        かさ密度 [kg/m³]。
    cp_table : ThermalTable
        比熱テーブル [J/(kg·K)]。
    rho_ratio_table : ThermalTable
        密度比テーブル ρ(T)/ρ₀。
    d_pore_mm : float
        代表細孔径 [mm]。輻射成分の実効光路長。
    emissivity : float
        細孔内壁の放射率。籾殻・燻炭とも 0.9（籾殻は加熱により炭化・黒色化
        するため、輻射が効く高温域では燻炭と同等とみなす）。
    T_cal_C : float
        実測の平均温度 [°C]。
    """

    def __init__(
        self,
        k_meas: float,
        rho_0: float,
        cp_table: ThermalTable,
        rho_ratio_table: ThermalTable,
        d_pore_mm: float = 0.5,
        emissivity: float = 0.9,
        T_cal_C: float = _VOID_CAL_T_C,
        smooth_half_width: float = 5.0,
    ) -> None:
        self.k_meas = float(k_meas)
        self.rho_0 = float(rho_0)
        self.d_pore_mm = float(d_pore_mm)
        self.emissivity = float(emissivity)
        self.T_cal_C = float(T_cal_C)

        # 固体骨格成分を残差として決定（較正温度で実測値を厳密再現）
        k_gas_cal = float(air_thermal_conductivity(self.T_cal_C))
        k_rad_cal = self._k_rad(self.T_cal_C)
        self.k_solid = max(self.k_meas - k_gas_cal - k_rad_cal, 0.0)

        self._cp_table = smooth_table_jumps(cp_table, smooth_half_width)
        self._rho_ratio_table = smooth_table_jumps(rho_ratio_table, smooth_half_width)

    def _k_rad(self, T_C):
        """細孔内輻射の等価熱伝導率 4εσT³d [W/(m·K)]。"""
        T_K = np.asarray(T_C, dtype=float) + 273.15
        return (4.0 * self.emissivity * _SIGMA_SB
                * T_K ** 3 * self.d_pore_mm * 1e-3)

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """温度依存熱伝導率 λ(T) = λ_solid + λ_gas(T) + λ_rad(T)。"""
        T_arr = np.asarray(T, dtype=float)
        return self.k_solid + air_thermal_conductivity(T_arr) + self._k_rad(T_arr)

    def get_k_components(self, T: np.ndarray) -> dict:
        """成分ごとの内訳を返す（診断・感度解析の可視化用）。"""
        T_arr = np.asarray(T, dtype=float)
        gas = air_thermal_conductivity(T_arr)
        rad = self._k_rad(T_arr)
        return {
            "solid": np.full_like(T_arr, self.k_solid),
            "gas": gas,
            "radiation": rad,
            "total": self.k_solid + gas + rad,
        }

    def get_rho_cp_array(self, T: np.ndarray) -> np.ndarray:
        """温度依存体積熱容量 ρ(T)·cp(T) [J/(m³·K)] を返す。"""
        T_arr = np.asarray(T, dtype=float)
        rho_ratio = table_interp(self._rho_ratio_table, T_arr)
        cp = table_interp(self._cp_table, T_arr)
        rho = np.maximum(self.rho_0 * rho_ratio, 1.0)
        return rho * cp


# ばら充填材ごとの成分分離モデル用パラメータ
#   d_pore : 代表細孔径 [mm]（輻射の実効光路長）
#   eps    : 細孔内壁の放射率
#
# 放射率は両材料とも 0.9 とする。燻炭は黒色炭素で当初から放射率が高く、
# 籾殻も加熱により炭化して黒色化するため、輻射が効き始める高温域では
# 両者の表面性状は同等とみなせる。
# （籾殻に低い放射率 0.85 を与えると、輻射差が T³ で拡大した結果
#  700°C 付近で λ の大小が逆転する。これは実測に基づかない仮定が
#  生む見かけの現象であり、物理的な裏付けがないため採用しない。）
#
# 細孔径は籾殻のセル構造（100μm〜1mm 程度）の中央値。
# **いずれも実測ではなく推定値**であり、高温 λ の最大の不確かさ要因
# （`LooseFillPorousProperties` 参照）。
_LOOSE_FILL_EMISSIVITY: float = 0.9

_LOOSE_FILL_PORE_PARAMS: dict[str, dict] = {
    "kuntan":   {"d_pore_mm": 0.5, "emissivity": _LOOSE_FILL_EMISSIVITY,
                 "cp": _KUNTAN_CP_TABLE, "rho_ratio": _KUNTAN_RHO_RATIO_TABLE},
    "momigara": {"d_pore_mm": 0.5, "emissivity": _LOOSE_FILL_EMISSIVITY,
                 "cp": _MOMIGARA_CP_TABLE, "rho_ratio": _MOMIGARA_RHO_RATIO_TABLE},
}


def make_loose_fill_properties(
    material: str,
    rho_0: float | None = None,
    d_pore_mm: float | None = None,
    emissivity: float | None = None,
) -> LooseFillPorousProperties:
    """ばら充填材（籾殻・燻炭）の成分分離型物性値オブジェクトを生成する。

    Parameters
    ----------
    material : str
        材料キー（"momigara" / "kuntan"）。
    rho_0 : float or None
        かさ密度 [kg/m³]。None なら MATERIAL_DB の既定値（実測値）。
    d_pore_mm, emissivity : float or None
        感度解析用の上書き。None なら材料既定値。
    """
    if material not in _LOOSE_FILL_PORE_PARAMS:
        raise ValueError(
            f"'{material}' はばら充填材ではありません。"
            f"使用可能: {', '.join(_LOOSE_FILL_PORE_PARAMS)}"
        )
    par = _LOOSE_FILL_PORE_PARAMS[material]
    rho = float(rho_0) if rho_0 is not None else float(MATERIAL_DB[material]["rho_0"])
    return LooseFillPorousProperties(
        k_meas=loose_fill_k_rt(rho, material),
        rho_0=rho,
        cp_table=par["cp"],
        rho_ratio_table=par["rho_ratio"],
        d_pore_mm=par["d_pore_mm"] if d_pore_mm is None else float(d_pore_mm),
        emissivity=(par["emissivity"] if emissivity is None
                    else float(emissivity)),
    )


# ---------------------------------------------------------------------------
# 有孔板の等価物性値クラス
# ---------------------------------------------------------------------------

# 空気の物性値（孔内の静止空気）
_AIR_K: float = 0.026        # W/m·K
_AIR_RHO_CP: float = 1206.0  # J/m³·K


# ---------------------------------------------------------------------------
# 空隙（孔・スリット）の温度依存等価熱伝導率
# ---------------------------------------------------------------------------
# 較正温度の定数はファイル前方（物性クラス定義より前）で宣言済み。


# ---------------------------------------------------------------------------
# ISO 6946 密閉空気層の熱抵抗
# ---------------------------------------------------------------------------

# ISO 6946 / JIS A 2102-1 の密閉空気層の熱抵抗 [m²K/W]
# （水平熱流・両面高放射率 ε>0.8。伝導・輻射・対流をすべて含む規格値）
# 厚さが増しても対流のため 25mm 以上で 0.18 に飽和する。
_ISO6946_GAP_D_MM: list[float] = [0.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0, 300.0]
_ISO6946_GAP_R: list[float] = [0.00, 0.11, 0.13, 0.15, 0.17, 0.18, 0.18, 0.18]

# 池畑(2021) がスリットで対流開始を実測した深さ [mm]
# 「深さ9~12mm の間に空気の対流が起こる境界が存在する」（5.3.2 節）
# 実測 Rs は D=9mm→0.0105, D=12mm→0.0108 とほぼ横ばいで頭打ちを示す。
_SLIT_CONV_ONSET_MM: float = 10.0

# 池畑(2021) の実測が存在する最大スリット深さ [mm]
# これを超える設計は外挿となる。D=21mm の実測では熱抵抗が D=3mm を下回り
# （対流により断熱性能が崩壊）、頭打ちモデルでは危険側の評価になる。
_SLIT_VALIDATED_MAX_MM: float = 12.0


def iso6946_air_gap_resistance(
    d_mm: float,
    plateau_mm: float | None = None,
) -> float:
    """ISO 6946 の密閉空気層熱抵抗 [m²K/W] を返す。

    規格値は伝導・輻射・対流をすべて含む実測ベースの値であり、
    空気層が厚くなるほど対流が発達して熱抵抗が飽和する挙動を内包する。

    Parameters
    ----------
    d_mm : float
        空気層の厚さ [mm]。
    plateau_mm : float or None
        この厚さで熱抵抗を頭打ちにする [mm]。
        スリットのように奥行きが開放され対流が早期に始まる場合、
        規格の飽和点（25mm）より手前で頭打ちさせるのに用いる。

    Returns
    -------
    float
        空気層の熱抵抗 [m²K/W]。

    Examples
    --------
    >>> round(iso6946_air_gap_resistance(10.0), 3)
    0.15
    >>> round(iso6946_air_gap_resistance(50.0), 3)   # 対流により飽和
    0.18
    """
    d = float(d_mm)
    if plateau_mm is not None:
        d = min(d, float(plateau_mm))
    return float(np.interp(d, _ISO6946_GAP_D_MM, _ISO6946_GAP_R))


def void_k_at_temperature(
    k_void_cal: float,
    T_C: np.ndarray,
    T_cal_C: float = _VOID_CAL_T_C,
) -> np.ndarray:
    """空隙の等価熱伝導率を較正温度から任意温度へ外挿する。

    【背景】
    実験式（池畑 Ra1 / スリット Rs）から得られる空隙の等価熱伝導率は
    実験温度（≈23°C）での値であり、伝導・輻射・対流をすべて含む。
    火災時は空隙が数百°C に達し、輻射伝熱が T⁴ 則で急増するため、
    較正値をそのまま使うと空隙を通る熱を大幅に過小評価する。

    【モデル】
    空隙の等価熱伝導率を「静止空気の伝導」と「輻射＋対流」に分解し、
    後者を輻射の温度依存性（h_rad ∝ T³）でスケールする::

        k_void(T) = k_air + (k_void_cal − k_air) × (T/T_cal)³

    実測値を基準にするため、輻射の視野係数（細長い空隙では端面間の
    直接授受が小さい）を陽に扱う必要がなく、二重計上も起きない。

    【効果の目安】(T_cal=23°C)
        T=300°C → 係数 8.9倍   T=500°C → 17.8倍   T=700°C → 30.9倍

    Parameters
    ----------
    k_void_cal : float
        較正温度における空隙の等価熱伝導率 [W/(m·K)]。
    T_C : np.ndarray
        温度 [°C]。
    T_cal_C : float
        較正温度 [°C]。デフォルト 23°C。

    Returns
    -------
    np.ndarray
        温度 T における空隙の等価熱伝導率 [W/(m·K)]。
    """
    T_K = np.asarray(T_C, dtype=float) + 273.15
    T_cal_K = T_cal_C + 273.15
    # 輻射＋対流成分（静止空気の伝導を差し引いた残り）
    k_rc = max(float(k_void_cal) - _AIR_K, 0.0)
    return _AIR_K + k_rc * (np.maximum(T_K, 1.0) / T_cal_K) ** 3


def _series_through_thickness(
    k_processed: np.ndarray,
    k_solid: np.ndarray,
    processed_mm: float,
    total_mm: float,
) -> np.ndarray:
    """加工部と無加工部を厚み方向の直列合成で結合する。

    加工部（深さ d）と残りの無垢部（厚さ t−d）は熱流方向に直列に並ぶため、
    熱抵抗を加算して合成する::

        R = d/k_processed + (t−d)/k_solid
        k_eff = t / R

    （従来は d/t を重みとする熱伝導率の加重平均を用いていたが、
    これは並列経路の合成式であり、直列に並ぶ領域には適用できない。）
    """
    d_m = max(min(float(processed_mm), float(total_mm)), 0.0) * 1e-3
    t_m = max(float(total_mm), 1e-6) * 1e-3
    rest_m = max(t_m - d_m, 0.0)
    R = (d_m / np.maximum(k_processed, 1e-10)
         + rest_m / np.maximum(k_solid, 1e-10))
    return t_m / np.maximum(R, 1e-12)


class PerforatedWoodProperties:
    """等間隔孔・スリット板の等価均質物性値クラス。

    木材の物性値に空洞率 φ を考慮した等価物性値を計算する（並列混合則）:
        k_eff     = (1-φ) * k_wood(T)     + φ * k_void(T)
        ρcp_eff   = (1-φ) * ρcp_wood(T)  + φ * ρcp_void(T)

    孔内は空気（デフォルト）または充填材（籾殻くん炭・籾殻など）。
    この近似は孔が加熱面と平行に配置（板厚方向に垂直）された場合に妥当。
    実際の 3D 効果（孔端部の温度集中など）は無視される。

    Parameters
    ----------
    base_props : WoodProperties
        ベースとなる木材の物性値オブジェクト。
    void_fraction : float
        空洞率（0〜0.95）。断面積に占める孔・スリットの割合。
    filler_props : object or None
        孔・スリットの充填材物性値オブジェクト
        （get_k_array / get_rho_cp_array を持つもの。例: KuntanProperties）。
        None の場合は静止空気として扱う（従来動作）。
    """

    def __init__(
        self,
        base_props: WoodProperties,
        void_fraction: float,
        filler_props=None,
    ) -> None:
        self.base = base_props
        self.vf = float(max(0.0, min(0.95, void_fraction)))
        self.filler = filler_props

    def _void_k(self, T: np.ndarray) -> np.ndarray | float:
        """孔部分の熱伝導率（充填材 or 空気）を返す。"""
        if self.filler is not None:
            return self.filler.get_k_array(np.asarray(T, dtype=float))
        return _AIR_K

    def _void_rho_cp(self, T: np.ndarray) -> np.ndarray | float:
        """孔部分の体積熱容量（充填材 or 空気）を返す。"""
        if self.filler is not None:
            return self.filler.get_rho_cp_array(np.asarray(T, dtype=float))
        return _AIR_RHO_CP

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """等価熱伝導率を返す（並列混合則）。"""
        k_wood = self.base.get_k_array(T)
        return (1.0 - self.vf) * k_wood + self.vf * self._void_k(T)

    def get_rho_cp_array(self, T: np.ndarray) -> np.ndarray:
        """等価体積熱容量を返す（並列混合則）。"""
        rho_cp_wood = self.base.get_rho_cp_array(T)
        return (1.0 - self.vf) * rho_cp_wood + self.vf * self._void_rho_cp(T)

    def get_rho_cp_dry_array(self, T: np.ndarray) -> np.ndarray:
        """乾燥後の等価体積熱容量を返す（放冷フェーズ用）。"""
        rho_cp_wood = self.base.get_rho_cp_dry_array(T)
        return (1.0 - self.vf) * rho_cp_wood + self.vf * self._void_rho_cp(T)


# ---------------------------------------------------------------------------
# 有孔加工精密モデル（池畑実験式、2021）
# ---------------------------------------------------------------------------

# 池畑(2021)実験式の較正基準となる試験体面積 [mm²]
# 表4.3.-5 の有孔加工ラミナ試験体は 300×300mm。(5-8) 式の N はこの試験体内の
# 孔数であり、Ra1 を面積正規化された Ra に戻すには N を掛ける必要がある。
# 論文の試験体から逆算した N: 10Φ27P→121(=11²), 18Φ52P→36(=6²) で
# 正方格子 (300/P)² と一致することを確認済み。
_IKEHATA_REF_AREA_MM2: float = 300.0 * 300.0


def _ra1_ikehata(phi_mm: float, h_mm: float) -> float:
    """池畑(2021)の実験式によって孔一つの単位面積熱抵抗 Ra1 [m²K/W] を返す。

    【式】（鹿児島大学 池畑修士論文 2021 (5)式）
        Ra1 = {(0.7·Φ² + 3·Φ - 8)·H + 5·(Φ² - 13·Φ + 37)} × 10⁻⁶

    適用範囲: Φ = 3〜18 mm、H = 6〜30 mm
    ただし 18Φ・24 mm および 18Φ・30 mm は空気対流が起こるため除外。

    Parameters
    ----------
    phi_mm : float
        孔の直径 [mm]。範囲: 3〜18 mm。
    h_mm : float
        孔の深さ（板厚）[mm]。範囲: 6〜30 mm。

    Returns
    -------
    float
        Ra1 [m²K/W]。
    """
    phi = float(np.clip(phi_mm, 3.0, 18.0))
    h = float(np.clip(h_mm, 6.0, 30.0))
    # 対流起動判定：φ=18mm かつ H≥24mm → 対流域。クランプで Ra1 を低く設定
    if phi >= 18.0 and h >= 24.0:
        h = 18.0  # 対流で熱抵抗が停滞する深さを上限とする
    ra1 = ((0.7 * phi**2 + 3.0 * phi - 8.0) * h
           + 5.0 * (phi**2 - 13.0 * phi + 37.0)) * 1.0e-6
    return max(ra1, 0.0)


class PerforatedWoodAdvanced:
    """池畑(2021)実験式を用いた有孔加工木材の精密等価熱物性クラス。

    単純な並列混合則（PerforatedWoodProperties）と異なり、
    孔径Φと孔深さHの関数として実測に基づいた熱抵抗を使用する。

    【等価熱伝導率の導出】
    試験体単位面積あたりの等価熱抵抗:
        R_eff = (1 - φ) · L/λ_wood(T) + φ · Ra(Φ, H)

    ここで φ = 開孔率（孔の断面積 / 全断面積）、L = 孔深さ [m]。
    等価熱伝導率は L/R_eff として算出する。

    【Ra と Ra1 の区別（重要）】
    池畑(2021)の (5-13) 式が与える Ra1 は「孔**一つ**当たり」の熱抵抗である。
    面積正規化された開孔部熱抵抗 Ra は同論文 (4-3)・(5-8) 式より

        Ra = Ra1 × N          N = 較正試験体(300×300mm)内の孔数

    として求まる。N は開孔率と孔径から N = A_ref・φ / (π(Φ/2)²) で算出する。
    N を掛け忘れると孔の熱抵抗が 2〜3 桁過小になり、孔が「熱の近道」として
    扱われて有孔層の等価熱伝導率が無垢材を上回る（非物理）。

    Ra が示強量である傍証として、開孔率を約10%に揃えた論文の3試験体では
    Ra1 が 33 倍ばらつくのに対し Ra はほぼ一定になる:
        H=18mm : 3Φ→0.234, 10Φ→0.209, 18Φ→0.185 [m²K/W]
    較正点 10Φ27P/H18 で Ra=0.209（論文グラフからの実測換算 0.202）と一致し、
    木材18mm(0.180) < Ra < 静止空気18mm(0.600) の妥当な範囲に収まる。

    【適用範囲の注意】
    論文の較正試験体は開孔率が約10%（3Φ8P/10Φ27P/18Φ52P）のみである。
    (5-8) の分解では N ∝ φ となるため孔の寄与は φ² に比例し、
    開孔率が較正点から離れるほど外挿の不確かさが増す。特に φ < 約8% では
    孔の熱抵抗が木材部を下回り、有孔層の等価λが無垢材をわずかに上回る
    （P=40mm/Φ10 で k≈0.122 > 0.120）。低開孔率域の結果は参考値として扱うこと。

    【適用上の注意】
    - 孔は板の加熱面から非加熱面に向かって直線状に開けられる想定
    - 孔が貫通（H = 板厚）の場合は void_fraction のみで十分
    - 孔が貫通しない場合、孔のない部分は通常の木材として扱う
    - 炭化後は孔内部の熱性状が変わるが、本モデルでは高温時は
      PerforatedWoodProperties（並列混合則）に自動的に切り替える

    Parameters
    ----------
    base_props : WoodProperties
        ベースの木材物性値。
    void_fraction : float
        開孔率（断面積ベース）: 0〜0.95。
    hole_diameter_mm : float
        孔径 [mm]。池畑式の適用範囲: 3〜18 mm。
    hole_depth_mm : float
        孔深さ（= 板の加熱面からの深さ）[mm]。適用範囲: 6〜30 mm。
        板厚と異なる場合（貫通孔でない場合）は、有孔層と無孔層に内部分割する。
    layer_thickness_mm : float
        ラミナ全体の厚さ [mm]。孔が貫通する場合は hole_depth_mm と同じにする。
    """

    def __init__(
        self,
        base_props: WoodProperties,
        void_fraction: float,
        hole_diameter_mm: float = 3.0,
        hole_depth_mm: float = 20.0,
        layer_thickness_mm: float | None = None,
        filler_props=None,
    ) -> None:
        self.base = base_props
        self.vf = float(np.clip(void_fraction, 0.0, 0.95))
        self.phi_mm = float(hole_diameter_mm)
        self.h_mm = float(hole_depth_mm)
        self.t_mm = float(layer_thickness_mm if layer_thickness_mm is not None else hole_depth_mm)
        # 孔の充填材（None = 空気孔）
        # 【重要】池畑(2021)実験式は空気孔（孔内の伝導+輻射+対流）に基づく
        # 実験回帰式のため、充填材がある場合は適用外 → 並列混合則に切り替える
        self.filler = filler_props

        # 池畑式による孔一つ当たりの熱抵抗 Ra1 [m²K/W]（(5-13)式）
        self._ra1 = _ra1_ikehata(self.phi_mm, self.h_mm)
        # 較正試験体(300×300mm)内の孔数 N = A_ref・φ / A_hole
        _a_hole_mm2 = np.pi * (self.phi_mm / 2.0) ** 2
        self.n_holes_ref = (
            _IKEHATA_REF_AREA_MM2 * self.vf / _a_hole_mm2 if _a_hole_mm2 > 1e-9 else 0.0
        )
        # 面積正規化された開孔部熱抵抗 Ra = Ra1 × N [m²K/W]（(4-3)・(5-8)式）
        self.ra = self._ra1 * self.n_holes_ref
        # 較正温度における孔の等価熱伝導率 k = L/Ra（伝導+輻射+対流を含む実測値）
        _L_m = self.h_mm * 1e-3
        self.k_void_cal = (_L_m / self.ra) if self.ra > 1e-12 else _AIR_K

        # 孔深さと板厚の比
        self._h_ratio = min(self.h_mm / max(self.t_mm, 1e-3), 1.0)

    def _void_k(self, T: np.ndarray) -> np.ndarray:
        """孔の等価熱伝導率を返す（充填材 or 空気＋輻射の温度依存）。"""
        if self.filler is not None:
            # 充填時は孔内の対流・輻射が抑制される → 充填材の物性値そのもの
            return self.filler.get_k_array(T)
        return void_k_at_temperature(self.k_void_cal, T)

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """等価熱伝導率を返す。

        【合成方法（物理的整合を優先）】
        1. 孔深さ範囲：木材部と孔は熱流方向に**並列**に並ぶため、
           コンダクタンス（熱伝導率）を面積按分して合成する::
               k_proc = (1−φ)·k_wood(T) + φ·k_void(T)
        2. 孔深さ範囲と無加工部（板厚の残り）は熱流方向に**直列**なので、
           熱抵抗を加算して合成する（`_series_through_thickness`）。

        孔の等価熱伝導率 k_void(T) は、空気孔なら池畑実験式から逆算した
        較正値を輻射の T³ 則で温度外挿し、充填孔なら充填材の物性値を使う。

        （従来は孔深さ範囲を熱抵抗の面積按分で合成し、板厚方向は熱伝導率の
        加重平均で結合していた。前者は直列、後者は並列の合成式であり、
        いずれも並列／直列の対応が逆であった。また 300°C 以上で静止空気の
        並列混合則へ不連続に切り替えていたため、輻射が消えて 30% の段差が
        生じていた。本実装では k_void(T) が連続に温度依存するため段差はない。）
        """
        T = np.asarray(T, dtype=float)
        k_wood = self.base.get_k_array(T)
        k_void = self._void_k(T)

        # 1. 孔深さ範囲：並列合成（コンダクタンス加算）
        k_proc = (1.0 - self.vf) * k_wood + self.vf * k_void

        # 2. 無加工部との直列合成（熱抵抗加算）
        return _series_through_thickness(k_proc, k_wood, self.h_mm, self.t_mm)

    def _void_rho_cp(self, T: np.ndarray) -> np.ndarray | float:
        """孔部分の体積熱容量（充填材 or 空気）を返す。"""
        if self.filler is not None:
            return self.filler.get_rho_cp_array(np.asarray(T, dtype=float))
        return _AIR_RHO_CP

    def get_rho_cp_array(self, T: np.ndarray) -> np.ndarray:
        """等価体積熱容量を返す（並列混合則で近似）。"""
        T = np.asarray(T, dtype=float)
        rho_cp_wood = self.base.get_rho_cp_array(T)
        # 孔深さ部分のみ充填材/空気で置換、それ以外は木材
        rho_cp_eff_holed = (1.0 - self.vf) * rho_cp_wood + self.vf * self._void_rho_cp(T)
        h_ratio = self._h_ratio
        return h_ratio * rho_cp_eff_holed + (1.0 - h_ratio) * rho_cp_wood

    def get_rho_cp_dry_array(self, T: np.ndarray) -> np.ndarray:
        """乾燥後の体積熱容量（蒸発ピーク再適用なし）。"""
        T = np.asarray(T, dtype=float)
        rho_cp_wood = self.base.get_rho_cp_dry_array(T)
        rho_cp_eff_holed = (1.0 - self.vf) * rho_cp_wood + self.vf * self._void_rho_cp(T)
        h_ratio = self._h_ratio
        return h_ratio * rho_cp_eff_holed + (1.0 - h_ratio) * rho_cp_wood


# ---------------------------------------------------------------------------
# スリット加工木材の等価物性値クラス
# ---------------------------------------------------------------------------

class SlittedWoodProperties:
    """スリット加工木材の等価熱物性クラス（池畑 2021 実験データ準拠）。

    ラミナ表面に切込み（スリット）を入れることで空気層を内包させる加工。
    柴田(2021)・池畑(2021)の研究では幅15mm深さ3mmのスリットが実用的と結論付けられている。

    【等価熱伝導率モデル】
    スリットを半密閉空気層として扱う。池畑(2021) 自身が
    「スリット加工部は半密閉の空気層と同様の状態」と述べており、
    これは ISO 6946 / JIS A 2102-1 が規格化している対象そのものである。

    【スリット部熱抵抗 Rs】
    ISO 6946 の密閉空気層熱抵抗（伝導・輻射・対流を含む規格値）を用い、
    対流開始深さ d_conv = 10mm で頭打ちにする::

        Rs = iso6946_air_gap_resistance(D, plateau_mm=d_conv)

    規格の飽和点は 25mm だが、スリットは奥行きが開放されて対流が早く
    始まるため、池畑(2021) が実測した境界（9〜12mm）に合わせて 10mm とした。
    同論文の実測 Rs は D=9mm→0.0105、D=12mm→0.0108 とほぼ横ばいであり、
    D=12mm までの頭打ち挙動は外挿ではなく実測に裏付けられている。

    【なぜ池畑の実験式(5-26)を使わないか】
    同論文はスリットの実験式 Rs1 = {(0.09W²+5W−20)D + (0.06W²+2W−10)}×10⁻⁵
    を導出しているが、Rs1 は「スリット一本当たり」の値で、面積正規化された
    Rs に戻すための規格化（空気量／開孔率／本数の対応）が (5-19) 式から
    一意に読み取れない。孔の Ra1×N と同じ手順で移植すると空隙の等価熱伝導率が
    木材を上回り、同論文自身の結論（熱抵抗 43% 改善）と矛盾する結果になる。
    規格化が確定するまでは ISO 6946 の規格値を用いる方が安全である。

    【既知の限界】
    ISO 6946 も空気層厚さのみの関数であるため、池畑(2021) が実測した
    「幅が広いほど熱抵抗が高い」というスリット固有の傾向（W=5→25mm で
    Rs が約15倍）は再現できない。W は開孔率 φ = W/P を通じてのみ効く。

    【適用範囲】
    深さ D ≤ 12mm（実測の裏付けあり）。D > 12mm は `depth_beyond_validated`
    が True になる。同論文の D=21mm 実測では対流により Rs が D=3mm を下回るが、
    頭打ちモデルはこの低下を再現せず、断熱性能を過大評価（危険側）する。

    Parameters
    ----------
    base_props : WoodProperties
        ベースの木材物性値。
    slit_width_mm : float
        スリット幅 [mm]。
    slit_depth_mm : float
        スリット深さ [mm]。実測の裏付けは D ≤ 12mm。
    slit_pitch_mm : float
        スリット中心間距離 [mm]。
    layer_thickness_mm : float
        ラミナ全体の厚さ [mm]。
    d_conv_mm : float
        対流開始深さ [mm]。デフォルト 10.0 mm（池畑 2021 の実測 9〜12mm より）。
    """

    def __init__(
        self,
        base_props: WoodProperties,
        slit_width_mm: float = 15.0,
        slit_depth_mm: float = 3.0,
        slit_pitch_mm: float = 30.0,
        layer_thickness_mm: float = 30.0,
        d_conv_mm: float = _SLIT_CONV_ONSET_MM,
        filler_props=None,
    ) -> None:
        self.base = base_props
        self.W = float(slit_width_mm)
        self.D = float(slit_depth_mm)
        self.P = float(max(slit_pitch_mm, slit_width_mm + 1.0))
        self.t = float(layer_thickness_mm)
        self.d_conv = float(d_conv_mm)
        # スリットの充填材（None = 空気スリット）
        # 【重要】対流遷移深さ・0.6 補正係数は空気スリットの実験知見のため、
        # 充填材がある場合は適用外 → 充填材との並列混合則に切り替える
        self.filler = filler_props

        # 開孔率（スリット断面積 / 総断面積）= W / P
        self.vf = min(self.W / self.P, 0.95)

        # スリット部の熱抵抗 Rs [m²K/W]（ISO 6946 密閉空気層の規格値）
        # 対流開始深さ d_conv で頭打ちにする。規格の飽和点は 25mm だが、
        # スリットは奥行きが開放され対流が早期に始まるため、池畑(2021) が
        # 実測した 9〜12mm の境界に基づき 10mm を既定値とする。
        self._rs = iso6946_air_gap_resistance(self.D, plateau_mm=self.d_conv)

        # 較正温度におけるスリットの等価熱伝導率
        # 【深さ解釈の統一】空隙が占める幾何学的な深さは D（対流による頭打ちは
        # 熱抵抗の飽和であって、空隙が浅くなるわけではない）。
        # したがって等価熱伝導率は k = D / Rs(D, 頭打ち) とする。
        self.k_void_cal = (
            (self.D * 1e-3) / self._rs if self._rs > 1e-12 else _AIR_K
        )

        # スリット深さの板厚に対する比
        self._d_ratio = min(self.D / max(self.t, 1e-3), 1.0)

    @property
    def rs(self) -> float:
        """スリット部の等価熱抵抗 Rs [m²K/W]（1 枚のラミナあたり）。"""
        return self._rs

    @property
    def depth_beyond_validated(self) -> bool:
        """スリット深さが実測の裏付け範囲（D ≤ 12mm）を超えているか。

        True の場合、対流により熱抵抗が低下する領域を頭打ちモデルで
        近似することになり、断熱性能を過大評価（＝危険側）する恐れがある。
        池畑(2021) の D=21mm 実測では Rs が D=3mm を下回っている。
        """
        return self.D > _SLIT_VALIDATED_MAX_MM

    def _void_k(self, T: np.ndarray) -> np.ndarray:
        """スリットの等価熱伝導率を返す（充填材 or 空気＋輻射の温度依存）。"""
        if self.filler is not None:
            # 充填時はスリット内の対流・輻射が抑制される
            return self.filler.get_k_array(T)
        return void_k_at_temperature(self.k_void_cal, T)

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """等価熱伝導率を返す。

        合成方法は `PerforatedWoodAdvanced.get_k_array` と統一されている::
            1. スリット深さ範囲：木材部とスリットは**並列** → k を面積按分
            2. スリット深さ範囲と無加工部：厚み方向に**直列** → R を加算

        スリットの等価熱伝導率 k_void(T) は、空気なら実験式（対流限界つき）
        から逆算した較正値を輻射の T³ 則で温度外挿し、充填時は充填材の値を使う。
        """
        T = np.asarray(T, dtype=float)
        k_wood = self.base.get_k_array(T)
        k_void = self._void_k(T)

        # 1. スリット深さ範囲：並列合成（コンダクタンス加算）
        k_proc = (1.0 - self.vf) * k_wood + self.vf * k_void

        # 2. 無加工部との直列合成（熱抵抗加算）
        return _series_through_thickness(k_proc, k_wood, self.D, self.t)

    def _void_rho_cp(self, T: np.ndarray) -> np.ndarray | float:
        """スリット部分の体積熱容量（充填材 or 空気）を返す。"""
        if self.filler is not None:
            return self.filler.get_rho_cp_array(np.asarray(T, dtype=float))
        return _AIR_RHO_CP

    def get_rho_cp_array(self, T: np.ndarray) -> np.ndarray:
        """等価体積熱容量を返す（並列混合則で近似）。"""
        T = np.asarray(T, dtype=float)
        rho_cp_wood = self.base.get_rho_cp_array(T)
        rho_cp_eff_slit = (1.0 - self.vf) * rho_cp_wood + self.vf * self._void_rho_cp(T)
        d_ratio = self._d_ratio
        return d_ratio * rho_cp_eff_slit + (1.0 - d_ratio) * rho_cp_wood

    def get_rho_cp_dry_array(self, T: np.ndarray) -> np.ndarray:
        """乾燥後の体積熱容量（蒸発ピーク再適用なし）。"""
        T = np.asarray(T, dtype=float)
        rho_cp_wood = self.base.get_rho_cp_dry_array(T)
        rho_cp_eff_slit = (1.0 - self.vf) * rho_cp_wood + self.vf * self._void_rho_cp(T)
        d_ratio = self._d_ratio
        return d_ratio * rho_cp_eff_slit + (1.0 - d_ratio) * rho_cp_wood


# ---------------------------------------------------------------------------
# 既存の make_properties 関数
# ---------------------------------------------------------------------------

def make_properties(
    material: str = "sugi",
    rho_0: float | None = None,
    moisture_content: float | None = None,
    k_char_factor: float = 1.0,
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

    # 定数物性値材料（非木質系：コルク無垢・グラスウールなど）
    if props_type == "constant":
        return ConstantProperties(
            k=defaults["k"],
            rho=rho_0 if rho_0 is not None else defaults["rho_0"],
            cp=defaults["cp"],
        )

    # 炭化コルク専用温度依存モデル（CharredCorkProperties）
    if props_type == "charred_cork":
        return CharredCorkProperties(
            rho_0=rho_0 if rho_0 is not None else defaults["rho_0"],
        )

    # ばら充填多孔質材（籾殻・燻炭）: 成分分離モデル
    if props_type == "loose_fill":
        return make_loose_fill_properties(material, rho_0)

    # 【旧モデル】燻炭の一括スケール方式。UI からは選択できないが、
    # 過去結果の再現用に properties_type="kuntan" を指定すれば利用できる。
    if props_type == "kuntan":
        return KuntanProperties(
            rho_0=rho_0 if rho_0 is not None else defaults["rho_0"],
        )

    # 不燃木（炭化なし・高温 k 上昇抑制モデル）
    if props_type == "funen_ki":
        return FunenKiProperties(
            rho_0=rho_0 if rho_0 is not None else defaults["rho_0"],
            moisture_content=(
                moisture_content if moisture_content is not None
                else defaults["moisture_content"]
            ),
        )

    # 木質系（Eurocode 5 温度依存モデル）
    # k_scale: 実測熱伝導率が DB に記録されている場合は補正スケールを計算
    # Eurocode 5 の基準値（20°C）= 0.12 W/m·K
    _K_EUROCODE_RT = 0.12  # Eurocode 5 室温熱伝導率 [W/m·K]
    _rho_eff = rho_0 if rho_0 is not None else defaults["rho_0"]
    if defaults.get("density_dependent_k"):
        # ばら充填材（籾殻など）: かさ密度から室温 λ を算出
        # → 押し込み具合（詰め方）が断熱性能に反映される
        # material キーを渡して実測アンカー（籾殻/燻炭）を選択させる
        k_measured = loose_fill_k_rt(_rho_eff, material)
    else:
        k_measured = defaults.get("k_measured")
    k_scale = (k_measured / _K_EUROCODE_RT) if k_measured is not None else 1.0

    return WoodProperties(
        rho_0=rho_0 if rho_0 is not None else defaults["rho_0"],
        moisture_content=(
            moisture_content
            if moisture_content is not None
            else defaults["moisture_content"]
        ),
        k_scale=k_scale,
        k_char_factor=k_char_factor,
    )
