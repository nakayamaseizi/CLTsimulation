"""
boundary.py
===========
【役割】
境界条件の定義と熱流束計算。

・加熱面（ISO 834）：対流 + 輻射の複合熱流束
・非加熱面：自然対流 + 輻射の冷却熱流束

【加熱面の正味熱流束（Eurocode 5 §3.2.2）】
    q"_net = α_c · (T_g - T_s) + ε_m · ε_f · σ · [(T_g+273)^4 - (T_s+273)^4]

【非加熱面の正味熱流束】
    q"_net = α_c · (T_s - T_∞) + ε_m · σ · [(T_s+273)^4 - (T_∞+273)^4]
    （正方向 = 表面からの放熱）

【実装の工夫】
輻射項は前ステップ温度 T_s^n で評価（ラグ処理）し、
対流項は陰解法で T_s^{n+1} を含む線形項として行列に組み込む。
これにより安定性を保ちながら輻射の非線形性を近似的に扱う。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 物理定数
# ---------------------------------------------------------------------------

# Stefan-Boltzmann 定数 [W/(m²·K⁴)]
SIGMA: float = 5.670374419e-8

# Eurocode 5 §3.2.2 の加熱面境界条件定数
ALPHA_C_HEATED: float = 25.0   # 対流熱伝達率（加熱面）[W/(m²·K)]
EPS_M: float = 0.8             # 材料（木材）の放射率 [-]
EPS_F: float = 1.0             # 火炎側の放射率 [-]

# Eurocode 5 §3.2.2 の非加熱面境界条件定数
ALPHA_C_UNHEATED: float = 9.0  # 対流熱伝達率（非加熱面、自然対流のみ）[W/(m²·K)]

# 周囲温度（非加熱面の外気）
T_AMBIENT: float = 20.0        # [°C]


# ---------------------------------------------------------------------------
# ISO 834 標準加熱曲線
# ---------------------------------------------------------------------------

def iso834_temperature(t_min: float) -> float:
    """ISO 834 標準加熱曲線の気相温度を返す。

    加熱炉内のガス温度を時刻 t [分] の関数として与える。

    【式】
        T_g(t) = 20 + 345 · log₁₀(8t + 1)   [°C]

    t = 0 で T_g = 20°C（室温）に一致する。
    t = 60 min で T_g ≈ 945°C。

    Parameters
    ----------
    t_min : float
        加熱開始からの経過時間 [分]。負の値は 20°C を返す。

    Returns
    -------
    float
        気相温度 [°C]。

    Examples
    --------
    >>> iso834_temperature(0)   # 20.0 [°C]
    >>> iso834_temperature(60)  # ~944.7 [°C]
    """
    if t_min <= 0.0:
        return 20.0
    return 20.0 + 345.0 * np.log10(8.0 * t_min + 1.0)


# ---------------------------------------------------------------------------
# 熱流束計算関数（スカラーおよび配列対応）
# ---------------------------------------------------------------------------

def net_flux_heated_face(
    T_g_C: float | np.ndarray,
    T_s_C: float | np.ndarray,
) -> np.ndarray:
    """加熱面への正味熱流束を計算する [W/m²]。

    正の値 = 材料へ入る熱流束。

    【式】
        q"_net = α_c·(T_g - T_s) + ε_m·ε_f·σ·[(T_g+273)^4 - (T_s+273)^4]

    Parameters
    ----------
    T_g_C : float or np.ndarray
        気相温度（ISO 834 ガス温度）[°C]。
    T_s_C : float or np.ndarray
        材料表面温度 [°C]。

    Returns
    -------
    np.ndarray
        正味熱流束 [W/m²]。
    """
    T_g = np.asarray(T_g_C, dtype=float)
    T_s = np.asarray(T_s_C, dtype=float)

    # 対流熱流束
    q_conv = ALPHA_C_HEATED * (T_g - T_s)

    # 輻射熱流束（単位に注意：ケルビン変換）
    T_g_K = T_g + 273.15
    T_s_K = T_s + 273.15
    q_rad = EPS_M * EPS_F * SIGMA * (T_g_K**4 - T_s_K**4)

    return q_conv + q_rad


def net_flux_unheated_face(
    T_s_C: float | np.ndarray,
    T_inf_C: float = T_AMBIENT,
) -> np.ndarray:
    """非加熱面からの正味放熱量を計算する [W/m²]。

    正の値 = 材料から外部へ出る熱流束（冷却方向が正）。

    【式】
        q"_out = α_c·(T_s - T_∞) + ε_m·σ·[(T_s+273)^4 - (T_∞+273)^4]

    Parameters
    ----------
    T_s_C : float or np.ndarray
        非加熱面の表面温度 [°C]。
    T_inf_C : float
        外気温度 [°C]。デフォルト 20°C。

    Returns
    -------
    np.ndarray
        放熱量 [W/m²]（正 = 外部への放熱）。
    """
    T_s = np.asarray(T_s_C, dtype=float)
    T_inf = float(T_inf_C)

    q_conv = ALPHA_C_UNHEATED * (T_s - T_inf)

    T_s_K = T_s + 273.15
    T_inf_K = T_inf + 273.15
    q_rad = EPS_M * SIGMA * (T_s_K**4 - T_inf_K**4)

    return q_conv + q_rad


# ---------------------------------------------------------------------------
# FVM ソルバー用境界条件クラス
# ---------------------------------------------------------------------------

class ISO834HeatedBC:
    """ISO 834 標準加熱の境界条件（加熱面）。

    FVM1DSolver の bc_left 引数として使用する。
    get_matrix_contrib(T_s, t_s) を呼ぶことで
    行列の寄与項（対流係数・参照温度・輻射ラグ項）を返す。

    【FVM への組み込み方】
    左端セル (i=0) のエネルギー収支に以下を加算：
        diag[0]  += alpha_c
        b[0]     += alpha_c * T_g + q_rad_lagged
    ここで q_rad は前ステップの T_s^n で評価した輻射熱流束（ラグ処理）。

    Parameters
    ----------
    alpha_c : float
        対流熱伝達率 [W/(m²·K)]。デフォルトは Eurocode 5 の 25 W/(m²·K)。
    eps_m : float
        材料表面放射率 [-]。デフォルト 0.8。
    eps_f : float
        火炎放射率 [-]。デフォルト 1.0。
    """

    # クラス識別子（FVM ソルバー内で BC の種類を判別するために使う）
    bc_type: str = "robin_heated"

    def __init__(
        self,
        alpha_c: float = ALPHA_C_HEATED,
        eps_m: float = EPS_M,
        eps_f: float = EPS_F,
    ) -> None:
        self.alpha_c = alpha_c
        self.eps_m = eps_m
        self.eps_f = eps_f

    def get_gas_temperature(self, t_s: float) -> float:
        """時刻 t [秒] における ISO 834 気相温度を返す [°C]。"""
        return iso834_temperature(t_s / 60.0)

    def get_matrix_contrib(
        self, T_s_lagged: float, t_s: float
    ) -> tuple[float, float, float]:
        """FVM 行列への寄与項を返す（輻射の線形化版）。

        【線形化の理由】
        輻射項 q_rad = ε·σ·(T_g⁴ - T_s⁴) は T_s に対して強い非線形性を持つ。
        単純なラグ処理（前ステップの T_s で評価）では、炭化が進んで
        セルの熱容量が小さくなると数値発散を起こす場合がある。

        対策：T_s^n 周りのテイラー展開で輻射を線形化し、係数を行列対角に
        組み込む（h_rad = 4·ε·σ·T_s_K³）。これにより対角項が増大して
        無条件安定性が向上する。

        【線形化の式】
            q_rad(T_s^{n+1}) ≈ q_rad(T_s^n) - 4·ε·σ·(T_s^n_K)³·(T_s^{n+1} - T_s^n)

        → 等価 BC: h_total·(T_eff - T_s^{n+1}) として FVM 行列に組み込む

        where:
            h_total = α_c + 4·ε·σ·(T_s^n_K)³   （対流＋線形化輻射）
            T_eff   = T_s^n + q_net(T_s^n) / h_total  （等価参照温度）
            q_net   = α_c·(T_g - T_s^n) + ε·σ·(T_g_K⁴ - T_s_K⁴)

        Parameters
        ----------
        T_s_lagged : float
            前ステップの表面温度 T_s^n [°C]（線形化展開点）。
        t_s : float
            現在の解析時刻 [秒]（ISO 834 温度の計算に使用）。

        Returns
        -------
        h_total : float
            等価熱伝達率（行列対角に加算）[W/(m²·K)]。
        T_eff : float
            等価参照温度（右辺項 h_total·T_eff の計算に使用）[°C]。
        q_extra : float
            追加熱流束（0：線形化で完全に行列に組み込み済み）[W/m²]。
        """
        T_g = self.get_gas_temperature(t_s)
        T_g_K = T_g + 273.15
        T_s_K = T_s_lagged + 273.15

        # T_s^n での正味熱流束
        q_net_n = (
            self.alpha_c * (T_g - T_s_lagged)
            + self.eps_m * self.eps_f * SIGMA * (T_g_K**4 - T_s_K**4)
        )

        # 線形化による放射等価熱伝達率 [W/(m²·K)]
        h_rad = 4.0 * self.eps_m * self.eps_f * SIGMA * T_s_K**3

        # 等価熱伝達率（対流＋線形化輻射）
        h_total = self.alpha_c + h_rad

        # 等価参照温度：T_eff = T_s^n + q_net(T_s^n) / h_total
        T_eff = T_s_lagged + q_net_n / h_total

        return h_total, T_eff, 0.0


class CoolingFurnaceBC:
    """加熱終了後の炉内自然冷却境界条件（加熱面側）。

    ISO 834 加熱試験では、所定の加熱時間終了後に炉内が自然冷却する。
    この BC は炉内温度の指数減衰を模擬し、放冷フェーズの伝熱計算を可能にする。

    【炉内温度の減衰モデル】
        T_furnace(t') = T_ambient + (T_end - T_ambient) × exp(-t' / τ)
    ここで t' = 加熱終了からの経過時間 [s]、τ = 冷却時定数（デフォルト 4500 s）。

    【実験根拠（鷹野研究室 2023, 2024, 2025）】
    小型電気マッフル炉での観察から、60 分加熱（炉内温度 ~945°C）終了後、
    炉内温度は概ね指数的に低下し、約 3 時間（180 分）で ~100°C まで降下する。
    τ ≈ 75 分 (4500 s) はこのデータより推定した値である。

    Parameters
    ----------
    T_end_C : float
        加熱終了時点の炉内温度 [°C]。通常は t_heat_end における ISO 834 温度。
    t_heat_end_s : float
        加熱終了時刻（絶対時刻）[s]。
    tau_s : float
        炉内冷却時定数 [s]。デフォルト 4500 s（75 分）。
    alpha_c : float
        対流熱伝達率 [W/(m²·K)]。加熱終了後は ISO 834 の 25 W/(m²·K) を維持。
    eps_m : float
        材料表面放射率 [-]。
    eps_f : float
        炉内壁放射率 [-]（加熱時と同じ 1.0）。
    T_ambient : float
        最終到達温度（周囲温度）[°C]。
    """

    bc_type: str = "robin_cooling_furnace"

    def __init__(
        self,
        T_end_C: float,
        t_heat_end_s: float,
        tau_s: float = 2700.0,  # 45分 = 2700s（旧デフォルト 4500s から修正）
        alpha_c: float = ALPHA_C_HEATED,
        eps_m: float = EPS_M,
        eps_f: float = EPS_F,
        T_ambient: float = T_AMBIENT,
    ) -> None:
        self.T_end_C = T_end_C
        self.t_heat_end_s = t_heat_end_s
        self.tau_s = tau_s
        self.alpha_c = alpha_c
        self.eps_m = eps_m
        self.eps_f = eps_f
        self.T_ambient = T_ambient

    def get_furnace_temperature(self, t_s: float) -> float:
        """時刻 t_s [s] における炉内温度を返す [°C]。

        加熱終了前は ISO 834 の最終温度を返す（この BC が使われることはないが安全のため）。
        加熱終了後は指数減衰モデルを適用する。
        """
        t_prime = max(0.0, t_s - self.t_heat_end_s)
        T_f = self.T_ambient + (self.T_end_C - self.T_ambient) * np.exp(-t_prime / self.tau_s)
        return float(T_f)

    def get_matrix_contrib(
        self, T_s_lagged: float, t_s: float
    ) -> tuple[float, float, float]:
        """FVM 行列への寄与項を返す（輻射の線形化版、ISO834HeatedBC と同形式）。

        Parameters
        ----------
        T_s_lagged : float
            前ステップの表面温度 T_s^n [°C]。
        t_s : float
            現在の解析時刻 [秒]。

        Returns
        -------
        h_total, T_eff, q_extra : tuple[float, float, float]
        """
        T_g = self.get_furnace_temperature(t_s)
        T_g_K = T_g + 273.15
        T_s_K = T_s_lagged + 273.15

        q_net_n = (
            self.alpha_c * (T_g - T_s_lagged)
            + self.eps_m * self.eps_f * SIGMA * (T_g_K**4 - T_s_K**4)
        )

        h_rad = 4.0 * self.eps_m * self.eps_f * SIGMA * T_s_K**3
        h_total = self.alpha_c + h_rad
        T_eff = T_s_lagged + q_net_n / h_total

        return h_total, T_eff, 0.0


class ConvRadCoolingBC:
    """自然対流＋輻射の冷却境界条件（非加熱面）。

    FVM1DSolver の bc_right 引数として使用する。

    【FVM への組み込み方】
    右端セル (i=n-1) のエネルギー収支に以下を加算：
        diag[n-1] += alpha_c
        b[n-1]    += alpha_c * T_inf - q_rad_lagged
    ここで q_rad は冷却方向（正 = 外部への放熱）。
    マイナス符号は「熱が出て行く = セルからの損失」を表す。

    Parameters
    ----------
    alpha_c : float
        対流熱伝達率 [W/(m²·K)]。デフォルトは Eurocode 5 の 9 W/(m²·K)。
    eps_m : float
        材料表面放射率 [-]。デフォルト 0.8。
    T_inf : float
        外気温度 [°C]。デフォルト 20°C。
    """

    bc_type: str = "robin_cooling"

    def __init__(
        self,
        alpha_c: float = ALPHA_C_UNHEATED,
        eps_m: float = EPS_M,
        T_inf: float = T_AMBIENT,
    ) -> None:
        self.alpha_c = alpha_c
        self.eps_m = eps_m
        self.T_inf = T_inf

    def get_matrix_contrib(
        self, T_s_lagged: float, t_s: float
    ) -> tuple[float, float, float]:
        """FVM 行列への寄与項を返す（輻射の線形化版）。

        【線形化の理由と方法】
        ISO834HeatedBC と同様に、輻射放熱を T_s^n 周りでテイラー展開して線形化する。
        線形化により等価熱伝達率 h_total = α_c + h_rad が対角項に加算され、
        行列の対角優位性が高まって数値安定性が向上する。

        【線形化の式】
            q_out(T_s^{n+1}) ≈ q_out(T_s^n) + h_total · (T_s^{n+1} - T_s^n)
        → セルへの流入（冷却 = 負）として：
            q_into = h_total · (T_ref - T_s^{n+1})
        where:
            h_total = α_c + 4·ε·σ·(T_s^n_K)³
            T_ref   = T_s^n - q_out(T_s^n) / h_total

        返り値の符号規則：
        q"_net_into_cell = h_total · (T_ref - T_s^{n+1}) + q_extra

        Parameters
        ----------
        T_s_lagged : float
            前ステップの非加熱面温度 T_s^n [°C]（線形化展開点）。
        t_s : float
            現在の解析時刻 [秒]（この BC では未使用）。

        Returns
        -------
        h_total : float
            等価熱伝達率（行列対角に加算）[W/(m²·K)]。
        T_ref : float
            等価参照温度（右辺項 h_total·T_ref の計算に使用）[°C]。
        q_extra : float
            追加熱流束（0：線形化で完全に行列に組み込み済み）[W/m²]。
        """
        T_s_K = T_s_lagged + 273.15
        T_inf_K = self.T_inf + 273.15

        # T_s^n での正味放熱量（外部への放熱が正）
        q_out_n = (
            self.alpha_c * (T_s_lagged - self.T_inf)
            + self.eps_m * SIGMA * (T_s_K**4 - T_inf_K**4)
        )

        # 線形化による輻射等価熱伝達率 [W/(m²·K)]
        h_rad = 4.0 * self.eps_m * SIGMA * T_s_K**3

        # 等価熱伝達率（対流＋線形化輻射）
        h_total = self.alpha_c + h_rad

        # 等価参照温度：T_ref = T_s^n - q_out(T_s^n) / h_total
        T_ref = T_s_lagged - q_out_n / h_total

        return h_total, T_ref, 0.0
