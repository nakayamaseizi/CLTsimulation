"""
test_iso834.py
==============
【役割】
ISO 834 標準加熱曲線と境界条件熱流束の数値検証テスト。

【テスト内容】
1. ISO 834 曲線の数値が規格値に一致するか
2. 加熱面・非加熱面の熱流束が物理的に正しい符号・大きさか
3. 室温初期条件での熱流束の整合性
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clt_fire_sim.boundary import (
    ALPHA_C_HEATED,
    ALPHA_C_UNHEATED,
    EPS_F,
    EPS_M,
    SIGMA,
    ISO834HeatedBC,
    ConvRadCoolingBC,
    iso834_temperature,
    net_flux_heated_face,
    net_flux_unheated_face,
)


class TestISO834Curve:
    """ISO 834 曲線の数値検証。"""

    def test_initial_temperature(self):
        """t=0 で 20°C（室温）に一致する。"""
        T = iso834_temperature(0.0)
        assert abs(T - 20.0) < 0.1

    def test_at_5_minutes(self):
        """t=5 分で規格値に近い（約 576°C）。"""
        T = iso834_temperature(5.0)
        # T = 20 + 345*log10(8*5+1) = 20 + 345*log10(41) ≈ 20 + 556 = 576
        expected = 20 + 345 * np.log10(8 * 5 + 1)
        assert abs(T - expected) < 0.1

    def test_at_60_minutes(self):
        """t=60 分で約 945°C（耐火評価の標準時刻）。"""
        T = iso834_temperature(60.0)
        expected = 20 + 345 * np.log10(8 * 60 + 1)
        assert abs(T - expected) < 0.1
        # 物理的に 900〜1000°C の範囲内
        assert 900 < T < 1000

    def test_at_90_minutes(self):
        """t=90 分で約 1006°C（準耐火上限の評価時刻）。"""
        T = iso834_temperature(90.0)
        assert 990 < T < 1020

    def test_monotonically_increasing(self):
        """時間とともに単調増加すること。"""
        times = np.linspace(0, 120, 100)
        temps = [iso834_temperature(t) for t in times]
        diffs = np.diff(temps)
        assert np.all(diffs >= 0), "ISO 834 曲線が単調増加でありません"

    def test_negative_time_returns_ambient(self):
        """負の時刻では室温 20°C を返す（安全側の処理）。"""
        T = iso834_temperature(-1.0)
        assert T == 20.0


class TestHeatedFaceFlux:
    """加熱面の熱流束計算テスト。"""

    def test_flux_at_start_high(self):
        """加熱開始直後（t≈0）は大きな熱流束が生じる。

        T_g ≈ 345°C（t=1min 時）、T_s = 20°C の時:
        対流: 25*(345-20) = 8125 W/m²
        輻射: 0.8*5.67e-8*(618^4-293^4) ≈ 大きな正の値
        合計は少なくとも 10 kW/m² を超えるはず。
        """
        T_g = iso834_temperature(1.0)
        T_s = 20.0
        q = net_flux_heated_face(T_g, T_s)
        assert q > 10000, f"t=1min の熱流束が小さすぎます: {q:.0f} W/m²"

    def test_flux_direction_positive(self):
        """ガス温度 > 表面温度のとき、熱流束は正（材料に入る方向）。"""
        T_g = 800.0
        T_s = 200.0
        q = net_flux_heated_face(T_g, T_s)
        assert q > 0, f"熱流束の符号が誤りです: {q:.0f} W/m²"

    def test_flux_zero_at_thermal_equilibrium(self):
        """ガス温度 = 表面温度のとき、熱流束はゼロ（熱平衡）。"""
        T = 500.0
        q = net_flux_heated_face(T, T)
        assert abs(q) < 1e-3, f"熱平衡時の熱流束がゼロでありません: {q:.6f} W/m²"

    def test_max_flux_around_100kWm2(self):
        """解析初期（大きな温度差）でも 200 kW/m² 以下。

        ISO 834 加熱の標準的な熱流束は最大で ~100 kW/m² 程度。
        過剰な値が出ていないかの正気チェック。
        """
        T_g = iso834_temperature(60.0)  # ~945°C
        T_s = 20.0
        q = net_flux_heated_face(T_g, T_s) / 1000  # [kW/m²]
        print(f"\nT_g={T_g:.0f}°C, T_s=20°C: q={q:.1f} kW/m²")
        # 30〜200 kW/m² の範囲が物理的に妥当
        assert 30 < q < 200, f"熱流束 {q:.1f} kW/m² が物理的に不合理な範囲"


class TestUnheatedFaceFlux:
    """非加熱面の放熱量計算テスト。"""

    def test_flux_zero_at_ambient(self):
        """表面温度 = 外気温度のとき、放熱量はゼロ。"""
        q = net_flux_unheated_face(T_s_C=20.0, T_inf_C=20.0)
        assert abs(q) < 1e-3, f"室温での放熱量がゼロでありません: {q:.6f} W/m²"

    def test_flux_positive_when_hot(self):
        """表面温度 > 外気温度のとき、放熱量は正（外部への熱流出）。"""
        q = net_flux_unheated_face(T_s_C=60.0, T_inf_C=20.0)
        assert q > 0, f"高温表面からの放熱量が負です: {q:.0f} W/m²"

    def test_flux_convection_component(self):
        """対流熱流束 = α_c × ΔT が計算に反映されているか（概算チェック）。"""
        T_s = 100.0
        T_inf = 20.0
        q = net_flux_unheated_face(T_s, T_inf)
        q_conv_expected = ALPHA_C_UNHEATED * (T_s - T_inf)  # 9 × 80 = 720 W/m²

        # 全体は輻射を含むので対流分より大きいはず
        assert q > q_conv_expected, "熱流束が対流だけより小さくなっています"


class TestISO834HeatedBC:
    """ISO834HeatedBC クラスの動作テスト。"""

    def test_get_gas_temperature(self):
        """get_gas_temperature が iso834_temperature と一致。"""
        bc = ISO834HeatedBC()
        for t_s in [0, 600, 3600]:
            T_bc = bc.get_gas_temperature(t_s)
            T_direct = iso834_temperature(t_s / 60.0)
            assert abs(T_bc - T_direct) < 0.01

    def test_get_matrix_contrib_returns_three_values(self):
        """get_matrix_contrib が 3 要素のタプルを返す。"""
        bc = ISO834HeatedBC()
        result = bc.get_matrix_contrib(T_s_lagged=20.0, t_s=60.0)
        assert len(result) == 3
        alpha_c, T_g, q_rad = result
        assert alpha_c > 0
        assert T_g > 20.0  # ISO 834 の気相温度は室温より高い

    def test_alpha_c_equals_eurocode_value(self):
        """対流係数が Eurocode 5 規定値 (25 W/m²K) に等しい（クラス属性）。

        線形化後の get_matrix_contrib は h_total = alpha_c + h_rad を返すため、
        alpha_c の値はクラス属性で直接確認する。
        """
        bc = ISO834HeatedBC()
        assert abs(bc.alpha_c - 25.0) < 1e-10

    def test_h_total_larger_than_alpha_c(self):
        """線形化により h_total = alpha_c + h_rad > alpha_c が行列対角に加算される。"""
        bc = ISO834HeatedBC()
        h_total, _, q_extra = bc.get_matrix_contrib(T_s_lagged=20.0, t_s=60.0)
        assert h_total > bc.alpha_c, f"h_total={h_total:.1f} が alpha_c={bc.alpha_c} 以下"
        assert abs(q_extra) < 1e-6, "線形化版では q_extra=0 であるべき"


class TestConvRadCoolingBC:
    """ConvRadCoolingBC クラスの動作テスト。"""

    def test_cooling_effect_when_hot(self):
        """表面温度 > 外気温度のとき、セルへの正味熱流入が負（冷却方向）。

        線形化版では q_extra=0 だが、T_ref < T_s_lagged となることで
        h_total*(T_ref - T_s) < 0 が冷却を表現する。
        """
        bc = ConvRadCoolingBC()
        T_s_hot = 60.0
        h_total, T_ref, q_extra = bc.get_matrix_contrib(T_s_lagged=T_s_hot, t_s=3600.0)
        net_into_cell = h_total * (T_ref - T_s_hot)
        assert net_into_cell < 0, f"高温表面からの熱流入が正です: {net_into_cell:.1f} W/m2"

    def test_equilibrium_at_ambient(self):
        """表面温度 = 外気温度のとき、T_ref = T_inf（熱平衡で正味熱流入ゼロ）。"""
        bc = ConvRadCoolingBC()
        h_total, T_ref, q_extra = bc.get_matrix_contrib(T_s_lagged=20.0, t_s=0.0)
        assert abs(T_ref - bc.T_inf) < 1e-6, f"熱平衡時の T_ref={T_ref:.6f} が T_inf={bc.T_inf} と異なります"
        assert abs(q_extra) < 1e-6, "線形化版では q_extra=0 であるべき"
