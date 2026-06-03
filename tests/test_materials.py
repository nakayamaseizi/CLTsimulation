"""
test_materials.py
=================
【役割】
materials.py の単体テスト。
物性値テーブルの補間が物理的に正しく動作しているかを検証する。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clt_fire_sim.materials import (
    WoodProperties,
    make_sugi_properties,
    smooth_table_jumps,
    table_interp,
    WOOD_THERMAL_CONDUCTIVITY,
    _make_density_ratio_table,
    _make_specific_heat_table,
)


class TestSmoothTableJumps:
    """smooth_table_jumps の動作テスト。"""

    def test_no_jumps(self):
        """段差がない場合はテーブルを変更しない。"""
        table = [(20, 1.0), (100, 2.0), (200, 3.0)]
        result = smooth_table_jumps(table)
        assert len(result) == len(table)

    def test_single_jump(self):
        """1つの段差を正しくスムージングする。"""
        table = [(50, 1.0), (99, 1.0), (99, 5.0), (200, 5.0)]
        result = smooth_table_jumps(table, smooth_half_width=2.5)
        # (99, 1.0), (99, 5.0) → (96.5, 1.0), (101.5, 5.0)
        assert len(result) == 4
        # スムージング後の温度が広がっているか
        temps = [t for t, _ in result]
        assert 96.5 in temps or any(abs(t - 96.5) < 0.01 for t in temps)
        assert any(abs(t - 101.5) < 0.01 for t in temps)

    def test_two_jumps(self):
        """2つの段差（99°C と 120°C）を処理できる。"""
        table = [
            (20, 100), (99, 200), (99, 5000), (120, 4900), (120, 300), (200, 280)
        ]
        result = smooth_table_jumps(table)
        assert len(result) == 6  # 要素数は変わらない（段差は2点→2点）


class TestTableInterp:
    """table_interp の補間テスト。"""

    def test_at_table_points(self):
        """テーブルの定義点では正確な値を返す。"""
        table = [(0, 1.0), (100, 2.0), (200, 4.0)]
        assert abs(table_interp(table, 0) - 1.0) < 1e-10
        assert abs(table_interp(table, 100) - 2.0) < 1e-10
        assert abs(table_interp(table, 200) - 4.0) < 1e-10

    def test_linear_interpolation(self):
        """2点間で線形補間されているか。"""
        table = [(0, 0.0), (100, 10.0)]
        val = table_interp(table, 50)
        assert abs(val - 5.0) < 1e-10

    def test_clamp_below_range(self):
        """範囲外（低温側）は端点値でクランプ。"""
        table = [(100, 1.0), (200, 2.0)]
        val = table_interp(table, 20)   # 100°C より低い
        assert abs(val - 1.0) < 1e-10

    def test_clamp_above_range(self):
        """範囲外（高温側）は端点値でクランプ。"""
        table = [(100, 1.0), (200, 2.0)]
        val = table_interp(table, 500)  # 200°C より高い
        assert abs(val - 2.0) < 1e-10

    def test_array_input(self):
        """numpy 配列の入力に対応。"""
        table = [(0, 0.0), (100, 10.0)]
        T = np.array([0, 25, 50, 75, 100])
        vals = table_interp(table, T)
        expected = np.array([0, 2.5, 5.0, 7.5, 10.0])
        np.testing.assert_allclose(vals, expected, atol=1e-10)


class TestWoodProperties:
    """WoodProperties クラスの物性値テスト。"""

    def setup_method(self):
        """各テスト前に標準スギ材を初期化。"""
        self.props = make_sugi_properties(rho_0=400.0, moisture_content=0.12)

    def test_k_at_room_temperature(self):
        """常温（20°C）の熱伝導率が Eurocode の値と一致。"""
        T = np.array([20.0])
        k = self.props.get_k_array(T)
        # Eurocode 5: k(20°C) = 0.12 W/(m·K)
        assert abs(k[0] - 0.12) < 0.01

    def test_k_at_high_temperature(self):
        """高温（800°C）の熱伝導率が Eurocode の値と一致。"""
        T = np.array([800.0])
        k = self.props.get_k_array(T)
        # Eurocode 5: k(800°C) = 0.35 W/(m·K)
        assert abs(k[0] - 0.35) < 0.01

    def test_rho_cp_positive(self):
        """全温度域で ρ·cp > 0。"""
        T = np.linspace(20, 1200, 100)
        rho_cp = self.props.get_rho_cp_array(T)
        assert np.all(rho_cp > 0), "ρ·cp が負になっています"

    def test_rho_cp_peak_at_evaporation(self):
        """99〜120°C 付近で cp が大きくなる（水の蒸発潜熱）。"""
        T_normal = np.array([50.0])
        T_evap = np.array([110.0])  # 蒸発帯

        rho_cp_normal = self.props.get_rho_cp_array(T_normal)
        rho_cp_evap = self.props.get_rho_cp_array(T_evap)

        # 蒸発帯では少なくとも 3 倍以上高い
        assert rho_cp_evap[0] > rho_cp_normal[0] * 3.0, (
            f"蒸発帯の ρ·cp = {rho_cp_evap[0]:.0f} が"
            f"常温の ρ·cp = {rho_cp_normal[0]:.0f} の 3 倍以上ありません"
        )

    def test_density_decreases_with_charring(self):
        """炭化が進む（温度が上がる）ほど密度が下がる。"""
        T_before_char = np.array([200.0])
        T_after_char = np.array([600.0])

        rho_before = self.props.get_density_array(T_before_char)
        rho_after = self.props.get_density_array(T_after_char)

        assert rho_before[0] > rho_after[0], (
            "炭化後（高温）の密度が炭化前（低温）より高くなっています"
        )

    def test_k_monotone_in_charring_zone(self):
        """350〜500°C 付近で k が低下した後、再上昇する（Eurocode の挙動）。"""
        T = np.linspace(200, 1200, 50)
        k = self.props.get_k_array(T)

        # 350°C 付近で k が最小値をとるはず
        idx_min = np.argmin(k)
        T_at_min = T[idx_min]
        assert 300 <= T_at_min <= 450, (
            f"k の最小値が {T_at_min:.0f}°C にあります（300〜450°C を期待）"
        )

    def test_dry_wood_no_evaporation_peak(self):
        """含水率 0% の場合は蒸発ピークがない（cp が小さい）。"""
        props_dry = WoodProperties(rho_0=400.0, moisture_content=0.0)
        T_evap = np.array([110.0])

        rho_cp_dry = props_dry.get_rho_cp_array(T_evap)
        rho_cp_wet = self.props.get_rho_cp_array(T_evap)

        # 乾燥材の蒸発帯 cp は湿潤材より大幅に小さい
        assert rho_cp_dry[0] < rho_cp_wet[0] * 0.5, (
            f"乾燥材の rho_cp={rho_cp_dry[0]:.0f} が"
            f"湿潤材の {rho_cp_wet[0]:.0f} の半分未満ではありません"
        )


class TestISO834BoundaryInMaterials:
    """材料物性値が境界条件と組み合わせて使えるかの統合的チェック。"""

    def test_props_interface_compatible_with_constant_props(self):
        """WoodProperties が ConstantProperties と同じインターフェースを持つ。"""
        from clt_fire_sim.solver.fvm_1d import ConstantProperties

        T = np.linspace(20, 600, 20)
        wood = make_sugi_properties()
        const = ConstantProperties(k=0.12, rho=400, cp=1600)

        # 両方とも同じ形の配列を返すか
        k_wood = wood.get_k_array(T)
        k_const = const.get_k_array(T)
        rho_cp_wood = wood.get_rho_cp_array(T)
        rho_cp_const = const.get_rho_cp_array(T)

        assert k_wood.shape == T.shape
        assert k_const.shape == T.shape
        assert rho_cp_wood.shape == T.shape
        assert rho_cp_const.shape == T.shape
