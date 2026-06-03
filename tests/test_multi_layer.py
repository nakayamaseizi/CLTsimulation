"""
test_multi_layer.py
===================
【役割】
Phase 3 の検証テスト：多層 CLT の設定読み込み・メッシュ生成・
物性値マッピング・統合実行の動作確認。

【テスト内容】
1. YAML 設定の読み込みと Pydantic 検証
2. make_multi_layer_mesh のメッシュ形状（総厚・セル数・界面座標）
3. MultiLayerProperties の物性値マッピング（層ごとの正しい割り当て）
4. run_from_yaml による統合実行（炭化速度・遮熱性）

【実行方法】
    cd clt_fire_sim
    pytest tests/test_multi_layer.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clt_fire_sim.config import (
    CLTConfig,
    HeatedBCConfig,
    LayerConfig,
    SpecimenConfig,
    load_config,
)
from clt_fire_sim.materials import WoodProperties, make_properties
from clt_fire_sim.runner import run_from_yaml
from clt_fire_sim.solver.fvm_1d import MultiLayerProperties, make_multi_layer_mesh

# ---------------------------------------------------------------------------
# テスト用定数
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
YAML_5LAYER = CONFIGS_DIR / "clt_5layer_sugi.yaml"


# ---------------------------------------------------------------------------
# YAML 設定の読み込みテスト
# ---------------------------------------------------------------------------

class TestYAMLConfig:
    """YAML 設定ファイルの読み込みと Pydantic 検証のテスト。"""

    def test_load_valid_yaml(self):
        """標準 5 層スギ CLT の YAML が正常に読み込める。"""
        config = load_config(YAML_5LAYER)
        assert config.specimen.name == "5層スギCLT 150mm"
        assert len(config.specimen.layers) == 5

    def test_total_thickness_150mm(self):
        """5 層 × 30mm = 150mm であることを確認。"""
        config = load_config(YAML_5LAYER)
        total_mm = sum(layer.thickness_mm for layer in config.specimen.layers)
        assert abs(total_mm - 150.0) < 0.01

    def test_heated_bc_eurocode_values(self):
        """加熱面 BC が Eurocode 5 規定値に一致。"""
        config = load_config(YAML_5LAYER)
        heated = config.boundary.heated
        assert heated.alpha_c == 25.0
        assert heated.eps_m == 0.8
        assert heated.eps_f == 1.0

    def test_unheated_bc_eurocode_values(self):
        """非加熱面 BC が Eurocode 5 規定値に一致。"""
        config = load_config(YAML_5LAYER)
        unheated = config.boundary.unheated
        assert unheated.alpha_c == 9.0
        assert unheated.eps_m == 0.8

    def test_simulation_params(self):
        """ソルバーパラメータが設定値通りに読み込まれる。"""
        config = load_config(YAML_5LAYER)
        sim = config.simulation
        assert sim.t_end_min == 90.0
        assert sim.n_cells_per_layer == 12
        assert sim.n_picard == 3

    def test_eval_times(self):
        """評価時刻が 60, 75, 90 分の 3 点を含む。"""
        config = load_config(YAML_5LAYER)
        eval_times = config.evaluation.eval_times_min
        assert 60.0 in eval_times
        assert 75.0 in eval_times
        assert 90.0 in eval_times

    def test_layer_material_is_sugi(self):
        """全層の材料が 'sugi' である。"""
        config = load_config(YAML_5LAYER)
        for layer in config.specimen.layers:
            assert layer.material == "sugi"

    def test_validation_negative_thickness_raises(self):
        """負の厚みで ValidationError が発生する。"""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LayerConfig(material="sugi", thickness_mm=-10.0)

    def test_validation_invalid_emissivity_raises(self):
        """放射率が 1 を超えると ValidationError が発生する。"""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            HeatedBCConfig(eps_m=1.5)

    def test_file_not_found_raises(self):
        """存在しないファイルで FileNotFoundError が発生する。"""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_default_boundary_applied_when_omitted(self):
        """boundary セクションを省略しても Eurocode 5 デフォルト値が適用される。"""
        minimal_data = {
            "specimen": {
                "layers": [{"thickness_mm": 30.0}]
            }
        }
        config = CLTConfig(**minimal_data)
        # デフォルト値が正しく適用される
        assert config.boundary.heated.alpha_c == 25.0
        assert config.boundary.unheated.alpha_c == 9.0


# ---------------------------------------------------------------------------
# make_multi_layer_mesh テスト
# ---------------------------------------------------------------------------

class TestMultiLayerMesh:
    """make_multi_layer_mesh 関数のメッシュ形状テスト。"""

    def test_total_thickness_matches(self):
        """生成したメッシュの総厚みが入力値の合計と一致する。"""
        thicknesses = [0.030, 0.025, 0.020]
        mesh = make_multi_layer_mesh(thicknesses, n_cells_per_layer=8)
        expected_total = sum(thicknesses)
        assert abs(mesh.x_faces[-1] - expected_total) < 1e-9

    def test_cell_count_equals_n_layers_times_n_cells(self):
        """総セル数 = 層数 × n_cells_per_layer。"""
        n_per_layer = 12
        mesh = make_multi_layer_mesh([0.030] * 5, n_cells_per_layer=n_per_layer)
        assert mesh.n_cells == 5 * n_per_layer

    def test_layer_boundaries_at_face_coordinates(self):
        """層境界座標（x=30, 60, 90, 120, 150mm）がメッシュ界面に存在する。"""
        thicknesses = [0.030] * 5
        mesh = make_multi_layer_mesh(thicknesses, n_cells_per_layer=10)
        boundaries = np.cumsum([0.0] + thicknesses)
        for b in boundaries:
            min_dist = np.abs(mesh.x_faces - b).min()
            assert min_dist < 1e-9, (
                f"層境界 x={b*1000:.0f}mm がメッシュ界面に存在しません "
                f"（最小距離={min_dist:.2e}m）"
            )

    def test_all_dx_positive(self):
        """全セルの幅が正であること（負のセル幅は物理的に不正）。"""
        mesh = make_multi_layer_mesh([0.030, 0.025], n_cells_per_layer=8)
        assert np.all(mesh.dx > 0)

    def test_starts_at_zero(self):
        """メッシュの左端が x=0 から始まる。"""
        mesh = make_multi_layer_mesh([0.030], n_cells_per_layer=5)
        assert mesh.x_faces[0] == 0.0

    def test_per_layer_cell_count(self):
        """各層に異なるセル数を指定できる。"""
        thicknesses = [0.030, 0.030, 0.030]
        n_cells = [8, 10, 12]
        mesh = make_multi_layer_mesh(thicknesses, n_cells_per_layer=n_cells)
        assert mesh.n_cells == sum(n_cells)


# ---------------------------------------------------------------------------
# MultiLayerProperties テスト
# ---------------------------------------------------------------------------

class TestMultiLayerProperties:
    """MultiLayerProperties クラスの物性値マッピングテスト。"""

    def test_same_material_k_matches_single_layer(self):
        """全層同一材料のとき、k 値が単層 WoodProperties と一致する。"""
        single_props = WoodProperties(rho_0=400.0, moisture_content=0.12)
        multi = MultiLayerProperties(
            layer_thicknesses=[0.030, 0.030, 0.030],
            layer_props=[single_props] * 3,
        )
        x_centers = np.linspace(0.005, 0.085, 20)
        multi.setup(x_centers)

        T = np.full(20, 20.0)
        k_multi = multi.get_k_array(T)
        k_ref = single_props.get_k_array(T)
        np.testing.assert_allclose(k_multi, k_ref, rtol=1e-10)

    def test_different_rho_different_rho_cp(self):
        """層ごとに異なる密度のとき、rho_cp が正しく層ごとに異なる。"""
        props_light = WoodProperties(rho_0=300.0, moisture_content=0.0)
        props_heavy = WoodProperties(rho_0=600.0, moisture_content=0.0)

        multi = MultiLayerProperties(
            layer_thicknesses=[0.030, 0.030],
            layer_props=[props_light, props_heavy],
        )
        # 左層の中心（x=15mm）と右層の中心（x=45mm）
        x_centers = np.array([0.015, 0.045])
        multi.setup(x_centers)

        T = np.array([20.0, 20.0])
        rho_cp = multi.get_rho_cp_array(T)

        # 重い右層の方が rho_cp が大きい
        assert rho_cp[1] > rho_cp[0], (
            f"右層 rho_cp({rho_cp[1]:.0f}) が左層 rho_cp({rho_cp[0]:.0f}) 以下"
        )

    def test_setup_required_before_get_k(self):
        """setup() を呼ばずに get_k_array を呼ぶと RuntimeError。"""
        multi = MultiLayerProperties(
            layer_thicknesses=[0.030],
            layer_props=[WoodProperties()],
        )
        T = np.array([20.0])
        with pytest.raises(RuntimeError, match="setup"):
            multi.get_k_array(T)

    def test_mismatched_lengths_raises(self):
        """層数と物性オブジェクト数が不一致で ValueError。"""
        with pytest.raises(ValueError):
            MultiLayerProperties(
                layer_thicknesses=[0.030, 0.030],
                layer_props=[WoodProperties()],  # 1 個しかない
            )

    def test_cell_count_mismatch_raises(self):
        """setup() 後に長さが違う T 配列を渡すと RuntimeError。"""
        multi = MultiLayerProperties(
            layer_thicknesses=[0.030, 0.030],
            layer_props=[WoodProperties()] * 2,
        )
        multi.setup(np.array([0.015, 0.045]))  # 2 セル分で setup
        T_wrong_size = np.array([20.0, 20.0, 20.0])  # 3 セル
        with pytest.raises(RuntimeError):
            multi.get_k_array(T_wrong_size)

    def test_k_positive_at_all_temps(self):
        """全温度域で熱伝導率が正。"""
        multi = MultiLayerProperties(
            layer_thicknesses=[0.030, 0.030],
            layer_props=[WoodProperties()] * 2,
        )
        x_centers = np.linspace(0.005, 0.055, 20)
        multi.setup(x_centers)

        T = np.linspace(20.0, 1200.0, 20)
        k = multi.get_k_array(T)
        assert np.all(k > 0)

    def test_rho_cp_positive_at_all_temps(self):
        """全温度域で体積熱容量が正。"""
        multi = MultiLayerProperties(
            layer_thicknesses=[0.030],
            layer_props=[WoodProperties(rho_0=400.0)],
        )
        x_centers = np.array([0.015])
        multi.setup(x_centers)

        T_range = np.linspace(20.0, 1200.0, 50)
        for T_val in T_range:
            rho_cp = multi.get_rho_cp_array(np.array([T_val]))
            assert rho_cp[0] > 0, f"T={T_val}°C で rho_cp={rho_cp[0]:.1f} ≤ 0"


# ---------------------------------------------------------------------------
# make_properties（材料DB）テスト
# ---------------------------------------------------------------------------

class TestMakeProperties:
    """materials.make_properties() のテスト。"""

    def test_sugi_returns_wood_properties(self):
        """'sugi' で WoodProperties が返る。"""
        props = make_properties("sugi")
        assert isinstance(props, WoodProperties)

    def test_all_materials_in_db(self):
        """データベース内の全材料でオブジェクトが生成できる。"""
        from clt_fire_sim.materials import MATERIAL_DB
        for name in MATERIAL_DB:
            props = make_properties(name)
            assert isinstance(props, WoodProperties), f"{name} の生成に失敗"

    def test_rho_0_override(self):
        """rho_0 を上書きしたとき、密度が反映される。"""
        props_default = make_properties("sugi")           # rho_0=400
        props_custom = make_properties("sugi", rho_0=500.0)

        T = np.array([20.0])
        # rho_0=500 の方が rho_cp が大きいはず
        assert props_custom.get_rho_cp_array(T) > props_default.get_rho_cp_array(T)

    def test_unknown_material_raises(self):
        """存在しない材料名で ValueError が発生する。"""
        with pytest.raises(ValueError, match="データベースにありません"):
            make_properties("unknown_wood")


# ---------------------------------------------------------------------------
# 統合テスト（run_from_yaml）
# ---------------------------------------------------------------------------

class TestRunnerFromYAML:
    """run_from_yaml による統合実行テスト。"""

    @pytest.fixture(scope="class")
    def result(self):
        """YAML からシミュレーションを実行し、結果をキャッシュ。"""
        return run_from_yaml(YAML_5LAYER)

    def test_result_contains_required_keys(self, result):
        """結果辞書に必要なキーが含まれる。"""
        for key in ("times", "temperatures", "x_centers", "char_depths",
                    "config", "evaluation"):
            assert key in result, f"結果辞書に '{key}' がありません"

    def test_evaluation_has_all_times(self, result):
        """evaluation に 60, 75, 90 分の評価結果が含まれる。"""
        eval_dict = result["evaluation"]
        assert "60min" in eval_dict
        assert "75min" in eval_dict
        assert "90min" in eval_dict

    def test_char_depth_at_60min_reasonable(self, result):
        """60 分の炭化深さが物理的に合理的な範囲（10〜80mm）。"""
        char_mm = result["evaluation"]["60min"]["char_depth_mm"]
        assert 10 < char_mm < 80, f"炭化深さ {char_mm:.1f}mm が合理的範囲外"

    def test_insulation_ok_at_60min(self, result):
        """5 層 150mm CLT で 60 分後に遮熱性基準を満たす。"""
        assert result["evaluation"]["60min"]["insulation_ok"], (
            "60 分後の遮熱性が不合格（非加熱面温度 > 160°C）"
        )

    def test_charring_rate_within_eurocode_range(self, result):
        """炭化速度が Eurocode 5 の ±20% 範囲内（0.52〜0.78 mm/min）。"""
        times_min = result["times"] / 60.0
        char_mm = result["char_depths"] * 1000.0

        # 10〜60 分区間で線形回帰
        mask = (times_min >= 10) & (times_min <= 60) & (char_mm > 1.0)
        if mask.sum() >= 5:
            coeffs = np.polyfit(times_min[mask], char_mm[mask], 1)
            beta = coeffs[0]  # [mm/min]
            assert 0.52 <= beta <= 0.78, (
                f"炭化速度 {beta:.3f} mm/min が Eurocode 5 範囲外"
            )

    def test_temperatures_shape(self, result):
        """温度配列の形状が (n_times, n_cells) である。"""
        T = result["temperatures"]
        x = result["x_centers"]
        assert T.ndim == 2
        assert T.shape[1] == len(x)

    def test_config_preserved_in_result(self, result):
        """実行した設定が結果に保存されている。"""
        config = result["config"]
        assert isinstance(config, CLTConfig)
        assert len(config.specimen.layers) == 5
