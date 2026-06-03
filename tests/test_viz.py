"""
test_viz.py
===========
【役割】
Phase 5 の検証テスト：可視化・レポート生成モジュールの動作確認。

【テスト内容】
1. plot_charring_progress  ─ 炭化深さグラフが Axes を返すこと
2. plot_temperature_profiles ─ 温度分布グラフが Axes を返すこと
3. plot_surface_history    ─ 表面温度グラフが Axes を返すこと
4. plot_summary_panel      ─ 3 パネル Figure を返し PNG 保存できること
5. save_results_csv        ─ CSV が正しい列構造で保存されること
6. compute_charring_rate   ─ 炭化速度が妥当な範囲に収まること
7. evaluate_performance    ─ 判定辞書のキーと型が正しいこと
8. generate_report         ─ 全出力ファイルが生成されること

【実行方法】
    cd clt_fire_sim
    pytest tests/test_viz.py -v
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # CI / GUI なし環境向け

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clt_fire_sim.boundary import ConvRadCoolingBC, ISO834HeatedBC
from clt_fire_sim.materials import make_sugi_properties
from clt_fire_sim.report import (
    compute_charring_rate,
    evaluate_performance,
    generate_report,
)
from clt_fire_sim.runner import run_from_yaml
from clt_fire_sim.solver.fvm_1d import (
    FVM1DSolver,
    MultiLayerProperties,
    make_multi_layer_mesh,
)
from clt_fire_sim.viz import (
    plot_charring_progress,
    plot_summary_panel,
    plot_surface_history,
    plot_temperature_profiles,
    save_results_csv,
)


# ---------------------------------------------------------------------------
# 共有フィクスチャ
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture(scope="module")
def short_result():
    """10 分の短時間シミュレーション結果（各可視化テストで共有）。"""
    layer_thicknesses = [0.030] * 3
    props_list = [make_sugi_properties(rho_0=400.0) for _ in range(3)]
    mesh = make_multi_layer_mesh(layer_thicknesses, n_cells_per_layer=8)
    props = MultiLayerProperties(layer_thicknesses, props_list)
    props.setup(mesh.x_centers)
    solver = FVM1DSolver(
        mesh, props,
        bc_left=ISO834HeatedBC(),
        bc_right=ConvRadCoolingBC(),
        T_init=20.0,
    )
    return solver.solve(600.0, dt_base=5.0, dt_max=5.0, n_picard=2,
                        record_interval=30.0)


@pytest.fixture(scope="module")
def full_result():
    """90 分フル YAML シミュレーション結果（レポート系テストで共有）。"""
    return run_from_yaml(CONFIGS_DIR / "clt_5layer_sugi.yaml")


# ---------------------------------------------------------------------------
# テスト 1: plot_charring_progress
# ---------------------------------------------------------------------------

class TestPlotCharringProgress:
    """炭化深さグラフのテスト。"""

    def test_returns_axes(self, short_result):
        """Axes オブジェクトが返る。"""
        ax = plot_charring_progress(short_result)
        assert hasattr(ax, "get_xlabel")
        plt.close("all")

    def test_axes_has_correct_labels(self, short_result):
        """x 軸ラベルが '時間 [分]'、y 軸ラベルが '炭化深さ [mm]' である。"""
        ax = plot_charring_progress(short_result)
        assert "時間" in ax.get_xlabel()
        assert "炭化深さ" in ax.get_ylabel()
        plt.close("all")

    def test_axes_has_lines(self, short_result):
        """グラフに複数のラインが描画されている。"""
        ax = plot_charring_progress(short_result)
        assert len(ax.get_lines()) >= 2
        plt.close("all")

    def test_custom_ax_accepted(self, short_result):
        """既存の Axes に追記できる。"""
        fig, ax_in = plt.subplots()
        ax_out = plot_charring_progress(short_result, ax=ax_in)
        assert ax_out is ax_in
        plt.close("all")


# ---------------------------------------------------------------------------
# テスト 2: plot_temperature_profiles
# ---------------------------------------------------------------------------

class TestPlotTemperatureProfiles:
    """温度分布グラフのテスト。"""

    def test_returns_axes(self, short_result):
        """Axes オブジェクトが返る。"""
        ax = plot_temperature_profiles(short_result)
        assert hasattr(ax, "get_xlabel")
        plt.close("all")

    def test_axes_has_correct_labels(self, short_result):
        """x 軸に深さ、y 軸に温度のラベルがある。"""
        ax = plot_temperature_profiles(short_result)
        assert "深さ" in ax.get_xlabel()
        assert "温度" in ax.get_ylabel()
        plt.close("all")

    def test_custom_times_respected(self, short_result):
        """指定した時刻数のラインが描画される（+ 300°C 参照線）。"""
        n_times = 3
        times_min = [2.0, 5.0, 10.0]
        ax = plot_temperature_profiles(short_result, times_min=times_min)
        # n_times 本のプロファイル + 1 本の 300°C 参照線
        assert len(ax.get_lines()) >= n_times + 1
        plt.close("all")


# ---------------------------------------------------------------------------
# テスト 3: plot_surface_history
# ---------------------------------------------------------------------------

class TestPlotSurfaceHistory:
    """表面温度グラフのテスト。"""

    def test_returns_axes(self, short_result):
        """Axes オブジェクトが返る。"""
        ax = plot_surface_history(short_result)
        assert hasattr(ax, "get_xlabel")
        plt.close("all")

    def test_axes_has_correct_labels(self, short_result):
        """x 軸に時間、y 軸に温度のラベルがある。"""
        ax = plot_surface_history(short_result)
        assert "時間" in ax.get_xlabel()
        assert "温度" in ax.get_ylabel()
        plt.close("all")

    def test_has_iso834_and_surface_lines(self, short_result):
        """ISO 834・加熱面・非加熱面・遮熱基準の 4 本以上が描画される。"""
        ax = plot_surface_history(short_result)
        assert len(ax.get_lines()) >= 4
        plt.close("all")


# ---------------------------------------------------------------------------
# テスト 4: plot_summary_panel
# ---------------------------------------------------------------------------

class TestPlotSummaryPanel:
    """3 パネル概要図のテスト。"""

    def test_returns_figure(self, short_result):
        """Figure オブジェクトが返る。"""
        fig = plot_summary_panel(short_result)
        assert hasattr(fig, "get_axes")
        plt.close("all")

    def test_has_three_axes(self, short_result):
        """Figure に Axes が 3 つある。"""
        fig = plot_summary_panel(short_result)
        assert len(fig.get_axes()) == 3
        plt.close("all")

    def test_saves_png(self, short_result, tmp_path):
        """out_path を指定すると PNG ファイルが保存される。"""
        out = tmp_path / "summary.png"
        plot_summary_panel(short_result, out_path=out)
        assert out.exists()
        assert out.stat().st_size > 1000  # 最低限の PNG サイズ
        plt.close("all")

    def test_saves_to_nested_directory(self, short_result, tmp_path):
        """存在しないネストしたディレクトリに保存できる。"""
        out = tmp_path / "a" / "b" / "summary.png"
        plot_summary_panel(short_result, out_path=out)
        assert out.exists()
        plt.close("all")


# ---------------------------------------------------------------------------
# テスト 5: save_results_csv
# ---------------------------------------------------------------------------

class TestSaveResultsCsv:
    """CSV 出力のテスト。"""

    def test_csv_created(self, short_result, tmp_path):
        """CSV ファイルが生成される。"""
        p = save_results_csv(short_result, tmp_path / "res.csv")
        assert p.exists()

    def test_csv_header(self, short_result, tmp_path):
        """ヘッダー行が規定の列名を持つ。"""
        p = save_results_csv(short_result, tmp_path / "res.csv")
        with p.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        expected = ["time_s", "time_min", "char_depth_mm",
                    "T_heated_C", "T_unheated_C", "iso834_gas_C"]
        assert header == expected

    def test_csv_row_count(self, short_result, tmp_path):
        """行数がレコード数と一致する（ヘッダー除く）。"""
        p = save_results_csv(short_result, tmp_path / "res.csv")
        with p.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        data_rows = rows[1:]
        assert len(data_rows) == len(short_result["times"])

    def test_csv_values_are_numeric(self, short_result, tmp_path):
        """各行の全列が数値として解釈できる。"""
        p = save_results_csv(short_result, tmp_path / "res.csv")
        with p.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                for val in row:
                    float(val)  # raises ValueError if non-numeric

    def test_time_increases_monotonically(self, short_result, tmp_path):
        """time_s 列が単調増加する。"""
        p = save_results_csv(short_result, tmp_path / "res.csv")
        with p.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            times = [float(row[0]) for row in reader]
        assert all(a < b for a, b in zip(times, times[1:]))

    def test_iso834_at_t0_is_20_celsius(self, short_result, tmp_path):
        """t=0 の ISO 834 温度は初期温度 20°C である。"""
        p = save_results_csv(short_result, tmp_path / "res.csv")
        with p.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            first_row = next(reader)
        iso_val = float(first_row[5])
        assert abs(iso_val - 20.0) < 1.0


# ---------------------------------------------------------------------------
# テスト 6: compute_charring_rate
# ---------------------------------------------------------------------------

class TestComputeCharringRate:
    """炭化速度算出のテスト。"""

    def test_returns_float(self, full_result):
        """戻り値が float である。"""
        beta = compute_charring_rate(full_result)
        assert isinstance(beta, float)

    def test_eurocode_range(self, full_result):
        """炭化速度が Eurocode 5 の ±20% 範囲（0.52〜0.78 mm/min）内。"""
        beta = compute_charring_rate(full_result)
        assert 0.52 <= beta <= 0.78, f"β={beta:.3f} mm/min が範囲外"

    def test_custom_time_range(self, full_result):
        """カスタム区間でも有効な値が返る。"""
        beta = compute_charring_rate(full_result, t_start_min=20.0, t_end_min=80.0)
        assert not np.isnan(beta)
        assert beta > 0.0

    def test_insufficient_data_returns_nan(self, short_result):
        """回帰区間にデータが 2 点未満のとき NaN を返す。"""
        # short_result は 10 分のシミュレーション; t_start=20 分では点が不足
        beta = compute_charring_rate(short_result, t_start_min=20.0, t_end_min=60.0)
        assert np.isnan(beta)


# ---------------------------------------------------------------------------
# テスト 7: evaluate_performance
# ---------------------------------------------------------------------------

class TestEvaluatePerformance:
    """性能評価のテスト。"""

    def test_returns_dict(self, full_result):
        """辞書が返る。"""
        jdg = evaluate_performance(full_result)
        assert isinstance(jdg, dict)

    def test_default_eval_times(self, full_result):
        """デフォルトで 60, 75, 90 分のキーが存在する。"""
        jdg = evaluate_performance(full_result)
        assert "60min" in jdg
        assert "75min" in jdg
        assert "90min" in jdg

    def test_judgment_keys(self, full_result):
        """各判定辞書に必要なキーが含まれる。"""
        jdg = evaluate_performance(full_result)
        for t_key, entry in jdg.items():
            for k in ("char_depth_mm", "unheated_face_temp_C",
                      "unheated_face_rise_K", "insulation_ok"):
                assert k in entry, f"{t_key}['{k}'] が欠落"

    def test_insulation_ok_type(self, full_result):
        """insulation_ok が bool 型である。"""
        jdg = evaluate_performance(full_result)
        for entry in jdg.values():
            assert isinstance(entry["insulation_ok"], bool)

    def test_char_depth_positive(self, full_result):
        """全評価時刻で炭化深さが正値。"""
        jdg = evaluate_performance(full_result)
        for entry in jdg.values():
            assert entry["char_depth_mm"] > 0.0

    def test_insulation_ok_at_60min(self, full_result):
        """5 層 150mm CLT は 60 分後に遮熱性を満たす。"""
        jdg = evaluate_performance(full_result)
        assert jdg["60min"]["insulation_ok"]

    def test_custom_eval_times(self, full_result):
        """任意の評価時刻リストを受け付ける。"""
        jdg = evaluate_performance(full_result, eval_times_min=[30.0, 60.0])
        assert "30min" in jdg
        assert "60min" in jdg
        assert "90min" not in jdg

    def test_with_config(self, full_result):
        """config を渡したとき結果が辞書を返す。"""
        config = full_result.get("config")
        if config is not None:
            jdg = evaluate_performance(full_result, config=config)
            assert "60min" in jdg


# ---------------------------------------------------------------------------
# テスト 8: generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    """レポート一括生成のテスト。"""

    @pytest.fixture(scope="class")
    def report_paths(self, full_result, tmp_path_factory):
        """レポート生成を一度だけ実行してパス辞書をキャッシュ。"""
        out_dir = tmp_path_factory.mktemp("report")
        config = full_result.get("config")
        return generate_report(full_result, config=config, out_dir=out_dir)

    def test_returns_dict(self, report_paths):
        """戻り値が辞書である。"""
        assert isinstance(report_paths, dict)

    def test_all_expected_keys_present(self, report_paths):
        """summary_png, charring_png, profiles_png, surface_png, csv, txt が揃う。"""
        for key in ("summary_png", "charring_png", "profiles_png",
                    "surface_png", "csv", "txt"):
            assert key in report_paths, f"'{key}' が戻り値辞書に欠落"

    def test_all_files_exist(self, report_paths):
        """全ファイルが実際にディスクに存在する。"""
        for key, path in report_paths.items():
            assert Path(path).exists(), f"'{key}' ({path}) が存在しません"

    def test_png_files_not_empty(self, report_paths):
        """PNG ファイルが空でない（最低 1KB）。"""
        for key in ("summary_png", "charring_png", "profiles_png", "surface_png"):
            size = Path(report_paths[key]).stat().st_size
            assert size > 1024, f"'{key}' の PNG サイズが小さすぎます ({size} bytes)"

    def test_txt_contains_beta(self, report_paths):
        """テキストサマリーに炭化速度 β の値が含まれる。"""
        txt = Path(report_paths["txt"]).read_text(encoding="utf-8")
        assert "β" in txt or "beta" in txt.lower() or "炭化速度" in txt

    def test_txt_contains_evaluation(self, report_paths):
        """テキストサマリーに性能評価の結果が含まれる。"""
        txt = Path(report_paths["txt"]).read_text(encoding="utf-8")
        assert "60min" in txt

    def test_csv_exists_and_has_data(self, report_paths):
        """CSV ファイルが存在しデータ行を持つ。"""
        csv_path = Path(report_paths["csv"])
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) > 1  # ヘッダー + 少なくとも 1 データ行
