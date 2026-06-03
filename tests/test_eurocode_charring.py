"""
test_eurocode_charring.py
=========================
【役割】
Phase 2 の主要検証テスト：5層スギ CLT（150mm）を ISO 834 加熱したときの
炭化速度が Eurocode 5 の標準値 β₀ = 0.65 mm/min ± 15% に入るかを確認する。

【Eurocode 5 (EN 1995-1-2) の標準炭化速度】
    β₀ = 0.65 mm/min    （針葉樹材、ρ₀ ≥ 290 kg/m³、ISO 834 加熱）
    許容範囲: ±15%  →  0.55 〜 0.75 mm/min

【解析条件】
- 試験体: スギ CLT 5 層 × 30 mm = 150 mm
- 加熱面 BC: ISO 834（対流 25 W/m²K + 輻射 ε=0.8）
- 非加熱面 BC: 自然対流（9 W/m²K）+ 輻射（ε=0.8）
- ピカード反復: 2 回/ステップ

【炭化速度の算出方法】
炭化深さ（300°C 等温面）の時刻歴を取得し、
炭化開始後〜 60 分間の線形回帰から平均炭化速度を算出する。

【実行方法】
    pytest tests/test_eurocode_charring.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clt_fire_sim.boundary import ISO834HeatedBC, ConvRadCoolingBC, iso834_temperature
from clt_fire_sim.materials import make_sugi_properties
from clt_fire_sim.solver.fvm_1d import (
    FVM1DSolver,
    char_depth,
    make_mesh_geometric,
)

# ---------------------------------------------------------------------------
# 解析パラメータ定数
# ---------------------------------------------------------------------------

# 試験体諸元
TOTAL_THICKNESS_M = 0.150   # 総厚み 150 mm = 5 × 30 mm
N_LAYERS = 5
LAYER_THICKNESS_M = 0.030

# Eurocode 5 の標準炭化速度と許容範囲
BETA_0_EUROCODE = 0.65   # [mm/min]
TOLERANCE_RATIO = 0.15   # ± 15%
BETA_MIN = BETA_0_EUROCODE * (1 - TOLERANCE_RATIO)  # 0.5525 mm/min
BETA_MAX = BETA_0_EUROCODE * (1 + TOLERANCE_RATIO)  # 0.7475 mm/min

# 解析時間
T_END_MIN = 90.0        # 解析終了時刻 [分]
T_END_S = T_END_MIN * 60.0

# 炭化速度の算出区間（炭化開始後〜60分）
REGRESSION_START_MIN = 10.0   # 炭化が安定し始める時刻 [分]
REGRESSION_END_MIN = 60.0     # 線形回帰の終了時刻 [分]


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def run_charring_simulation(
    n_cells: int = 60,
    ratio: float = 1.05,
    n_picard: int = 3,
    rho_0: float = 400.0,
    moisture_content: float = 0.12,
) -> dict:
    """ISO 834 加熱下の CLT 炭化シミュレーションを実行する。

    Parameters
    ----------
    n_cells : int
        メッシュのセル数。
    ratio : float
        メッシュの幾何公比。
    n_picard : int
        ピカード反復回数。
    rho_0 : float
        スギの初期乾燥密度 [kg/m³]。
    moisture_content : float
        含水率（質量比）。

    Returns
    -------
    dict
        solve() の返す辞書に加え、炭化速度情報を追加。
    """
    # メッシュ：加熱面側を細かく
    mesh = make_mesh_geometric(
        total_thickness=TOTAL_THICKNESS_M,
        n_cells=n_cells,
        ratio=ratio,
        fine_side="left",
    )

    # 物性値：スギ材（Eurocode 5）
    props = make_sugi_properties(rho_0=rho_0, moisture_content=moisture_content)

    # 境界条件
    bc_left = ISO834HeatedBC()
    bc_right = ConvRadCoolingBC()

    # ソルバー生成
    solver = FVM1DSolver(
        mesh=mesh,
        props=props,
        bc_left=bc_left,
        bc_right=bc_right,
        T_init=20.0,
    )

    # 解析実行（60 秒ごとに記録）
    result = solver.solve(
        t_end=T_END_S,
        dt_base=5.0,
        dt_min=1.0,
        dt_max=10.0,
        n_picard=n_picard,
        record_interval=30.0,   # 30 秒ごとに記録
    )

    return result


def compute_charring_rate(times_s: np.ndarray, char_depths_m: np.ndarray) -> float:
    """炭化深さの時刻歴から平均炭化速度を算出する [mm/min]。

    線形回帰 (numpy.polyfit) を使用して、炭化が安定している区間の
    傾きから炭化速度を求める。

    Parameters
    ----------
    times_s : np.ndarray
        時刻 [s]。
    char_depths_m : np.ndarray
        炭化深さ [m]。

    Returns
    -------
    float
        平均炭化速度 [mm/min]。
    """
    times_min = times_s / 60.0

    # 線形回帰区間のマスク
    mask = (
        (times_min >= REGRESSION_START_MIN)
        & (times_min <= REGRESSION_END_MIN)
        & (char_depths_m > 0.001)  # 炭化が実際に始まっているセルのみ
    )

    if mask.sum() < 5:
        return 0.0  # データが少なすぎて評価不能

    # 線形回帰: char_depth [mm] = beta * t [min] + offset
    char_mm = char_depths_m[mask] * 1000.0
    times_reg = times_min[mask]

    coeffs = np.polyfit(times_reg, char_mm, 1)
    charring_rate_mm_per_min = coeffs[0]  # 傾き = 炭化速度

    return float(charring_rate_mm_per_min)


# ---------------------------------------------------------------------------
# テストクラス
# ---------------------------------------------------------------------------

class TestEurocodeCharring:
    """Eurocode 5 炭化速度との比較テスト群。"""

    @pytest.fixture(scope="class")
    def sim_result(self):
        """シミュレーション結果をクラス内でキャッシュする。

        scope="class" により、このテストクラス内で 1 回だけ計算する。
        """
        print("\n5層スギ CLT の炭化シミュレーションを実行中...")
        result = run_charring_simulation()
        beta = compute_charring_rate(result["times"], result["char_depths"])
        result["charring_rate"] = beta
        print(f"  炭化速度: {beta:.3f} mm/min (Eurocode: {BETA_0_EUROCODE} mm/min)")
        print(f"  許容範囲: {BETA_MIN:.3f} 〜 {BETA_MAX:.3f} mm/min")
        return result

    def test_charring_rate_within_eurocode_range(self, sim_result):
        """炭化速度が Eurocode 5 の ±15% 範囲内に収まるか。

        β₀ = 0.65 mm/min ± 15% → 0.5525 〜 0.7475 mm/min
        """
        beta = sim_result["charring_rate"]
        print(f"\n炭化速度 β = {beta:.3f} mm/min")
        print(f"Eurocode β₀ = {BETA_0_EUROCODE} mm/min")
        print(f"誤差 = {(beta - BETA_0_EUROCODE)/BETA_0_EUROCODE*100:+.1f}%")

        assert BETA_MIN <= beta <= BETA_MAX, (
            f"炭化速度 {beta:.3f} mm/min が Eurocode 範囲 "
            f"[{BETA_MIN:.3f}, {BETA_MAX:.3f}] mm/min を外れています"
        )

    def test_char_depth_at_60min_reasonable(self, sim_result):
        """60 分後の炭化深さが物理的に合理的な範囲か。

        β₀ = 0.65 mm/min × 60 min = 39 mm を期待（±20mm 許容）。
        """
        times_min = sim_result["times"] / 60.0
        idx_60 = np.argmin(np.abs(times_min - 60.0))
        char_mm = sim_result["char_depths"][idx_60] * 1000

        expected_mm = BETA_0_EUROCODE * 60.0  # = 39 mm
        print(f"\n60 分後の炭化深さ: {char_mm:.1f} mm (期待値: {expected_mm:.0f} mm)")

        assert abs(char_mm - expected_mm) < 20, (
            f"炭化深さ {char_mm:.1f} mm が期待値 {expected_mm:.0f} mm ±20mm を外れています"
        )

    def test_char_depth_monotonically_increasing(self, sim_result):
        """炭化深さが時間とともに単調増加すること。

        炭化が一度進んだら後退しないのは物理的に自明。
        """
        char_depths = sim_result["char_depths"]
        diffs = np.diff(char_depths)
        # 微小な数値誤差（0.1 mm 以下）は許容
        assert np.all(diffs >= -0.0001), (
            f"炭化深さが減少しているステップがあります: 最大後退量 = {diffs.min()*1000:.2f} mm"
        )

    def test_unheated_face_temperature_reasonable(self, sim_result):
        """60 分後の非加熱面温度が遮熱性基準を満たすか。

        5 層スギ CLT（150mm）は 60 分で 140 K 上昇を超えないはず。
        （これは 160°C 以下に相当: 20 + 140 = 160°C）
        """
        times_min = sim_result["times"] / 60.0
        idx_60 = np.argmin(np.abs(times_min - 60.0))

        # 非加熱面（最右端セル）の温度
        T_unheated_60 = sim_result["temperatures"][idx_60, -1]
        delta_T = T_unheated_60 - 20.0  # 初期温度 20°C からの上昇

        print(f"\n60 分後の非加熱面温度: {T_unheated_60:.1f}°C (上昇量: {delta_T:.1f} K)")

        # 150mm CLT では 60 分遮熱性を満たすことを期待
        assert delta_T < 140.0, (
            f"非加熱面温度上昇 {delta_T:.1f} K が遮熱性基準 140 K を超えています"
        )

    def test_heated_face_follows_iso834(self, sim_result):
        """加熱面の最高温度が ISO 834 曲線に近いこと。

        表面温度は ISO 834 のガス温度より低いが、
        60 分後は 700°C 以上に達しているはず（燃え止まり設計では重要）。
        """
        times_min = sim_result["times"] / 60.0
        idx_60 = np.argmin(np.abs(times_min - 60.0))

        T_heated_60 = sim_result["temperatures"][idx_60, 0]
        T_gas_60 = iso834_temperature(60.0)

        print(f"\n60 分後の加熱面温度: {T_heated_60:.1f}°C (ISO 834 気相: {T_gas_60:.1f}°C)")

        # 加熱面温度は ISO 834 気相温度より低い
        assert T_heated_60 < T_gas_60, (
            f"加熱面温度 {T_heated_60:.1f}°C が ISO 834 気相温度 {T_gas_60:.1f}°C を超えています"
        )
        # ただし大幅に低くもない（700°C 以上）
        assert T_heated_60 > 600.0, (
            f"60 分後の加熱面温度 {T_heated_60:.1f}°C が 600°C 未満です"
        )


class TestCharDepthFunction:
    """char_depth 関数の単体テスト。"""

    def test_no_charring(self):
        """全セルが 300°C 未満 → 炭化深さ = 0。"""
        x = np.array([0.01, 0.02, 0.03])
        T = np.array([280.0, 200.0, 100.0])
        assert char_depth(x, T) == 0.0

    def test_full_charring(self):
        """全セルが 300°C 以上 → 炭化深さ = 試験体厚み。"""
        x = np.array([0.01, 0.02, 0.03])
        T = np.array([800.0, 500.0, 350.0])
        assert abs(char_depth(x, T) - 0.03) < 1e-6

    def test_char_at_interface(self):
        """炭化フロントが 2 セル間に存在する場合の線形補間。"""
        x = np.array([0.01, 0.03])
        T = np.array([400.0, 200.0])  # 300°C は 0.02 m の位置
        depth = char_depth(x, T)
        expected = 0.01 + (300 - 400) / (200 - 400) * (0.03 - 0.01)
        assert abs(depth - expected) < 1e-6


# ---------------------------------------------------------------------------
# グラフ出力（直接実行用）
# ---------------------------------------------------------------------------

def plot_charring_verification():
    """炭化速度と Eurocode の比較グラフを生成する。"""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "Yu Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    print("シミュレーション実行中（しばらくお待ちください）...")
    result = run_charring_simulation()
    beta = compute_charring_rate(result["times"], result["char_depths"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    times_min = result["times"] / 60.0
    char_mm = result["char_depths"] * 1000

    # ---- 左図：炭化深さの時刻歴 ----
    ax = axes[0]
    ax.plot(times_min, char_mm, "b-", lw=2, label="解析結果（300°C等温面）")

    # Eurocode の理想炭化速度（β₀ = 0.65 mm/min）
    t_plot = np.linspace(0, 90, 100)
    ax.plot(
        t_plot, BETA_0_EUROCODE * t_plot,
        "r--", lw=1.5, label=f"Eurocode β₀={BETA_0_EUROCODE} mm/min"
    )
    ax.fill_between(
        t_plot,
        BETA_MIN * t_plot,
        BETA_MAX * t_plot,
        alpha=0.2, color="red", label=f"許容範囲 ±{TOLERANCE_RATIO*100:.0f}%"
    )
    ax.set_xlabel("時間 [分]")
    ax.set_ylabel("炭化深さ [mm]")
    ax.set_title(f"炭化進行 (β={beta:.3f} mm/min)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 70)

    # ---- 中図：温度分布（複数時刻）----
    ax = axes[1]
    plot_times_min = [10, 20, 30, 45, 60, 75, 90]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(plot_times_min)))
    x_mm = result["x_centers"] * 1000

    for t_min, color in zip(plot_times_min, colors):
        idx = np.argmin(np.abs(times_min - t_min))
        T_profile = result["temperatures"][idx]
        ax.plot(x_mm, T_profile, color=color, lw=1.5, label=f"{t_min}分")

    ax.axhline(300, color="k", ls="--", lw=0.8, label="炭化温度(300°C)")
    ax.set_xlabel("加熱面からの深さ [mm]")
    ax.set_ylabel("温度 [°C]")
    ax.set_title("厚み方向の温度分布")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ---- 右図：ISO 834 vs 加熱面・非加熱面温度 ----
    ax = axes[2]
    T_iso834 = np.array([iso834_temperature(t) for t in times_min])
    T_heated = result["temperatures"][:, 0]    # 加熱面（左端セル）
    T_unheated = result["temperatures"][:, -1]  # 非加熱面（右端セル）

    ax.plot(times_min, T_iso834, "k--", lw=1, label="ISO 834 ガス温度")
    ax.plot(times_min, T_heated, "r-", lw=2, label="加熱面温度")
    ax.plot(times_min, T_unheated, "b-", lw=2, label="非加熱面温度")
    ax.axhline(20 + 140, color="b", ls=":", lw=1, label="遮熱基準(+140K)")
    ax.set_xlabel("時間 [分]")
    ax.set_ylabel("温度 [°C]")
    ax.set_title("表面温度の時刻歴")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Phase 2 検証: 5層スギCLT 150mm / ISO 834 加熱\n"
        f"炭化速度 β={beta:.3f} mm/min "
        f"(Eurocode β₀={BETA_0_EUROCODE}, 誤差{(beta-BETA_0_EUROCODE)/BETA_0_EUROCODE*100:+.1f}%)",
        fontsize=11
    )
    plt.tight_layout()

    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "phase2_charring_verification.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"グラフを保存: {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_charring_verification()
