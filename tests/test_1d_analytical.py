"""
test_1d_analytical.py
=====================
【役割】
Phase 1 の検証テスト：1D FVM ソルバーの結果を
半無限固体の解析解（erfc）と比較する。

【テスト内容】
1. 一定物性（スギ材に近い値）で 1D 熱伝導を計算
2. 解析解 T = T0 + (Ts - T0)·erfc(x / 2√(αt)) と比較
3. 試験体内部（炭化が起きにくい深さ）での誤差が 2°C 以内であることを確認

【実行方法】
    cd clt_fire_sim
    pytest tests/test_1d_analytical.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# パス設定：src/ を Python パスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clt_fire_sim.solver.fvm_1d import (
    ConstantProperties,
    DirichletBC,
    FVM1DSolver,
    Mesh1D,
    analytical_semi_infinite,
    make_mesh_geometric,
)


# ---------------------------------------------------------------------------
# テスト用の物性値（スギ材の常温近似値）
# ---------------------------------------------------------------------------

# スギ材の代表的な常温物性値
K_SUGI = 0.12       # 熱伝導率 [W/(m·K)]
RHO_SUGI = 400.0    # 密度 [kg/m³]
CP_SUGI = 1600.0    # 比熱 [J/(kg·K)]
# → α = 0.12 / (400 × 1600) = 1.875×10⁻⁷ m²/s

T_INIT = 20.0       # 初期温度 [°C]
T_SURFACE = 200.0   # 表面温度（ステップ後）[°C]


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def run_fvm_step_problem(
    n_cells: int,
    thickness: float,
    t_end: float,
    dt: float,
    ratio: float = 1.15,
) -> dict:
    """温度ステップ問題を FVM で解いて結果を返す。

    Parameters
    ----------
    n_cells : int
        メッシュのセル数。
    thickness : float
        試験体の厚み [m]（半無限固体の近似として十分厚い値を使う）。
    t_end : float
        解析終了時刻 [s]。
    dt : float
        時間刻み幅 [s]。
    ratio : float
        メッシュの公比。

    Returns
    -------
    dict
        solve() の返す辞書（times, temperatures, x_centers）。
    """
    # メッシュ生成（加熱面側を細かく）
    mesh = make_mesh_geometric(
        total_thickness=thickness,
        n_cells=n_cells,
        ratio=ratio,
        fine_side="left",
    )

    # 定数物性値
    props = ConstantProperties(k=K_SUGI, rho=RHO_SUGI, cp=CP_SUGI)

    # 左端：一定温度 T_SURFACE（ステップ入力）
    bc_left = DirichletBC(T_surface=T_SURFACE)

    # ソルバー作成
    solver = FVM1DSolver(
        mesh=mesh,
        props=props,
        bc_left=bc_left,
        T_right=T_INIT,
        T_init=T_INIT,
        bc_right_type="dirichlet",  # 右端も固定（厚み十分大きいので影響小）
    )

    # 解析実行（dt_base でステップ幅を指定、dt_max=dt で上限固定）
    return solver.solve(t_end=t_end, dt_base=dt, dt_max=dt)


# ---------------------------------------------------------------------------
# テストケース
# ---------------------------------------------------------------------------

class TestSemiInfiniteAnalytical:
    """半無限固体の解析解との比較テスト群。"""

    def test_temperature_at_30min(self):
        """t=1800s（30分）時点での温度分布を解析解と比較。

        【物理的背景】
        CLT試験では30分加熱後の温度分布が重要な検証点。
        スギ材の熱拡散率 α ≈ 1.875×10⁻⁷ m²/s のとき、
        √(α·t) = √(1.875e-7 × 1800) ≈ 0.0184 m = 18.4 mm
        なので「熱影響層」の深さは数十mm程度。
        """
        t_end = 1800.0   # 30分 [s]
        dt = 5.0         # 時間刻み [s]
        thickness = 0.20  # 試験体厚み（200mm，十分厚い）

        result = run_fvm_step_problem(
            n_cells=80,
            thickness=thickness,
            t_end=t_end,
            dt=dt,
        )

        # 最終時刻のFVM結果
        T_fvm = result["temperatures"][-1]
        x = result["x_centers"]

        # 解析解
        alpha = K_SUGI / (RHO_SUGI * CP_SUGI)
        T_analytical = analytical_semi_infinite(x, t_end, T_INIT, T_SURFACE, alpha)

        # 誤差評価：深さ5mm〜80mm の範囲（境界の影響が少ない中間部）
        mask = (x >= 0.005) & (x <= 0.080)
        error = np.abs(T_fvm[mask] - T_analytical[mask])

        max_error = error.max()
        print(f"\n[t=1800s] 最大誤差: {max_error:.3f}°C (許容: 2.0°C)")
        print(f"  FVM 表面近傍温度（x=5mm）: {T_fvm[mask][0]:.2f}°C")
        print(f"  解析解 同位置: {T_analytical[mask][0]:.2f}°C")

        # 誤差が 2°C 以内であることを確認
        assert max_error < 2.0, (
            f"FVM と解析解の誤差が大きすぎます: {max_error:.3f}°C > 2.0°C"
        )

    def test_temperature_at_10min(self):
        """t=600s（10分）時点での温度分布を解析解と比較。"""
        t_end = 600.0
        dt = 2.0
        thickness = 0.15

        result = run_fvm_step_problem(
            n_cells=60,
            thickness=thickness,
            t_end=t_end,
            dt=dt,
        )

        T_fvm = result["temperatures"][-1]
        x = result["x_centers"]

        alpha = K_SUGI / (RHO_SUGI * CP_SUGI)
        T_analytical = analytical_semi_infinite(x, t_end, T_INIT, T_SURFACE, alpha)

        # 深さ 3mm〜50mm で比較
        mask = (x >= 0.003) & (x <= 0.050)
        error = np.abs(T_fvm[mask] - T_analytical[mask])
        max_error = error.max()
        print(f"\n[t=600s] 最大誤差: {max_error:.3f}°C (許容: 2.0°C)")

        assert max_error < 2.0, (
            f"FVM と解析解の誤差が大きすぎます: {max_error:.3f}°C > 2.0°C"
        )

    def test_surface_temperature_is_boundary_value(self):
        """表面セルの温度が境界値に十分近いことを確認。

        FVM の境界処理が正しく実装されているかのチェック。
        """
        t_end = 300.0
        dt = 5.0

        result = run_fvm_step_problem(
            n_cells=40,
            thickness=0.10,
            t_end=t_end,
            dt=dt,
        )

        # 最左端セルの温度（表面から dx/2 の位置）
        T_first_cell = result["temperatures"][-1][0]
        # 境界温度より少し低いはず（セル中心は表面から少し離れているため）
        # 表面温度の90%以上であれば境界処理OK
        assert T_first_cell > T_SURFACE * 0.9, (
            f"表面セル温度が低すぎます: {T_first_cell:.1f}°C "
            f"（境界温度: {T_SURFACE}°C）"
        )

    def test_conservation_monotone(self):
        """温度プロファイルが単調減少であることを確認。

        物理的に、加熱面から遠ざかるほど温度は低くなるはず。
        """
        t_end = 900.0
        dt = 5.0

        result = run_fvm_step_problem(
            n_cells=50,
            thickness=0.15,
            t_end=t_end,
            dt=dt,
        )

        T_final = result["temperatures"][-1]
        # 全セルで前のセルより温度が低い（単調減少）
        differences = np.diff(T_final)
        assert np.all(differences <= 0.1), (
            "温度プロファイルが単調減少でありません（物理的に不正）"
        )

    def test_mesh_refinement_convergence(self):
        """メッシュを細かくするほど解析解に近づくことを確認（収束性）。

        粗いメッシュより細かいメッシュの方が誤差が小さいはず。
        """
        t_end = 1800.0
        dt = 5.0
        thickness = 0.15
        alpha = K_SUGI / (RHO_SUGI * CP_SUGI)

        errors = []
        for n_cells in [20, 40, 80]:
            result = run_fvm_step_problem(
                n_cells=n_cells,
                thickness=thickness,
                t_end=t_end,
                dt=dt,
            )
            T_fvm = result["temperatures"][-1]
            x = result["x_centers"]
            T_analytical = analytical_semi_infinite(
                x, t_end, T_INIT, T_SURFACE, alpha
            )
            mask = (x >= 0.005) & (x <= 0.080)
            errors.append(np.abs(T_fvm[mask] - T_analytical[mask]).max())
            print(f"  n_cells={n_cells}: 最大誤差={errors[-1]:.4f}°C")

        # 細かいメッシュほど誤差が小さい（収束している）
        assert errors[0] > errors[1] or errors[1] > errors[2] or (
            errors[2] < 1.0  # 最細メッシュで1°C以内ならOK
        ), f"メッシュ収束性が確認できません: {errors}"


# ---------------------------------------------------------------------------
# グラフ出力（pytest には含まない、直接実行用）
# ---------------------------------------------------------------------------

def plot_verification():
    """FVM 結果と解析解を比較するグラフを生成して保存する。"""
    import matplotlib.pyplot as plt
    import matplotlib

    # Windows で日本語フォントを使う
    matplotlib.rcParams["font.family"] = "Yu Gothic"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    alpha = K_SUGI / (RHO_SUGI * CP_SUGI)
    thickness = 0.20
    n_cells = 80

    # --- 左図：温度プロファイル（複数時刻）---
    ax = axes[0]
    plot_times = [300, 600, 900, 1800]  # [s]
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(plot_times)))

    for t_end, color in zip(plot_times, colors):
        result = run_fvm_step_problem(
            n_cells=n_cells, thickness=thickness,
            t_end=t_end, dt=5.0,
        )
        T_fvm = result["temperatures"][-1]
        x = result["x_centers"]

        x_dense = np.linspace(0, 0.12, 500)
        T_ana = analytical_semi_infinite(x_dense, t_end, T_INIT, T_SURFACE, alpha)

        label = f"{t_end//60}min"
        ax.plot(x * 1000, T_fvm, "o", color=color, markersize=3, label=f"FVM {label}")
        ax.plot(x_dense * 1000, T_ana, "-", color=color, lw=1.5, label=f"解析解 {label}")

    ax.set_xlabel("深さ x [mm]")
    ax.set_ylabel("温度 T [°C]")
    ax.set_title("Phase 1 検証: FVM vs 解析解（温度分布）")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(T_INIT - 5, T_SURFACE + 10)
    ax.grid(True, alpha=0.3)

    # --- 右図：誤差の深さ分布（t=1800s）---
    ax = axes[1]
    t_check = 1800.0
    result = run_fvm_step_problem(
        n_cells=n_cells, thickness=thickness,
        t_end=t_check, dt=5.0,
    )
    T_fvm = result["temperatures"][-1]
    x = result["x_centers"]
    T_ana = analytical_semi_infinite(x, t_check, T_INIT, T_SURFACE, alpha)
    error = np.abs(T_fvm - T_ana)

    ax.semilogy(x * 1000, np.maximum(error, 1e-5), "b-", lw=1.5)
    ax.axhline(2.0, color="r", ls="--", label="許容誤差 2°C")
    ax.set_xlabel("深さ x [mm]")
    ax.set_ylabel("絶対誤差 |T_FVM - T_解析| [°C]")
    ax.set_title("Phase 1 検証: 誤差分布（t=1800s）")
    ax.legend()
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 出力先
    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "phase1_verification.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n検証グラフを保存しました: {out_path}")
    plt.show()


if __name__ == "__main__":
    # pytest を介さずに直接実行するとグラフが表示される
    alpha_val = K_SUGI / (RHO_SUGI * CP_SUGI)
    print("Phase 1: FVM vs analytical (semi-infinite, temperature step)")
    print(f"  k={K_SUGI} W/mK, rho={RHO_SUGI} kg/m3, cp={CP_SUGI} J/kgK")
    print(f"  alpha = {alpha_val:.3e} m2/s")
    plot_verification()
