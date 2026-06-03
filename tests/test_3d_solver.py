"""
test_3d_solver.py
=================
【役割】
Phase 4 の検証テスト：3D FVM ソルバーの動作確認。

【テスト内容】
1. ny=nz=1 のとき 3D 解が 1D 解と一致すること（基本検証）
2. 温度ステップ問題が解析解（erfc）と一致すること
3. 炭化速度が Eurocode 5 範囲内であること（1D と同等）
4. 断熱 y,z 面で温度が y,z 方向に一様であること

【実行方法】
    cd clt_fire_sim
    pytest tests/test_3d_solver.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clt_fire_sim.boundary import (
    ConvRadCoolingBC,
    ISO834HeatedBC,
    iso834_temperature,
)
from clt_fire_sim.materials import make_sugi_properties
from clt_fire_sim.solver.fvm_1d import (
    ConstantProperties,
    DirichletBC,
    FVM1DSolver,
    MultiLayerProperties,
    analytical_semi_infinite,
    make_mesh_geometric,
    make_multi_layer_mesh,
)
from clt_fire_sim.solver.fvm_3d import (
    FVM3DSolver,
    Mesh3D,
    make_clt_mesh_3d,
    setup_multi_layer_props_3d,
)


# ---------------------------------------------------------------------------
# テスト 1: ny=nz=1 で 3D が 1D と一致
# ---------------------------------------------------------------------------

class TestMatch1D:
    """ny=nz=1 のとき 3D ソルバーが 1D ソルバーと数値的に一致する。"""

    def _run_both(self, t_end: float = 600.0, dt: float = 5.0):
        """1D と 3D（ny=nz=1）を同じ条件で走らせ、両方の結果を返す。"""
        # スギ材 5層 CLT 150mm
        layer_thicknesses = [0.030] * 5
        n_per = 12
        ratio = 1.05

        props_list = [make_sugi_properties(rho_0=400.0) for _ in range(5)]
        bc_left = ISO834HeatedBC()
        bc_right = ConvRadCoolingBC()
        T_init = 20.0

        # ---- 1D ----
        mesh_1d = make_multi_layer_mesh(layer_thicknesses, n_per, ratio)
        props_1d = MultiLayerProperties(layer_thicknesses, props_list)
        props_1d.setup(mesh_1d.x_centers)
        solver_1d = FVM1DSolver(mesh_1d, props_1d, bc_left, bc_right, T_init=T_init)
        result_1d = solver_1d.solve(t_end, dt_base=dt, dt_max=dt, n_picard=3,
                                    record_interval=dt)

        # ---- 3D (ny=nz=1, specimen面積=1m×1m) ----
        mesh_3d = make_clt_mesh_3d(
            layer_thicknesses, 1.0, 1.0, n_per, ratio, n_cells_y=1, n_cells_z=1
        )
        props_3d = setup_multi_layer_props_3d(layer_thicknesses, props_list, mesh_3d)
        bc_left_3d = ISO834HeatedBC()
        bc_right_3d = ConvRadCoolingBC()
        solver_3d = FVM3DSolver(mesh_3d, props_3d, bc_left_3d, bc_right_3d, T_init=T_init)
        result_3d = solver_3d.solve(t_end, dt_base=dt, dt_max=dt, n_picard=3,
                                    record_interval=dt)

        return result_1d, result_3d

    def test_final_temperature_matches(self):
        """最終時刻の温度プロファイルが 1D と一致する（許容誤差 0.01°C）。"""
        r1d, r3d = self._run_both(t_end=600.0, dt=5.0)

        T_1d = r1d["temperatures"][-1]          # (nx,)
        T_3d = r3d["temperatures"][-1].ravel()  # (nx*1*1,) = (nx,)

        max_err = np.max(np.abs(T_1d - T_3d))
        assert max_err < 0.01, (
            f"3D と 1D の温度差が大きすぎます: 最大誤差 = {max_err:.4f}°C"
        )

    def test_char_depth_matches(self):
        """炭化深さが 1D と一致する（許容誤差 0.1mm）。"""
        r1d, r3d = self._run_both(t_end=1800.0, dt=5.0)
        cd_1d = r1d["char_depths"][-1] * 1000   # [mm]
        cd_3d = r3d["char_depths"][-1] * 1000   # [mm]
        assert abs(cd_1d - cd_3d) < 0.1, (
            f"炭化深さの差: 1D={cd_1d:.2f}mm, 3D={cd_3d:.2f}mm"
        )


# ---------------------------------------------------------------------------
# テスト 2: 温度ステップ問題と解析解の比較
# ---------------------------------------------------------------------------

class TestStepProblem3D:
    """Dirichlet BC 温度ステップ問題での erfc 解析解との比較。"""

    def test_temperature_profile_vs_erfc(self):
        """t=600s の温度分布が半無限固体の解析解と 2°C 以内に一致する。"""
        # スギ材定数物性値
        k_val, rho_val, cp_val = 0.12, 400.0, 1600.0
        alpha = k_val / (rho_val * cp_val)
        T_init = 20.0
        T_surface = 200.0
        t_end = 600.0
        thickness = 0.15

        # 3D メッシュ（ny=nz=1 → 1D 等価）
        mesh_1d = make_mesh_geometric(thickness, 60, ratio=1.05, fine_side="left")
        mesh_3d = Mesh3D(
            x_faces=mesh_1d.x_faces,
            y_faces=np.array([0.0, 1.0]),
            z_faces=np.array([0.0, 1.0]),
        )

        props_1d = ConstantProperties(k=k_val, rho=rho_val, cp=cp_val)

        # MultiLayerProperties として wrap
        props_3d = setup_multi_layer_props_3d(
            layer_thicknesses=[thickness],
            layer_props=[props_1d],
            mesh=mesh_3d,
        )

        bc_left = DirichletBC(T_surface=T_surface)
        solver = FVM3DSolver(
            mesh_3d, props_3d, bc_left, bc_right="dirichlet",
            T_right=T_init, T_init=T_init
        )
        result = solver.solve(t_end, dt_base=2.0, dt_max=2.0, n_picard=1,
                              record_interval=t_end)

        T_3d_flat = result["temperatures"][-1]  # (nx,)
        x = mesh_3d.x_centers

        T_ana = analytical_semi_infinite(x, t_end, T_init, T_surface, alpha)
        mask = (x >= 0.005) & (x <= 0.080)
        max_err = np.max(np.abs(T_3d_flat[mask] - T_ana[mask]))

        assert max_err < 2.0, (
            f"解析解との誤差が大きすぎます: {max_err:.3f}°C (許容 2°C)"
        )


# ---------------------------------------------------------------------------
# テスト 3: y,z 方向の断熱性（均一加熱では温度が y,z に依らない）
# ---------------------------------------------------------------------------

class TestYZUniformity:
    """均一加熱・断熱 y,z 面のとき温度は y,z 方向に一様であること。"""

    def test_temperature_uniform_in_yz(self):
        """ny=3, nz=3 で均一加熱したとき、同じ x 位置の全セルが同温になる。"""
        layer_thicknesses = [0.030] * 3
        n_per = 8
        props_list = [make_sugi_properties() for _ in range(3)]

        mesh_3d = make_clt_mesh_3d(
            layer_thicknesses, 0.3, 0.3,
            n_cells_per_layer=n_per, mesh_ratio=1.05,
            n_cells_y=3, n_cells_z=3,
        )
        props_3d = setup_multi_layer_props_3d(layer_thicknesses, props_list, mesh_3d)

        solver = FVM3DSolver(
            mesh_3d, props_3d,
            bc_left=ISO834HeatedBC(),
            bc_right=ConvRadCoolingBC(),
            T_init=20.0,
        )
        result = solver.solve(600.0, dt_base=5.0, dt_max=5.0, n_picard=2,
                              record_interval=600.0)

        T_final = result["temperatures"][-1].reshape(
            mesh_3d.nx, mesh_3d.ny, mesh_3d.nz
        )

        # 各 x 位置で y,z 方向の温度のばらつきが 1e-6 °C 以下
        for i in range(mesh_3d.nx):
            T_slice = T_final[i, :, :]
            spread = T_slice.max() - T_slice.min()
            assert spread < 1e-6, (
                f"x={mesh_3d.x_centers[i]*1000:.1f}mm で y,z 方向の温度差 = {spread:.2e}°C"
            )


# ---------------------------------------------------------------------------
# テスト 4: メッシュ・ソルバー基本動作
# ---------------------------------------------------------------------------

class TestMesh3D:
    """Mesh3D と make_clt_mesh_3d の基本テスト。"""

    def test_total_thickness(self):
        """x 方向の総厚みが層厚みの合計と一致する。"""
        thicknesses = [0.030] * 5
        mesh = make_clt_mesh_3d(thicknesses, 0.2, 0.2, n_cells_y=4, n_cells_z=4)
        assert abs(mesh.x_faces[-1] - sum(thicknesses)) < 1e-9

    def test_n_cells(self):
        """総セル数 = nx × ny × nz。"""
        thicknesses = [0.030] * 3
        n_per = 10
        ny, nz = 4, 5
        mesh = make_clt_mesh_3d(thicknesses, 0.2, 0.2, n_cells_per_layer=n_per,
                                n_cells_y=ny, n_cells_z=nz)
        assert mesh.n_cells == 3 * n_per * ny * nz

    def test_idx_roundtrip(self):
        """idx(i,j,k) が正しい線形インデックスを返す。"""
        mesh = Mesh3D(
            x_faces=np.linspace(0, 0.09, 4),
            y_faces=np.linspace(0, 0.2, 3),
            z_faces=np.linspace(0, 0.2, 3),
        )
        assert mesh.idx(0, 0, 0) == 0
        assert mesh.idx(0, 0, 1) == 1
        assert mesh.idx(0, 1, 0) == mesh.nz
        assert mesh.idx(1, 0, 0) == mesh.ny * mesh.nz


# ---------------------------------------------------------------------------
# テスト 5: 炭化速度（3D で 1D と同等の結果）
# ---------------------------------------------------------------------------

class TestCharringRate3D:
    """3D ソルバーの炭化速度が Eurocode 5 範囲内かつ 1D と一致。"""

    @pytest.fixture(scope="class")
    def result_3d(self):
        """5層スギ CLT 90分シミュレーション（ny=nz=1）。"""
        layer_thicknesses = [0.030] * 5
        props_list = [make_sugi_properties(rho_0=400.0) for _ in range(5)]
        mesh_3d = make_clt_mesh_3d(
            layer_thicknesses, 1.0, 1.0,
            n_cells_per_layer=12, mesh_ratio=1.05,
            n_cells_y=1, n_cells_z=1,
        )
        props_3d = setup_multi_layer_props_3d(layer_thicknesses, props_list, mesh_3d)
        solver = FVM3DSolver(
            mesh_3d, props_3d,
            bc_left=ISO834HeatedBC(),
            bc_right=ConvRadCoolingBC(),
            T_init=20.0,
        )
        return solver.solve(5400.0, dt_base=5.0, dt_min=1.0, dt_max=10.0,
                            n_picard=3, record_interval=30.0)

    def test_charring_rate_eurocode(self, result_3d):
        """炭化速度が Eurocode 5 の ±15% 範囲（0.55〜0.75 mm/min）内。"""
        times_min = result_3d["times"] / 60.0
        char_mm = result_3d["char_depths"] * 1000.0
        mask = (times_min >= 10) & (times_min <= 60) & (char_mm > 1.0)
        assert mask.sum() >= 5
        coeffs = np.polyfit(times_min[mask], char_mm[mask], 1)
        beta = coeffs[0]
        assert 0.55 <= beta <= 0.75, (
            f"3D炭化速度 {beta:.3f} mm/min が Eurocode 5 範囲外"
        )
