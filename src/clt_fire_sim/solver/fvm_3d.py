"""
fvm_3d.py
=========
【役割】
3次元有限体積法（FVM）による非定常熱伝導ソルバー。

【支配方程式】
    ρ(T)·cp(T) · ∂T/∂t = ∂/∂x[k·∂T/∂x] + ∂/∂y[k·∂T/∂y] + ∂/∂z[k·∂T/∂z]

【離散化】
- 空間: 3次元 FVM、セル中心定義、調和平均熱伝導率
- 時間: 後退オイラー法（完全陰解法）→ 無条件安定
- 非線形: ピカード反復

【座標系と境界条件】
    x=0 (加熱面) ─→ x=Lx (非加熱面)
    y=0 ─→ y=Ly   （幅方向、デフォルト断熱）
    z=0 ─→ z=Lz   （長さ方向、デフォルト断熱）

【セル線形インデックス】
    idx(i,j,k) = i*(ny*nz) + j*nz + k
    i: x方向 (0..nx-1), j: y方向 (0..ny-1), k: z方向 (0..nz-1)

【1D との一致】
    ny=nz=1, dy=dz=1m のとき FVM3DSolver は FVM1DSolver と数値的に同一の
    結果を返す（境界条件の実装が同じ線形化 Robin BC のため）。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 空気の物性値（孔・スリット内の空洞セルに使用）
# ---------------------------------------------------------------------------
AIR_K: float = 0.026        # 熱伝導率 [W/m·K]（静止空気 20°C）
AIR_RHO_CP: float = 1206.0  # 体積熱容量 [J/m³·K]（ρ=1.2 kg/m³ × cp=1005 J/kg·K）


# ---------------------------------------------------------------------------
# 3D メッシュ
# ---------------------------------------------------------------------------

@dataclass
class Mesh3D:
    """3次元構造化直交メッシュ。

    Parameters
    ----------
    x_faces : np.ndarray
        x 方向の界面座標 [m]、長さ nx+1。x=0 が加熱面。
    y_faces : np.ndarray
        y 方向の界面座標 [m]、長さ ny+1。
    z_faces : np.ndarray
        z 方向の界面座標 [m]、長さ nz+1。
    """

    x_faces: np.ndarray
    y_faces: np.ndarray
    z_faces: np.ndarray

    @property
    def nx(self) -> int:
        return len(self.x_faces) - 1

    @property
    def ny(self) -> int:
        return len(self.y_faces) - 1

    @property
    def nz(self) -> int:
        return len(self.z_faces) - 1

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def x_centers(self) -> np.ndarray:
        return 0.5 * (self.x_faces[:-1] + self.x_faces[1:])

    @property
    def y_centers(self) -> np.ndarray:
        return 0.5 * (self.y_faces[:-1] + self.y_faces[1:])

    @property
    def z_centers(self) -> np.ndarray:
        return 0.5 * (self.z_faces[:-1] + self.z_faces[1:])

    @property
    def dx(self) -> np.ndarray:
        return np.diff(self.x_faces)

    @property
    def dy(self) -> np.ndarray:
        return np.diff(self.y_faces)

    @property
    def dz(self) -> np.ndarray:
        return np.diff(self.z_faces)

    def idx(self, i: int, j: int, k: int) -> int:
        """セル (i,j,k) の線形インデックス。"""
        return i * (self.ny * self.nz) + j * self.nz + k


# ---------------------------------------------------------------------------
# メッシュ生成ヘルパー
# ---------------------------------------------------------------------------

def make_clt_mesh_3d(
    layer_thicknesses: list[float],
    specimen_width: float = 1.0,
    specimen_height: float = 1.0,
    n_cells_per_layer: int = 12,
    mesh_ratio: float = 1.05,
    n_cells_y: int = 1,
    n_cells_z: int = 1,
) -> Mesh3D:
    """多層 CLT の 3 次元メッシュを生成する。

    x 方向は各層を幾何級数メッシュで生成して連結する（1D と同じ方法）。
    y, z 方向は等間隔メッシュ（デフォルト均一）。

    Parameters
    ----------
    layer_thicknesses : list[float]
        各層の厚み [m]、加熱面側から非加熱面側の順。
    specimen_width : float
        試験体の幅（y 方向）[m]。
    specimen_height : float
        試験体の高さ（z 方向）[m]。
    n_cells_per_layer : int
        x 方向の層あたりセル数。
    mesh_ratio : float
        x 方向のメッシュ幾何公比。
    n_cells_y : int
        y 方向のセル数。1 のとき 1D と等価。
    n_cells_z : int
        z 方向のセル数。1 のとき 1D と等価。

    Returns
    -------
    Mesh3D
        生成した 3 次元メッシュ。
    """
    from .fvm_1d import make_mesh_geometric, make_multi_layer_mesh

    mesh_1d = make_multi_layer_mesh(
        layer_thicknesses=layer_thicknesses,
        n_cells_per_layer=n_cells_per_layer,
        ratio=mesh_ratio,
    )
    x_faces = mesh_1d.x_faces

    y_faces = np.linspace(0.0, specimen_width, n_cells_y + 1)
    z_faces = np.linspace(0.0, specimen_height, n_cells_z + 1)

    logger.info(
        f"3Dメッシュ生成: {len(layer_thicknesses)}層, "
        f"nx={len(x_faces)-1}, ny={n_cells_y}, nz={n_cells_z}, "
        f"総セル数={(len(x_faces)-1)*n_cells_y*n_cells_z}"
    )
    return Mesh3D(x_faces=x_faces, y_faces=y_faces, z_faces=z_faces)


# ---------------------------------------------------------------------------
# 3D FVM ソルバー
# ---------------------------------------------------------------------------

class FVM3DSolver:
    """3次元有限体積法・完全陰解法ソルバー。

    FVM1DSolver と同じ BC クラス（ISO834HeatedBC, ConvRadCoolingBC）を使用。
    props オブジェクトは MultiLayerProperties（1D）を事前に setup() したものを渡す。

    Parameters
    ----------
    mesh : Mesh3D
        3次元メッシュ。
    props : Any
        物性値オブジェクト。get_k_array(T) と get_rho_cp_array(T) を実装。
        T は長さ N = nx*ny*nz の 1D 配列。
        MultiLayerProperties を使う場合は setup(x_all) を事前に呼ぶこと。
    bc_left : Any
        x=0 面の境界条件（ISO834HeatedBC または DirichletBC）。
    bc_right : Any
        x=Lx 面の境界条件（ConvRadCoolingBC, "adiabatic", または DirichletBC）。
    T_right : float
        右端 Dirichlet BC の温度 [°C]。
    T_init : float
        初期温度 [°C]（全セル均一）。
    """

    def __init__(
        self,
        mesh: Mesh3D,
        props: Any,
        bc_left: Any,
        bc_right: Any = "adiabatic",
        T_right: float = 20.0,
        T_init: float = 20.0,
        void_mask: np.ndarray | None = None,
        burn_through: bool = False,
        void_props: Any = None,
    ) -> None:
        """
        Parameters
        ----------
        void_mask : np.ndarray or None
            shape (nx, ny, nz) の bool 配列。True のセルを空洞（空気）として扱う。
            孔・スリット最適化で使用。None の場合は通常の木材解析。
        burn_through : bool
            True の場合、空洞セルを ISO 834 火炎温度に固定（燃え抜けモード）。
            False（デフォルト）は空洞を空気物性値として扱う（保守側でない）。

            【燃え抜けモードの物理的意味】
            実際の貫通孔では孔内部に火炎ガスが侵入し、
            孔の内壁を直接加熱する（ISO 834 相当の温度が孔全体に作用）。
            このモードでは void セルを内部熱源として扱うことで
            「孔を通じた次層への直接加熱」を再現する。
        void_props : object or None
            空洞セルの充填材物性値オブジェクト
            （get_k_array / get_rho_cp_array を持つもの。例: KuntanProperties）。
            指定時は空洞セルに空気の代わりに充填材の温度依存物性値を適用する。
            充填孔は火炎ガスが侵入できないため burn_through とは併用不可
            （両方指定時は void_props を優先）。
        """
        self.mesh = mesh
        self.props = props
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.T_right = T_right
        self.T = np.full(mesh.n_cells, float(T_init))
        self.t = 0.0
        self.void_props = void_props
        # 充填材があれば燃え抜けは物理的に起こらない
        self.burn_through = bool(burn_through) and void_props is None
        # void_mask を 1D フラットインデックスで保持
        if void_mask is not None:
            self._void_flat = void_mask.ravel().astype(bool)
        else:
            self._void_flat = None

    def _build_matrix(
        self,
        dt: float,
        t_new: float,
        T_iter: np.ndarray,
        T_prev: np.ndarray,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """3D FVM 係数行列 A と右辺ベクトル b を構築する。

        【エネルギー収支（セルあたり、単位 W/K と W）】
        各セルの体積熱容量 cap = ρcp·V/dt が対角に、
        各界面の熱コンダクタンス G = k_face·A_face/d_center が
        対角（加算）と非対角（減算）に入る。
        境界セルには BC の等価熱コンダクタンス h_total·A_face が加わる。

        Parameters
        ----------
        dt : float
            時間刻み幅 [s]。
        t_new : float
            次ステップの時刻 [s]（BC 評価用）。
        T_iter : np.ndarray
            ピカード反復の現在温度（物性値評価に使用）。
        T_prev : np.ndarray
            前ステップ温度（右辺の蓄熱項に使用）。

        Returns
        -------
        A : scipy.sparse.csr_matrix
        b : np.ndarray
        """
        mesh = self.mesh
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
        N = nx * ny * nz
        dx = mesh.dx   # (nx,)
        dy = mesh.dy   # (ny,)
        dz = mesh.dz   # (nz,)

        # 物性値（ピカール反復温度で評価）
        k_arr = self.props.get_k_array(T_iter)       # (N,)
        rho_cp = self.props.get_rho_cp_array(T_iter)  # (N,)

        # 孔・スリット部分の物性値を設定
        if self._void_flat is not None:
            k_arr = k_arr.copy()
            rho_cp = rho_cp.copy()
            if self.void_props is not None:
                # 充填モード: 孔・スリット = 充填材（くん炭・籾殻など）
                T_void = T_iter[self._void_flat]
                k_arr[self._void_flat] = self.void_props.get_k_array(T_void)
                rho_cp[self._void_flat] = self.void_props.get_rho_cp_array(T_void)
            elif self.burn_through:
                # 燃え抜けモード: 孔内部は高有効熱伝達率（輻射＋対流を模擬）
                # 火炎ガスが孔に侵入し、孔内壁を強制加熱する効果を
                # 孔セル内の高い有効熱伝導率で近似する
                # h_eff ≈ 対流(25) + 輻射(~100 at 500°C) ≈ 125 W/m²K
                # k_eff = h_eff × cell_size ≈ 125 × 0.005 = 0.625 W/m·K
                k_arr[self._void_flat] = 0.625
                rho_cp[self._void_flat] = AIR_RHO_CP
            else:
                # 通常モード: 孔 = 静止空気（保守側でない）
                k_arr[self._void_flat] = AIR_K
                rho_cp[self._void_flat] = AIR_RHO_CP

        k_3d = k_arr.reshape(nx, ny, nz)
        rho_cp_3d = rho_cp.reshape(nx, ny, nz)

        # セル体積 V[i,j,k] = dx[i]*dy[j]*dz[k]
        vol_3d = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]

        # 熱容量（対角の初期値）
        cap_3d = rho_cp_3d * vol_3d / dt  # [W/K]
        diag = cap_3d.ravel().copy()       # (N,)
        b = (cap_3d * T_prev.reshape(nx, ny, nz)).ravel()  # (N,) [W]

        # COO データの蓄積リスト
        row_list = []
        col_list = []
        dat_list = []

        def _harm_mean(a: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
            s = a + b_arr
            s = np.where(s < 1e-10, 1e-10, s)
            return 2.0 * a * b_arr / s

        # ---- x 方向界面 (nx-1)*ny*nz 個 ----
        if nx > 1:
            k_face_x = _harm_mean(k_3d[:-1, :, :], k_3d[1:, :, :])  # (nx-1,ny,nz)
            d_x = 0.5 * (dx[:-1, None, None] + dx[1:, None, None])    # (nx-1,1,1)
            A_x = dy[None, :, None] * dz[None, None, :]               # (1,ny,nz)
            G_x = (k_face_x * A_x / d_x).ravel()                     # ((nx-1)*ny*nz,)

            ii = np.arange(nx - 1)[:, None, None]
            jj = np.arange(ny)[None, :, None]
            kk = np.arange(nz)[None, None, :]
            n1 = (ii * ny * nz + jj * nz + kk).ravel()  # idx(i,j,k)
            n2 = n1 + ny * nz                             # idx(i+1,j,k)

            np.add.at(diag, n1, G_x)
            np.add.at(diag, n2, G_x)
            row_list += [n1, n2]
            col_list += [n2, n1]
            dat_list += [-G_x, -G_x]

        # ---- y 方向界面 nx*(ny-1)*nz 個 ----
        if ny > 1:
            k_face_y = _harm_mean(k_3d[:, :-1, :], k_3d[:, 1:, :])  # (nx,ny-1,nz)
            d_y = 0.5 * (dy[None, :-1, None] + dy[None, 1:, None])   # (1,ny-1,1)
            A_y = dx[:, None, None] * dz[None, None, :]               # (nx,1,nz)
            G_y = (k_face_y * A_y / d_y).ravel()                     # (nx*(ny-1)*nz,)

            ii = np.arange(nx)[:, None, None]
            jj = np.arange(ny - 1)[None, :, None]
            kk = np.arange(nz)[None, None, :]
            n1 = (ii * ny * nz + jj * nz + kk).ravel()
            n2 = n1 + nz   # idx(i,j+1,k)

            np.add.at(diag, n1, G_y)
            np.add.at(diag, n2, G_y)
            row_list += [n1, n2]
            col_list += [n2, n1]
            dat_list += [-G_y, -G_y]

        # ---- z 方向界面 nx*ny*(nz-1) 個 ----
        if nz > 1:
            k_face_z = _harm_mean(k_3d[:, :, :-1], k_3d[:, :, 1:])   # (nx,ny,nz-1)
            d_z = 0.5 * (dz[None, None, :-1] + dz[None, None, 1:])   # (1,1,nz-1)
            A_z = dx[:, None, None] * dy[None, :, None]               # (nx,ny,1)
            G_z = (k_face_z * A_z / d_z).ravel()                     # (nx*ny*(nz-1),)

            ii = np.arange(nx)[:, None, None]
            jj = np.arange(ny)[None, :, None]
            kk = np.arange(nz - 1)[None, None, :]
            n1 = (ii * ny * nz + jj * nz + kk).ravel()
            n2 = n1 + 1   # idx(i,j,k+1)

            np.add.at(diag, n1, G_z)
            np.add.at(diag, n2, G_z)
            row_list += [n1, n2]
            col_list += [n2, n1]
            dat_list += [-G_z, -G_z]

        # ---- 左端 BC (x=0 面) ----
        # 左面セルの線形インデックス: idx(0,j,k) = j*nz+k
        jj, kk = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
        left_idx = (jj * nz + kk).ravel()           # (ny*nz,)
        A_left = (dy[jj] * dz[kk]).ravel()           # [m²]

        T_left = T_iter[left_idx]

        if getattr(self.bc_left, "bc_type", "") == "dirichlet":
            T_bc = self.bc_left.get_temperature(t_new)
            k_left = k_arr[left_idx]
            cond_l = k_left / (0.5 * dx[0]) * A_left  # [W/K]
            np.add.at(diag, left_idx, cond_l)
            np.add.at(b, left_idx, cond_l * T_bc)

        elif hasattr(self.bc_left, "get_matrix_contrib"):
            h_tot, T_eff, q_ex = self.bc_left.get_matrix_contrib(T_left, t_new)
            # h_tot, T_eff may be arrays (ny*nz,) or scalars
            h_tot = np.asarray(h_tot) * np.ones(len(left_idx))
            T_eff = np.asarray(T_eff) * np.ones(len(left_idx))
            q_ex  = np.asarray(q_ex)  * np.ones(len(left_idx))
            cond_l = h_tot * A_left
            np.add.at(diag, left_idx, cond_l)
            np.add.at(b, left_idx, cond_l * T_eff + q_ex * A_left)

        # ---- 右端 BC (x=Lx 面) ----
        right_idx = ((nx - 1) * ny * nz + jj * nz + kk).ravel()  # (ny*nz,)
        A_right = A_left

        T_right_face = T_iter[right_idx]

        if isinstance(self.bc_right, str) and self.bc_right == "adiabatic":
            pass

        elif isinstance(self.bc_right, str) and self.bc_right == "dirichlet":
            k_right = k_arr[right_idx]
            cond_r = k_right / (0.5 * dx[-1]) * A_right
            np.add.at(diag, right_idx, cond_r)
            np.add.at(b, right_idx, cond_r * self.T_right)

        elif hasattr(self.bc_right, "get_matrix_contrib"):
            h_tot, T_ref, q_ex = self.bc_right.get_matrix_contrib(T_right_face, t_new)
            h_tot = np.asarray(h_tot) * np.ones(len(right_idx))
            T_ref = np.asarray(T_ref) * np.ones(len(right_idx))
            q_ex  = np.asarray(q_ex)  * np.ones(len(right_idx))
            cond_r = h_tot * A_right
            np.add.at(diag, right_idx, cond_r)
            np.add.at(b, right_idx, cond_r * T_ref + q_ex * A_right)

        # ---- スパース行列を組み立て ----
        all_rows = np.concatenate([np.arange(N)] + row_list)
        all_cols = np.concatenate([np.arange(N)] + col_list)
        all_data = np.concatenate([diag] + dat_list)

        A = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(N, N)).tocsr()
        return A, b

    def step(self, dt: float, n_picard: int = 2) -> None:
        """1 タイムステップ進める（ピカード反復つき）。"""
        from clt_fire_sim.boundary import iso834_temperature

        T_prev = self.T.copy()
        t_new = self.t + dt
        T_iter = T_prev.copy()

        # 燃え抜けモード: ピカール初期値の孔内部を ISO 834 温度に設定
        if self._void_flat is not None and self.burn_through:
            T_fire_now = float(iso834_temperature(t_new / 60.0))
            T_iter[self._void_flat] = T_fire_now

        for picard_idx in range(n_picard):
            A, b = self._build_matrix(dt, t_new, T_iter, T_prev)
            T_new = spla.spsolve(A, b)

            # 燃え抜けモード: 各反復後も孔内部を火炎温度で上書き
            # → 孔内壁から周囲の木材へ熱が流れ込む効果を実現
            if self._void_flat is not None and self.burn_through:
                T_new[self._void_flat] = T_fire_now

            if np.any(np.isnan(T_new)) or np.any(T_new < -300.0):
                logger.error(
                    f"3D 発散: t={self.t:.1f}s dt={dt}s picard={picard_idx} "
                    f"T_min={T_new.min():.1f} T_max={T_new.max():.1f}"
                )
                raise RuntimeError(
                    f"温度が発散しました（t={self.t:.1f}s, picard={picard_idx}）。"
                    f" T_min={T_new.min():.1f}°C"
                )

            if picard_idx > 0 and np.max(np.abs(T_new - T_iter)) < 0.1:
                break

            T_iter = T_new

        self.T = T_iter
        self.t += dt
        if hasattr(self.props, 'update_dried_state'):
            self.props.update_dried_state(self.T)

    def solve(
        self,
        t_end: float,
        dt_base: float = 5.0,
        dt_min: float = 1.0,
        dt_max: float = 10.0,
        n_picard: int = 2,
        record_interval: float | None = None,
        eval_times: list[float] | None = None,
        stop_event: threading.Event | None = None,
        progress_callback: Any | None = None,
        # ---- 放冷フェーズ設定（1D と共通インターフェース）----
        t_cooling: float = 0.0,
        bc_left_cooling: Any | None = None,
        cooling_dt_base: float = 30.0,
    ) -> dict:
        """指定時刻まで時間積分を実行する。

        Parameters
        ----------
        t_cooling : float
            加熱終了後の放冷シミュレーション時間 [s]。0 = 放冷なし。
        bc_left_cooling : Any or None
            放冷フェーズの左端（加熱面）境界条件。None の場合は加熱 BC のまま維持。
            通常は CoolingFurnaceBC インスタンスを渡す。
        cooling_dt_base : float
            放冷フェーズの基本時間刻み [s]。デフォルト 30 s。

        Returns
        -------
        dict
            {
              "times"         : np.ndarray  (n_times,)
              "temperatures"  : np.ndarray  (n_times, N)  ← 平坦化済み
              "x_centers"     : np.ndarray  (nx,)         ← x 方向セル中心
              "char_depths"   : np.ndarray  (n_times,)    ← y,z 平均炭化深さ
              "mesh"          : Mesh3D
              "t_heating_end" : float                     ← 加熱終了時刻 [s]
            }
        """
        if eval_times is None:
            eval_times = [3600.0, 4500.0, 5400.0]

        times_out: list[float] = []
        temps_out: list[np.ndarray] = []
        char_out: list[float] = []

        def _choose_dt(t_now: float) -> float:
            if t_now < 30.0:
                return dt_min
            for te in eval_times:
                if 0 < te - t_now <= 60.0:
                    return 2.0
            T_3d = self.T.reshape(self.mesh.nx, self.mesh.ny, self.mesh.nz)
            if np.any((T_3d >= 96.0) & (T_3d <= 125.0)):
                return dt_min
            return dt_base

        def _should_record(t_now: float) -> bool:
            if record_interval is not None:
                return (t_now % record_interval) < dt_min * 1.5
            return True

        def _mean_char_depth() -> float:
            T_3d = self.T.reshape(self.mesh.nx, self.mesh.ny, self.mesh.nz)
            depths = []
            for j in range(self.mesh.ny):
                for k in range(self.mesh.nz):
                    depths.append(_char_depth_1d(self.mesh.x_centers, T_3d[:, j, k]))
            return float(np.mean(depths))

        # 初期状態を記録
        times_out.append(self.t)
        temps_out.append(self.T.copy())
        char_out.append(_mean_char_depth())

        logger.info(
            f"3D解析開始: t_end={t_end:.0f}s, "
            f"nx={self.mesh.nx}, ny={self.mesh.ny}, nz={self.mesh.nz}, "
            f"N={self.mesh.n_cells}"
        )

        # ---- 加熱フェーズ ----
        while self.t < t_end - 1e-6:
            if stop_event is not None and stop_event.is_set():
                break

            dt = _choose_dt(self.t)
            dt = min(dt, dt_max, t_end - self.t)
            dt = max(dt, 1e-3)
            self.step(dt, n_picard=n_picard)

            if _should_record(self.t):
                times_out.append(self.t)
                temps_out.append(self.T.copy())
                char_out.append(_mean_char_depth())

            if progress_callback is not None:
                progress_callback(self.t, t_end, self.T)

        t_heating_end = self.t
        logger.info(
            f"3D加熱フェーズ完了: t={self.t:.1f}s, "
            f"T_max={self.T.max():.1f}°C, 平均炭化深さ={char_out[-1]*1000:.1f}mm"
        )

        # ---- 放冷フェーズ ----
        if t_cooling > 0.0:
            t_cool_end = self.t + t_cooling
            bc_left_orig = self.bc_left
            if bc_left_cooling is not None:
                self.bc_left = bc_left_cooling
            logger.info(
                f"3D放冷フェーズ開始: {t_cooling/60:.0f}分間放冷, "
                f"dt={cooling_dt_base}s"
            )
            while self.t < t_cool_end - 1e-6:
                if stop_event is not None and stop_event.is_set():
                    logger.info(f"3D放冷フェーズを中断: t={self.t:.1f}s")
                    break
                dt_c = min(cooling_dt_base, dt_max, t_cool_end - self.t)
                dt_c = max(dt_c, 1e-3)
                self.step(dt_c, n_picard=n_picard)

                if _should_record(self.t):
                    times_out.append(self.t)
                    temps_out.append(self.T.copy())
                    char_out.append(_mean_char_depth())
                    if progress_callback is not None:
                        progress_callback(self.t, t_cool_end, self.T)

            self.bc_left = bc_left_orig
            logger.info(
                f"3D放冷フェーズ完了: t={self.t:.1f}s ({self.t/60:.1f}min), "
                f"T_max={self.T.max():.1f}°C"
            )

        return {
            "times": np.array(times_out),
            "temperatures": np.array(temps_out),  # (n_times, N)
            "x_centers": self.mesh.x_centers,
            "char_depths": np.array(char_out),
            "mesh": self.mesh,
            "t_heating_end": t_heating_end,
        }


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def _char_depth_1d(x_centers: np.ndarray, T: np.ndarray, char_temp: float = 300.0) -> float:
    """1 つの x 列の炭化深さを計算する（char_depth 関数の内部版）。"""
    if T[0] < char_temp:
        return 0.0
    if T[-1] >= char_temp:
        return float(x_centers[-1])
    idx = int(np.argmax(T < char_temp)) - 1
    if idx < 0:
        return float(x_centers[0])
    x0, x1 = x_centers[idx], x_centers[idx + 1]
    T0, T1 = T[idx], T[idx + 1]
    if abs(T1 - T0) < 1e-10:
        return float(x0)
    return float(x0 + (char_temp - T0) / (T1 - T0) * (x1 - x0))


def setup_multi_layer_props_3d(
    layer_thicknesses: list[float],
    layer_props: list[Any],
    mesh: Mesh3D,
) -> Any:
    """MultiLayerProperties を 3D メッシュ用に setup して返す。

    3D メッシュの各セルに対して x 座標を展開し、
    MultiLayerProperties.setup(x_all) を呼ぶ。

    Parameters
    ----------
    layer_thicknesses : list[float]
        各層の厚み [m]（MultiLayerProperties の初期化に使用）。
    layer_props : list[Any]
        各層の物性値オブジェクト。
    mesh : Mesh3D
        対象の 3D メッシュ。

    Returns
    -------
    MultiLayerProperties
        setup() 済みの多層物性値オブジェクト。
    """
    from .fvm_1d import MultiLayerProperties

    props = MultiLayerProperties(
        layer_thicknesses=layer_thicknesses,
        layer_props=layer_props,
    )
    # 各セル (i,j,k) の x 座標: idx(i,j,k) = i*ny*nz + j*nz + k
    # → x_centers[i] を ny*nz 回繰り返したもの
    x_all = np.repeat(mesh.x_centers, mesh.ny * mesh.nz)
    props.setup(x_all)
    return props
