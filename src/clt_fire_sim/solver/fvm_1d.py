"""
fvm_1d.py
=========
【役割】
1次元有限体積法（FVM）による非定常熱伝導ソルバー。

Phase 1：定数物性値・ディリクレ境界条件で基礎動作を確認
Phase 2：温度依存物性値（Eurocode 5）+ ISO 834 ロバン境界条件 + ピカード反復

【支配方程式】
    ρ(T)·cp(T) · ∂T/∂t = ∂/∂x [k(T) · ∂T/∂x]   (1次元版)

【離散化方法】
- 空間：有限体積法（セル中心定義、調和平均熱伝導率）
- 時間：後退オイラー法（陰解法）→ 無条件安定
- 境界：ディリクレ（第 1 種）またはロバン（第 3 種：対流＋輻射）
- 非線形：ピカード反復（物性値を前ステップ温度で評価して反復更新）

【座標系】
    x=0（加熱面） ──→ x=L（非加熱面）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# メッシュ定義
# ---------------------------------------------------------------------------

@dataclass
class Mesh1D:
    """1次元非等間隔メッシュの情報をまとめたデータクラス。

    Parameters
    ----------
    x_faces : np.ndarray
        セル界面の座標 [m]。長さ n_cells+1。
        x_faces[0]=0（加熱面）、x_faces[-1]=L（非加熱面）。
    """

    x_faces: np.ndarray

    @property
    def n_cells(self) -> int:
        """セル数。"""
        return len(self.x_faces) - 1

    @property
    def x_centers(self) -> np.ndarray:
        """各セル中心の座標 [m]。温度はここに定義される。"""
        return 0.5 * (self.x_faces[:-1] + self.x_faces[1:])

    @property
    def dx(self) -> np.ndarray:
        """各セルの幅 Δx_i [m]。"""
        return np.diff(self.x_faces)

    @property
    def d_center(self) -> np.ndarray:
        """隣接セル中心間の距離 δ_{i,i+1} [m]。長さ n_cells-1。"""
        return np.diff(self.x_centers)


def make_mesh_geometric(
    total_thickness: float,
    n_cells: int,
    ratio: float = 1.15,
    fine_side: str = "left",
) -> Mesh1D:
    """幾何級数的に粗くなる非等間隔メッシュを生成する。

    加熱面側（left）を細かくすることで炭化境界付近の精度を確保する。

    Parameters
    ----------
    total_thickness : float
        試験体の総厚み [m]。
    n_cells : int
        セル総数。
    ratio : float
        公比（隣接セルの幅の比）。1.1〜1.2 が推奨。
    fine_side : str
        細かくする側。"left"（加熱面側）または "right"（非加熱面側）。

    Returns
    -------
    Mesh1D
        生成したメッシュオブジェクト。

    Notes
    -----
    等比数列の和 = L から最小セル幅 dx_min を逆算:
        dx_min * (r^n - 1) / (r - 1) = L
    """
    if abs(ratio - 1.0) < 1e-10:
        dx_array = np.full(n_cells, total_thickness / n_cells)
    else:
        dx_min = total_thickness * (ratio - 1.0) / (ratio**n_cells - 1.0)
        dx_array = dx_min * ratio ** np.arange(n_cells)

    if fine_side == "right":
        dx_array = dx_array[::-1]

    x_faces = np.concatenate([[0.0], np.cumsum(dx_array)])
    x_faces = x_faces * (total_thickness / x_faces[-1])

    logger.info(
        f"メッシュ生成: n_cells={n_cells}, "
        f"最小セル幅={dx_array.min()*1000:.2f}mm, "
        f"最大セル幅={dx_array.max()*1000:.2f}mm"
    )
    return Mesh1D(x_faces=x_faces)


# ---------------------------------------------------------------------------
# 多層メッシュ生成（Phase 3）
# ---------------------------------------------------------------------------

def make_multi_layer_mesh(
    layer_thicknesses: list[float],
    n_cells_per_layer: int | list[int] = 12,
    ratio: float | list[float] = 1.05,
) -> Mesh1D:
    """多層 CLT の非等間隔メッシュを生成する。

    各層を独立した幾何級数メッシュとして生成し、連結する。
    層境界（各層の接合面）は必ずメッシュの界面座標に含まれるため、
    MultiLayerProperties との座標マッピングが正確に行える。

    【各層のメッシュ方向】
    - 第 1 層（加熱面側）：left 細かい（加熱面に細かいセルを配置）
    - 最終層（非加熱面側）：right 細かい（非加熱面に細かいセルを配置）
    - 中間層：left 細かい（炭化が内部に進行する方向に細かく）

    Parameters
    ----------
    layer_thicknesses : list[float]
        各層の厚み [m]。加熱面側（左）から非加熱面側（右）の順。
    n_cells_per_layer : int or list[int]
        各層のセル数。int の場合は全層に同じ値を使用。
    ratio : float or list[float]
        各層のメッシュ幾何公比。float の場合は全層に同じ値。

    Returns
    -------
    Mesh1D
        連結された多層メッシュ。総セル数 = sum(n_cells_per_layer)。

    Examples
    --------
    >>> # 5 層 × 30mm スギ CLT（合計 150mm、60 セル）
    >>> mesh = make_multi_layer_mesh(
    ...     layer_thicknesses=[0.030] * 5,
    ...     n_cells_per_layer=12,
    ...     ratio=1.05,
    ... )
    >>> mesh.n_cells  # 60
    >>> mesh.x_faces[-1]  # 0.150 m
    """
    n_layers = len(layer_thicknesses)

    if isinstance(n_cells_per_layer, int):
        n_cells_list = [n_cells_per_layer] * n_layers
    else:
        n_cells_list = list(n_cells_per_layer)

    if isinstance(ratio, (int, float)):
        ratio_list = [float(ratio)] * n_layers
    else:
        ratio_list = [float(r) for r in ratio]

    if len(n_cells_list) != n_layers or len(ratio_list) != n_layers:
        raise ValueError(
            "layer_thicknesses, n_cells_per_layer, ratio の長さが一致しません。"
        )

    all_face_coords: list[float] = [0.0]
    x_offset = 0.0

    for i, (thickness, n_cells, r) in enumerate(
        zip(layer_thicknesses, n_cells_list, ratio_list)
    ):
        # 最終層だけ非加熱面側（right）を細かく、それ以外は加熱面側（left）
        fine_side = "right" if i == n_layers - 1 else "left"

        layer_mesh = make_mesh_geometric(
            total_thickness=thickness,
            n_cells=n_cells,
            ratio=r,
            fine_side=fine_side,
        )
        # 各層の界面座標（x=0 以外）をオフセットして追加
        shifted_faces = layer_mesh.x_faces[1:] + x_offset
        all_face_coords.extend(shifted_faces.tolist())
        x_offset += thickness

    x_faces = np.array(all_face_coords)
    logger.info(
        f"多層メッシュ生成: {n_layers}層, 総セル数={len(x_faces)-1}, "
        f"最小セル幅={np.diff(x_faces).min()*1000:.3f}mm, "
        f"総厚={x_faces[-1]*1000:.1f}mm"
    )
    return Mesh1D(x_faces=x_faces)


# ---------------------------------------------------------------------------
# 多層物性値クラス（Phase 3）
# ---------------------------------------------------------------------------

class MultiLayerProperties:
    """多層 CLT の温度依存物性値。

    各層に異なる材料（WoodProperties など）を割り当て、
    セルの座標に基づいて正しい層の物性値を返す。

    FVM1DSolver の props 引数として ConstantProperties や
    WoodProperties と透過的に交換できる（ダック・タイピング）。
    ただし使用前に必ず setup(x_centers) を呼んでセル-層マッピングを確立すること。

    Parameters
    ----------
    layer_thicknesses : list[float]
        各層の厚み [m]。加熱面側（左）から非加熱面側（右）の順。
    layer_props : list[Any]
        各層の物性値オブジェクト。layer_thicknesses と同じ長さ。
        各オブジェクトは get_k_array(T) と get_rho_cp_array(T) を実装すること。

    Examples
    --------
    >>> props_list = [WoodProperties(rho_0=400.0)] * 5
    >>> multi_props = MultiLayerProperties(
    ...     layer_thicknesses=[0.030] * 5,
    ...     layer_props=props_list,
    ... )
    >>> multi_props.setup(mesh.x_centers)  # ← 使用前に必須
    """

    def __init__(
        self,
        layer_thicknesses: list[float],
        layer_props: list[Any],
        contact_resistances: list[float] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        contact_resistances : list[float] or None
            各層の非加熱面側界面の接触熱抵抗 [m²K/W]。
            長さは layer_thicknesses と同じ。None の場合はすべてゼロ。
            最終層（layer_thicknesses[-1]）の値は使用されない（非加熱面 BC が担当）。
        """
        if len(layer_thicknesses) != len(layer_props):
            raise ValueError(
                f"layer_thicknesses の長さ ({len(layer_thicknesses)}) と "
                f"layer_props の長さ ({len(layer_props)}) が一致しません。"
            )
        self.layer_thicknesses = list(layer_thicknesses)
        self.layer_props = list(layer_props)

        # 接触熱抵抗（各層の右端界面）
        n = len(layer_thicknesses)
        if contact_resistances is not None:
            if len(contact_resistances) != n:
                raise ValueError(
                    f"contact_resistances の長さ ({len(contact_resistances)}) が "
                    f"layer_thicknesses の長さ ({n}) と一致しません。"
                )
            self.contact_resistances = list(contact_resistances)
        else:
            self.contact_resistances = [0.0] * n

        # 各層の右端座標（cumsum で計算）
        # _boundaries[0]=0, _boundaries[i]=層 0〜i-1 の厚みの合計
        self._boundaries = np.cumsum([0.0] + list(layer_thicknesses))
        self._cell_layer: np.ndarray | None = None
        self._contact_R_at_interface: np.ndarray | None = None  # セル界面の接触熱抵抗配列

    def setup(self, x_centers: np.ndarray) -> None:
        """セル中心座標から各セルが属する層インデックスを計算する。

        FVM1DSolver インスタンス化直後、または solve() 呼び出し前に実行する。

        【マッピングの方法】
        各セル中心 x に対して、x が属する層を numpy.searchsorted で特定する。
        _boundaries = [0, L1, L1+L2, ..., L_total] に対して
        searchsorted(x, side='left') が返す値 i は:
            _boundaries[i-1] <= x < _boundaries[i]
        → セルは層 i-1 に属する（0-indexed）。

        Parameters
        ----------
        x_centers : np.ndarray
            セル中心の座標 [m]。通常は Mesh1D.x_centers を渡す。
        """
        # searchsorted(..., side='left'): _boundaries[k-1] <= x < _boundaries[k] → k
        # 層インデックス = k - 1（boundaries[0]=0 より k>=1 は保証）
        indices = np.searchsorted(self._boundaries, x_centers, side='right') - 1
        # 境界上の点は右の層に入れ、最右端（x=L_total）は最終層に収める
        self._cell_layer = np.clip(indices, 0, len(self.layer_props) - 1)

        # セル界面ごとの接触熱抵抗配列を構築（長さ = n_cells - 1）
        # 層境界のセル界面にのみ接触熱抵抗を配置する
        n_cells = len(x_centers)
        R_contact = np.zeros(n_cells - 1, dtype=float)
        for i in range(n_cells - 1):
            layer_i = int(self._cell_layer[i])
            layer_j = int(self._cell_layer[i + 1])
            if layer_i != layer_j:
                # セル i が属する層の接触熱抵抗を適用
                R_contact[i] = self.contact_resistances[layer_i]
        self._contact_R_at_interface = R_contact

        any_contact = np.any(R_contact > 0)
        logger.debug(
            f"MultiLayerProperties.setup: {len(x_centers)} セル, "
            f"各層のセル数={[int((self._cell_layer == i).sum()) for i in range(len(self.layer_props))]}, "
            f"接触熱抵抗={'あり' if any_contact else 'なし'}"
        )

    def _check_setup(self, n_cells: int) -> None:
        """setup() が正しく呼ばれているか確認する。"""
        if self._cell_layer is None or len(self._cell_layer) != n_cells:
            raise RuntimeError(
                "MultiLayerProperties.setup(x_centers) を呼んでください。"
                "FVM1DSolver に渡す前に setup() を実行する必要があります。"
            )

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """全セルの熱伝導率 k(T) [W/(m·K)] を返す。

        各セルが属する層の props.get_k_array(T) を呼び出してまとめる。

        Parameters
        ----------
        T : np.ndarray
            各セルの温度 [°C]。

        Returns
        -------
        np.ndarray
            各セルの熱伝導率 [W/(m·K)]。
        """
        self._check_setup(len(T))
        k = np.empty_like(T, dtype=float)
        for layer_idx, props in enumerate(self.layer_props):
            mask = self._cell_layer == layer_idx
            if mask.any():
                k[mask] = props.get_k_array(T[mask])
        return k

    def get_rho_cp_array(self, T: np.ndarray) -> np.ndarray:
        """全セルの体積熱容量 ρ(T)·cp(T) [J/(m³·K)] を返す。

        Parameters
        ----------
        T : np.ndarray
            各セルの温度 [°C]。

        Returns
        -------
        np.ndarray
            各セルの体積熱容量 [J/(m³·K)]。
        """
        self._check_setup(len(T))
        rho_cp = np.empty_like(T, dtype=float)
        for layer_idx, props in enumerate(self.layer_props):
            mask = self._cell_layer == layer_idx
            if mask.any():
                rho_cp[mask] = props.get_rho_cp_array(T[mask])
        return rho_cp


# ---------------------------------------------------------------------------
# 物性値（Phase 1 用の定数物性値クラス）
# ---------------------------------------------------------------------------

@dataclass
class ConstantProperties:
    """定数物性値。Phase 1 の検証用。

    Phase 2 以降では materials.py の WoodProperties に切り替える。
    両クラスとも get_k_array(T) と get_rho_cp_array(T) を実装しており、
    FVM1DSolver はどちらでも透過的に使用できる（ダック・タイピング）。

    Parameters
    ----------
    k : float
        熱伝導率 [W/(m·K)]。
    rho : float
        密度 [kg/m³]。
    cp : float
        比熱 [J/(kg·K)]。
    """

    k: float
    rho: float
    cp: float

    @property
    def alpha(self) -> float:
        """熱拡散率 α = k / (ρ·cp) [m²/s]。"""
        return self.k / (self.rho * self.cp)

    def get_k_array(self, T: np.ndarray) -> np.ndarray:
        """全セルの熱伝導率を返す（定数）。"""
        return np.full(len(T), self.k, dtype=float)

    def get_rho_cp_array(self, T: np.ndarray) -> np.ndarray:
        """全セルの ρ·cp を返す（定数）。"""
        return np.full(len(T), self.rho * self.cp, dtype=float)


# ---------------------------------------------------------------------------
# 境界条件クラス（Phase 1 用）
# ---------------------------------------------------------------------------

@dataclass
class DirichletBC:
    """ディリクレ（温度指定）境界条件。Phase 1 の検証用。

    Parameters
    ----------
    T_surface : float
        表面温度 [°C]（定数）。
    """

    T_surface: float

    # FVM ソルバーが BC の種類を判別するための識別子
    bc_type: str = "dirichlet"

    def get_temperature(self, t: float) -> float:
        """時刻 t [s] における表面温度を返す（定数）。"""
        return float(self.T_surface)


# ---------------------------------------------------------------------------
# ヘルパー関数：炭化深さ計算
# ---------------------------------------------------------------------------

def char_depth(
    x_centers: np.ndarray,
    T: np.ndarray,
    char_temp: float = 300.0,
) -> float:
    """炭化深さ（300°C 等温面の位置）を計算する [m]。

    加熱面（x=0）からの距離として炭化フロントの位置を返す。
    温度プロファイルは単調減少であることを前提とする。

    Parameters
    ----------
    x_centers : np.ndarray
        各セル中心の座標 [m]。
    T : np.ndarray
        各セルの温度 [°C]。
    char_temp : float
        炭化温度 [°C]。Eurocode 5 では 300°C を使用。

    Returns
    -------
    float
        炭化深さ [m]。炭化がまだ始まっていない場合は 0.0。
    """
    # 炭化温度に達していないセルがなければ炭化なし
    if T[0] < char_temp:
        return 0.0

    # 全セルが炭化温度以上 → 全断面炭化
    if T[-1] >= char_temp:
        return float(x_centers[-1])

    # T[i] >= char_temp かつ T[i+1] < char_temp となる境界を探す
    idx = np.argmax(T < char_temp) - 1

    if idx < 0:
        return float(x_centers[0])

    # 2点間の線形補間で等温面の位置を求める
    x0, x1 = x_centers[idx], x_centers[idx + 1]
    T0, T1 = T[idx], T[idx + 1]
    if abs(T1 - T0) < 1e-10:
        return float(x0)
    return float(x0 + (char_temp - T0) / (T1 - T0) * (x1 - x0))


# ---------------------------------------------------------------------------
# 1D FVM ソルバー（陰解法・ピカード反復対応）
# ---------------------------------------------------------------------------

class FVM1DSolver:
    """1次元有限体積法・陰解法ソルバー。

    Phase 1 の DirichletBC・ConstantProperties と
    Phase 2 の ISO834HeatedBC・WoodProperties の両方を透過的に扱える。

    境界条件の判別はダック・タイピング：
    - bc_type == "dirichlet" → ディリクレ（温度固定）
    - get_matrix_contrib メソッドあり → ロバン（対流＋輻射）

    Parameters
    ----------
    mesh : Mesh1D
        計算格子。
    props : Any
        物性値オブジェクト（get_k_array・get_rho_cp_array を持つ）。
    bc_left : Any
        左端（加熱面）の境界条件。DirichletBC または ISO834HeatedBC。
    bc_right : Any
        右端（非加熱面）の境界条件。
        文字列 "adiabatic" / "dirichlet"、または ConvRadCoolingBC。
    T_right : float
        右端ディリクレ BC 時の温度 [°C]（bc_right=="dirichlet" の場合のみ使用）。
    T_init : float
        初期温度 [°C]（全セル均一）。
    """

    def __init__(
        self,
        mesh: Mesh1D,
        props: Any,
        bc_left: Any,
        bc_right: Any = "adiabatic",
        T_right: float = 20.0,
        T_init: float = 20.0,
        # 後方互換性：旧 bc_right_type 引数を受け付ける
        bc_right_type: str | None = None,
    ) -> None:
        self.mesh = mesh
        self.props = props
        self.bc_left = bc_left

        # 後方互換性：bc_right_type が渡された場合は bc_right として使う
        if bc_right_type is not None:
            self.bc_right: Any = bc_right_type
        else:
            self.bc_right = bc_right

        self.T_right = T_right
        self.T = np.full(mesh.n_cells, T_init, dtype=float)
        self.t = 0.0

    def _build_matrix(
        self,
        dt: float,
        t_new: float,
        T_props: np.ndarray,
        T_prev: np.ndarray,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """係数行列 A と右辺ベクトル b を構築する。

        ピカード反復のため、物性評価温度 T_props と
        前ステップ温度 T_prev を分離している。

        Parameters
        ----------
        dt : float
            時間刻み幅 [s]。
        t_new : float
            次ステップの時刻 [s]（BC の評価に使う）。
        T_props : np.ndarray
            物性値の評価に使う温度（ピカード反復時は前イテレートの温度）。
        T_prev : np.ndarray
            前ステップの温度（右辺の蓄熱項に使う）。

        Returns
        -------
        A : scipy.sparse.csr_matrix
            係数行列。
        b : np.ndarray
            右辺ベクトル。
        """
        n = self.mesh.n_cells
        dx = self.mesh.dx
        d_c = self.mesh.d_center

        # 温度依存物性値を T_props で評価
        k_arr = self.props.get_k_array(T_props)
        rho_cp = self.props.get_rho_cp_array(T_props)

        # 界面熱伝導率（調和平均）
        # k_{i+1/2} = 2·k_i·k_{i+1} / (k_i + k_{i+1})
        # ゼロ除算防止（炭化後 k が非常に小さくなる場合）
        k_sum = k_arr[:-1] + k_arr[1:]
        k_sum = np.where(k_sum < 1e-10, 1e-10, k_sum)
        k_face = 2.0 * k_arr[:-1] * k_arr[1:] / k_sum

        # 界面コンダクタンス [W/(m²·K)]
        # 接触熱抵抗 R_contact [m²K/W] がある場合は分母に加算する
        # G = 1 / (δ/k_harmonic + R_contact)
        R_face = d_c / np.maximum(k_face, 1e-10)  # 純熱伝導抵抗
        if (hasattr(self.props, "_contact_R_at_interface")
                and self.props._contact_R_at_interface is not None):
            R_face = R_face + self.props._contact_R_at_interface
        cond = 1.0 / np.maximum(R_face, 1e-12)  # コンダクタンス [W/(m²·K)]

        # 蓄熱係数
        cap = rho_cp * dx / dt  # [J/(m³·K)] / s

        # 三重対角行列の各成分を構築
        lower = -cond
        upper = -cond
        diag = cap.copy()

        # 内部セル：左右のコンダクタンスを主対角に加算
        diag[1:-1] += cond[:-1] + cond[1:]
        # 左端セル（i=0）：右側界面のコンダクタンスのみ
        diag[0] += cond[0]
        # 右端セル（i=n-1）：左側界面のコンダクタンスのみ
        diag[-1] += cond[-1]

        # 右辺：前ステップの蓄熱項
        b = cap * T_prev

        # ---- 左端境界条件の組み込み ----
        if getattr(self.bc_left, "bc_type", "") == "dirichlet":
            # ディリクレ BC：表面温度を固定
            T_s_left = self.bc_left.get_temperature(t_new)
            cond_left_bc = k_arr[0] / (0.5 * dx[0])
            diag[0] += cond_left_bc
            b[0] += cond_left_bc * T_s_left

        elif hasattr(self.bc_left, "get_matrix_contrib"):
            # ロバン BC（対流＋輻射）：前ステップの表面温度でラグ評価
            alpha_c, T_ref, q_extra = self.bc_left.get_matrix_contrib(
                T_props[0], t_new
            )
            # q"_net = alpha_c*(T_ref - T_0^{n+1}) + q_extra
            # → 対角: +alpha_c, 右辺: +alpha_c*T_ref + q_extra
            diag[0] += alpha_c
            b[0] += alpha_c * T_ref + q_extra

        # ---- 右端境界条件の組み込み ----
        bc_right = self.bc_right

        if isinstance(bc_right, str) and bc_right == "dirichlet":
            cond_right_bc = k_arr[-1] / (0.5 * dx[-1])
            diag[-1] += cond_right_bc
            b[-1] += cond_right_bc * self.T_right

        elif isinstance(bc_right, str) and bc_right == "adiabatic":
            pass  # 断熱：何もしない（熱流束ゼロが自然に実現）

        elif hasattr(bc_right, "get_matrix_contrib"):
            # ロバン BC（非加熱面：自然対流＋輻射）
            alpha_c, T_ref, q_extra = bc_right.get_matrix_contrib(
                T_props[-1], t_new
            )
            diag[-1] += alpha_c
            b[-1] += alpha_c * T_ref + q_extra

        # scipy.sparse の三重対角行列を構築（COO → CSR）
        rows = np.concatenate([
            np.arange(1, n),    # lower
            np.arange(n),       # diag
            np.arange(n - 1),   # upper
        ])
        cols = np.concatenate([
            np.arange(n - 1),   # lower
            np.arange(n),       # diag
            np.arange(1, n),    # upper
        ])
        data = np.concatenate([lower, diag, upper])
        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

        return A, b

    def step(self, dt: float, n_picard: int = 2) -> None:
        """1タイムステップ進める（ピカード反復つき）。

        【ピカード反復の流れ】
        1. T_props = T^n（前ステップ温度）で物性値を評価
        2. A(T_props)·T^{n+1,1} = b(T^n) を解く
        3. T_props = T^{n+1,1} で物性値を再評価
        4. A(T_props)·T^{n+1,2} = b(T^n) を解く
        5. 収束するまで繰り返す（通常 2〜3 回で十分）

        右辺 b は常に T^n（前ステップ）で固定することに注意。
        変わるのは左辺の係数行列のみ。

        Parameters
        ----------
        dt : float
            時間刻み幅 [s]。
        n_picard : int
            ピカード反復回数。1 = 非反復（陽的物性評価）。
        """
        T_prev = self.T.copy()  # 前ステップ温度（右辺に使う）
        t_new = self.t + dt

        T_iter = T_prev.copy()  # ピカード反復の初期値

        for picard_idx in range(n_picard):
            A, b = self._build_matrix(dt, t_new, T_iter, T_prev)
            T_new = spla.spsolve(A, b)

            # 物理チェック
            if np.any(np.isnan(T_new)) or np.any(T_new < -300.0):
                bad_idx = np.where(np.isnan(T_new) | (T_new < -300.0))[0]
                logger.error(
                    f"発散: t={self.t:.1f}s dt={dt}s picard={picard_idx} "
                    f"T_min={T_new.min():.1f} T_max={T_new.max():.1f} "
                    f"問題セル={bad_idx[:5]} "
                    f"T_prev[0]={T_prev[0]:.1f} T_props[0]={T_iter[0]:.1f}"
                )
                # 行列の診断
                A_diag = A.diagonal()
                logger.error(
                    f"行列: diag_min={A_diag.min():.3e} diag_max={A_diag.max():.3e} "
                    f"b_min={b.min():.3e} b_max={b.max():.3e}"
                )
                raise RuntimeError(
                    f"温度が発散しました（t={self.t:.1f}s, picard={picard_idx}）。"
                    f"dt={dt}s を小さくするか境界条件を確認してください。"
                    f" T_min={T_new.min():.1f}°C"
                )

            # 収束判定（温度変化が 0.1°C 以下なら早期終了）
            if picard_idx > 0:
                max_change = np.max(np.abs(T_new - T_iter))
                if max_change < 0.1:
                    break

            T_iter = T_new

        self.T = T_iter
        self.t += dt

    def solve(
        self,
        t_end: float,
        dt_base: float = 5.0,
        dt_min: float = 1.0,
        dt_max: float = 10.0,
        n_picard: int = 2,
        record_interval: float | None = None,
        record_times: list[float] | None = None,
        eval_times: list[float] | None = None,
        progress_callback=None,
        stop_event=None,
        # ---- 放冷フェーズ設定 ----
        t_cooling: float = 0.0,
        bc_left_cooling=None,
        cooling_dt_base: float = 30.0,
    ) -> dict:
        """指定時刻まで時間積分を実行する。

        【適応時間刻みの制御ルール】
        - 解析開始 30 秒以内：dt = dt_min（初期過渡を正確に捉える）
        - 蒸発帯（99〜120°C）を通過中のセルが存在：dt = dt_min（数値安定性）
        - 評価時刻（60, 75, 90 分など）の 60 秒前から：dt = 2s（精確な判定）
        - それ以外：dt = dt_base（通常運転）

        Parameters
        ----------
        t_end : float
            加熱終了時刻 [s]。
        dt_base : float
            通常運転時の時間刻み [s]。
        dt_min : float
            最小時間刻み（水蒸発帯通過時など）[s]。
        dt_max : float
            最大時間刻み [s]。
        n_picard : int
            ピカード反復回数。
        record_interval : float or None
            この間隔ごとに温度場を記録 [s]。None の場合は全ステップ記録。
        record_times : list[float] or None
            指定時刻のみを記録するリスト [s]（record_interval より優先）。
        eval_times : list[float] or None
            性能評価時刻のリスト [s]。この 60s 前から dt を小さくする。
        t_cooling : float
            加熱終了後の放冷シミュレーション時間 [s]。0 = 放冷なし。
            防耐火性能基準・評価業務方法書では加熱時間の 3 倍（60 分試験→180 分放冷
            + 安全マージン = 合計 4 時間）が規定されている。
        bc_left_cooling : Any or None
            放冷フェーズの左端（加熱面）境界条件。
            None の場合は bc_right（非加熱面と同じ自然冷却 BC）を使用する。
            通常は CoolingFurnaceBC インスタンスを渡す。
        cooling_dt_base : float
            放冷フェーズの基本時間刻み [s]。デフォルト 30 s。

        Returns
        -------
        dict
            {
              "times":        np.ndarray,          # 記録時刻 [s]
              "temperatures": np.ndarray,          # shape (n_times, n_cells)
              "x_centers":    np.ndarray,          # セル中心座標 [m]
              "char_depths":  np.ndarray,          # 炭化深さ [m] (n_times,)
              "t_heating_end": float,              # 加熱終了時刻 [s]（放冷あり時のみ）
            }
        """
        times_out: list[float] = []
        temps_out: list[np.ndarray] = []
        char_depths_out: list[float] = []

        # デフォルトの評価時刻（60, 75, 90 分）
        if eval_times is None:
            eval_times = [3600.0, 4500.0, 5400.0]  # [s]

        def _should_record(t_now: float) -> bool:
            if record_times is not None:
                return any(abs(t_now - tr) < 0.5 * dt_min for tr in record_times)
            if record_interval is not None:
                return (t_now % record_interval) < dt_min * 1.5
            return True  # デフォルト：全ステップ記録

        def _choose_dt(t_now: float) -> float:
            """現在の状態に応じて適切な時間刻みを返す。"""
            # 開始直後
            if t_now < 30.0:
                return dt_min

            # 評価時刻の 60 秒前から細かく
            for t_eval in eval_times:
                if 0 < t_eval - t_now <= 60.0:
                    return 2.0

            # 水蒸発帯（99〜120°C）を通過中のセルがある
            in_evap = np.any((self.T >= 96.0) & (self.T <= 125.0))
            if in_evap:
                return dt_min

            return dt_base

        # 初期状態を記録
        times_out.append(self.t)
        temps_out.append(self.T.copy())
        char_depths_out.append(char_depth(self.mesh.x_centers, self.T))

        logger.info(
            f"解析開始: t_end={t_end}s ({t_end/60:.1f}min), "
            f"dt_base={dt_base}s, n_picard={n_picard}"
        )

        while self.t < t_end - 1e-6:
            # 中断シグナルのチェック（Web UI の「中断」ボタン向け）
            if stop_event is not None and stop_event.is_set():
                logger.info(f"解析を中断しました: t={self.t:.1f}s")
                break

            dt = _choose_dt(self.t)
            dt = min(dt, dt_max, t_end - self.t)
            dt = max(dt, 1e-3)  # 最小刻み幅の下限

            self.step(dt, n_picard=n_picard)

            if _should_record(self.t):
                times_out.append(self.t)
                temps_out.append(self.T.copy())
                char_depths_out.append(char_depth(self.mesh.x_centers, self.T))

                # 進捗コールバック（記録タイミングで呼び出す）
                # signature: callback(current_t_s, t_end_s, T_array) -> None
                if progress_callback is not None:
                    progress_callback(self.t, t_end, self.T)

        t_heating_end = self.t
        logger.info(
            f"加熱フェーズ完了: t={self.t:.1f}s, "
            f"最高温度={self.T.max():.1f}°C, "
            f"炭化深さ={char_depths_out[-1]*1000:.1f}mm"
        )

        # ---- 放冷フェーズ ----
        if t_cooling > 0.0:
            t_cool_end = self.t + t_cooling
            bc_left_orig = self.bc_left
            if bc_left_cooling is not None:
                self.bc_left = bc_left_cooling
            # 放冷フェーズは蒸発帯チェック不要・粗い時間刻みで OK
            logger.info(
                f"放冷フェーズ開始: {t_cooling/60:.0f}分間放冷, "
                f"dt={cooling_dt_base}s"
            )
            while self.t < t_cool_end - 1e-6:
                if stop_event is not None and stop_event.is_set():
                    logger.info(f"放冷フェーズを中断しました: t={self.t:.1f}s")
                    break
                dt_c = min(cooling_dt_base, dt_max, t_cool_end - self.t)
                dt_c = max(dt_c, 1e-3)
                self.step(dt_c, n_picard=n_picard)

                if _should_record(self.t):
                    times_out.append(self.t)
                    temps_out.append(self.T.copy())
                    char_depths_out.append(char_depth(self.mesh.x_centers, self.T))
                    if progress_callback is not None:
                        progress_callback(self.t, t_cool_end, self.T)

            # BC を元に戻す
            self.bc_left = bc_left_orig
            logger.info(
                f"放冷フェーズ完了: t={self.t:.1f}s ({self.t/60:.1f}min), "
                f"最高温度={self.T.max():.1f}°C"
            )

        return {
            "times": np.array(times_out),
            "temperatures": np.array(temps_out),
            "x_centers": self.mesh.x_centers,
            "char_depths": np.array(char_depths_out),
            "t_heating_end": t_heating_end,
        }


# ---------------------------------------------------------------------------
# 解析解（Phase 1 の検証用）：半無限固体への温度ステップ応答
# ---------------------------------------------------------------------------

def analytical_semi_infinite(
    x: np.ndarray,
    t: float,
    T_init: float,
    T_surface: float,
    alpha: float,
) -> np.ndarray:
    """半無限固体への一定温度ステップ問題の解析解。

    【解析解の式】
        T(x, t) = T_init + (T_surface - T_init) · erfc(x / (2·√(α·t)))

    erfc は相補誤差関数: erfc(z) = 1 - erf(z)

    Parameters
    ----------
    x : np.ndarray
        位置 [m]。
    t : float
        時刻 [s]。
    T_init : float
        初期温度 [°C]。
    T_surface : float
        表面温度（ステップ後）[°C]。
    alpha : float
        熱拡散率 [m²/s]。

    Returns
    -------
    np.ndarray
        各位置での温度 [°C]。
    """
    from scipy.special import erfc

    eta = x / (2.0 * np.sqrt(alpha * t))
    return T_init + (T_surface - T_init) * erfc(eta)
