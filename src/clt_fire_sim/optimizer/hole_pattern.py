"""
hole_pattern.py
===============
【役割】
孔・スリットパターンの生成モジュール。

y-z 断面（CLT パネルの幅×長さ方向）に対して
複数の候補パターン（円形孔格子・千鳥・水平スリット・縦スリット）を生成し、
FVM3DSolver に渡せる void_mask として返す。

【孔の物理的意味】
- void_mask = True のセル → 空気（k=0.026 W/m·K、ρcp=1206 J/m³·K）
- 孔は断面を貫通（x 方向全セルが void）
- 空気は木材より熱伝導率が低い → 断熱性向上の可能性
- 一方、熱容量も小さい → 早期温度上昇のリスク

【公開 API】
- HolePattern         : パターン情報のデータクラス
- make_void_mask_3d() : y-z マスクを 3D ボリュームに拡張
- make_grid_circles() : 規則格子状の円形孔
- make_staggered_circles() : 千鳥配置の円形孔
- make_horizontal_slots()  : 水平スリット
- make_vertical_slots()    : 縦スリット
- generate_candidates()    : 目標空洞率の候補パターン一覧を生成
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class HolePattern:
    """孔パターンの定義と評価結果を保持するデータクラス。

    Attributes
    ----------
    name : str
        パターン名（UI 表示用）。
    void_yz : np.ndarray
        y-z 断面の空洞マスク。shape=(ny, nz)、True=空洞。
    volume_fraction : float
        実際の空洞率（void_yz.mean()）。
    description : str
        パターンの説明（詳細テキスト）。
    params : dict
        生成パラメータ（記録・比較用）。
    """
    name: str
    void_yz: np.ndarray
    volume_fraction: float
    description: str = ""
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3D マスク変換
# ---------------------------------------------------------------------------

def make_void_mask_3d(void_yz: np.ndarray, nx: int) -> np.ndarray:
    """y-z 断面マスクを 3D 貫通孔ボリュームに拡張する。

    Parameters
    ----------
    void_yz : np.ndarray
        shape (ny, nz) の bool マスク。
    nx : int
        x 方向（厚み方向）のセル数。

    Returns
    -------
    np.ndarray
        shape (nx, ny, nz) の bool マスク。True=空洞（貫通孔）。
    """
    return np.tile(void_yz[np.newaxis, :, :], (nx, 1, 1))


# ---------------------------------------------------------------------------
# 形状プリミティブ（内部ヘルパー）
# ---------------------------------------------------------------------------

def _y_centers(ny: int, Ly: float) -> np.ndarray:
    dy = Ly / ny
    return np.arange(ny) * dy + dy / 2


def _z_centers(nz: int, Lz: float) -> np.ndarray:
    dz = Lz / nz
    return np.arange(nz) * dz + dz / 2


def _circle_mask_yz(
    ny: int, nz: int, Ly: float, Lz: float,
    cy: float, cz: float, r: float,
) -> np.ndarray:
    """1 つの円形孔の y-z マスクを生成する。"""
    Y = _y_centers(ny, Ly)[:, None]   # (ny, 1)
    Z = _z_centers(nz, Lz)[None, :]   # (1, nz)
    return (Y - cy) ** 2 + (Z - cz) ** 2 <= r ** 2


def _rect_mask_yz(
    ny: int, nz: int, Ly: float, Lz: float,
    y0: float, y1: float, z0: float, z1: float,
) -> np.ndarray:
    """1 つの矩形スリットの y-z マスクを生成する。"""
    Y = _y_centers(ny, Ly)[:, None]
    Z = _z_centers(nz, Lz)[None, :]
    return (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)


# ---------------------------------------------------------------------------
# パターン生成関数
# ---------------------------------------------------------------------------

def make_grid_circles(
    ny: int, nz: int, Ly: float, Lz: float,
    radius: float, n_y: int, n_z: int,
) -> HolePattern:
    """規則格子状の円形孔パターンを生成する。

    Parameters
    ----------
    ny, nz : int
        y, z 方向のメッシュセル数。
    Ly, Lz : float
        断面サイズ [m]。
    radius : float
        孔の半径 [m]。
    n_y, n_z : int
        y, z 方向の孔の個数。
    """
    y_pos = np.linspace(Ly / (n_y + 1), Ly * n_y / (n_y + 1), n_y)
    z_pos = np.linspace(Lz / (n_z + 1), Lz * n_z / (n_z + 1), n_z)

    mask = np.zeros((ny, nz), dtype=bool)
    for cy in y_pos:
        for cz in z_pos:
            mask |= _circle_mask_yz(ny, nz, Ly, Lz, cy, cz, radius)

    vf = float(mask.mean())
    n_holes = n_y * n_z
    return HolePattern(
        name=f"格子円 {n_holes}個 φ{radius * 1000:.0f}mm",
        void_yz=mask,
        volume_fraction=vf,
        description=f"規則格子 {n_y}×{n_z} 円形孔（半径 {radius*1000:.1f}mm）",
        params=dict(
            pattern="grid_circle",
            radius_m=radius, n_y=n_y, n_z=n_z,
        ),
    )


def make_staggered_circles(
    ny: int, nz: int, Ly: float, Lz: float,
    radius: float, spacing_y: float, spacing_z: float,
) -> HolePattern:
    """千鳥（互い違い）配置の円形孔パターンを生成する。

    Parameters
    ----------
    radius : float
        孔の半径 [m]。
    spacing_y, spacing_z : float
        y, z 方向の中心間距離 [m]。
    """
    mask = np.zeros((ny, nz), dtype=bool)
    row = 0
    y = spacing_y / 2.0
    while y < Ly:
        offset = (spacing_z / 2.0) if (row % 2 == 1) else 0.0
        z = spacing_z / 2.0 + offset
        while z < Lz:
            mask |= _circle_mask_yz(ny, nz, Ly, Lz, y, z, radius)
            z += spacing_z
        y += spacing_y
        row += 1

    vf = float(mask.mean())
    return HolePattern(
        name=f"千鳥円 φ{radius*1000:.0f}mm ピッチ{spacing_y*1000:.0f}mm",
        void_yz=mask,
        volume_fraction=vf,
        description=(
            f"千鳥配置 円形孔（半径 {radius*1000:.1f}mm、"
            f"y ピッチ {spacing_y*1000:.0f}mm、z ピッチ {spacing_z*1000:.0f}mm）"
        ),
        params=dict(
            pattern="staggered_circle",
            radius_m=radius, spacing_y=spacing_y, spacing_z=spacing_z,
        ),
    )


def make_horizontal_slots(
    ny: int, nz: int, Ly: float, Lz: float,
    slot_height: float, slot_width: float, n_slots: int,
) -> HolePattern:
    """水平スリット（z 方向に延びる帯状）パターンを生成する。

    Parameters
    ----------
    slot_height : float
        スリットの高さ（y 方向幅）[m]。
    slot_width : float
        スリットの長さ（z 方向）[m]。スリットは断面中央に配置。
    n_slots : int
        y 方向に並べるスリットの本数。
    """
    y_pos = np.linspace(Ly / (n_slots + 1), Ly * n_slots / (n_slots + 1), n_slots)
    z_margin = (Lz - slot_width) / 2.0
    z0, z1 = max(0.0, z_margin), min(Lz, z_margin + slot_width)

    mask = np.zeros((ny, nz), dtype=bool)
    for cy in y_pos:
        mask |= _rect_mask_yz(
            ny, nz, Ly, Lz,
            cy - slot_height / 2, cy + slot_height / 2,
            z0, z1,
        )

    vf = float(mask.mean())
    return HolePattern(
        name=f"水平スリット {n_slots}本 h{slot_height*1000:.0f}mm",
        void_yz=mask,
        volume_fraction=vf,
        description=(
            f"水平スリット {n_slots}本"
            f"（高さ {slot_height*1000:.0f}mm × 幅 {slot_width*1000:.0f}mm）"
        ),
        params=dict(
            pattern="h_slot",
            slot_height=slot_height, slot_width=slot_width, n_slots=n_slots,
        ),
    )


def make_vertical_slots(
    ny: int, nz: int, Ly: float, Lz: float,
    slot_width: float, slot_height: float, n_slots: int,
) -> HolePattern:
    """縦スリット（y 方向に延びる帯状）パターンを生成する。

    Parameters
    ----------
    slot_width : float
        スリットの幅（z 方向）[m]。
    slot_height : float
        スリットの高さ（y 方向）[m]。スリットは断面中央に配置。
    n_slots : int
        z 方向に並べるスリットの本数。
    """
    z_pos = np.linspace(Lz / (n_slots + 1), Lz * n_slots / (n_slots + 1), n_slots)
    y_margin = (Ly - slot_height) / 2.0
    y0, y1 = max(0.0, y_margin), min(Ly, y_margin + slot_height)

    mask = np.zeros((ny, nz), dtype=bool)
    for cz in z_pos:
        mask |= _rect_mask_yz(
            ny, nz, Ly, Lz,
            y0, y1,
            cz - slot_width / 2, cz + slot_width / 2,
        )

    vf = float(mask.mean())
    return HolePattern(
        name=f"縦スリット {n_slots}本 w{slot_width*1000:.0f}mm",
        void_yz=mask,
        volume_fraction=vf,
        description=(
            f"縦スリット {n_slots}本"
            f"（幅 {slot_width*1000:.0f}mm × 高さ {slot_height*1000:.0f}mm）"
        ),
        params=dict(
            pattern="v_slot",
            slot_width=slot_width, slot_height=slot_height, n_slots=n_slots,
        ),
    )


# ---------------------------------------------------------------------------
# 候補パターン一括生成
# ---------------------------------------------------------------------------

def generate_candidates(
    ny: int, nz: int,
    Ly: float, Lz: float,
    target_vf: float = 0.20,
    hole_types: list[str] | None = None,
    max_candidates: int = 10,
) -> list[HolePattern]:
    """目標空洞率に近い候補パターン群を生成する。

    Parameters
    ----------
    ny, nz : int
        y, z 方向のメッシュセル数。
    Ly, Lz : float
        断面サイズ [m]。
    target_vf : float
        目標空洞率（0〜1）。
    hole_types : list[str] or None
        使用する孔タイプ。None の場合はすべてを使用。
        指定可能な値: "grid_circle", "staggered_circle", "h_slot", "v_slot"
    max_candidates : int
        返す候補の最大数。

    Returns
    -------
    list[HolePattern]
        空洞率が target_vf ±10% の候補パターンリスト。
    """
    if hole_types is None:
        hole_types = ["grid_circle", "staggered_circle", "h_slot", "v_slot"]

    tol = 0.10  # ±10% 許容
    candidates: list[HolePattern] = []
    seen: set[bytes] = set()

    def _add(p: HolePattern) -> None:
        if abs(p.volume_fraction - target_vf) <= tol:
            key = p.void_yz.tobytes()
            if key not in seen:
                seen.add(key)
                candidates.append(p)

    # ── 格子円形 ────────────────────────────────────────────────────
    if "grid_circle" in hole_types:
        for n_y in [2, 3, 4, 5]:
            for n_z in [3, 4, 5, 6, 8]:
                if len(candidates) >= max_candidates:
                    break
                # 目標空洞率から半径を逆算
                area_hole = target_vf * Ly * Lz / (n_y * n_z)
                r = min(
                    np.sqrt(area_hole / np.pi),
                    Ly / (2.5 * (n_y + 1)),
                    Lz / (2.5 * (n_z + 1)),
                )
                if r < 0.005:
                    continue
                _add(make_grid_circles(ny, nz, Ly, Lz, r, n_y, n_z))

    # ── 千鳥円形 ────────────────────────────────────────────────────
    if "staggered_circle" in hole_types:
        for r_mm in [15, 20, 25, 30, 40]:
            for spacing_mult in [1.8, 2.2, 2.8, 3.5]:
                if len(candidates) >= max_candidates:
                    break
                r = r_mm / 1000.0
                sp = spacing_mult * r * 2
                if sp > min(Ly, Lz) / 2:
                    continue
                _add(make_staggered_circles(ny, nz, Ly, Lz, r, sp, sp))

    # ── 水平スリット ─────────────────────────────────────────────────
    if "h_slot" in hole_types:
        for n_slots in [2, 3, 4]:
            for h_mm in [15, 20, 30, 40]:
                if len(candidates) >= max_candidates:
                    break
                h = h_mm / 1000.0
                # 目標空洞率から幅を逆算
                w = target_vf * Ly * Lz / (n_slots * h)
                w = min(w, Lz * 0.85)
                if w < 0.02:
                    continue
                _add(make_horizontal_slots(ny, nz, Ly, Lz, h, w, n_slots))

    # ── 縦スリット ───────────────────────────────────────────────────
    if "v_slot" in hole_types:
        for n_slots in [2, 3, 4]:
            for w_mm in [15, 20, 30, 40]:
                if len(candidates) >= max_candidates:
                    break
                w = w_mm / 1000.0
                h = target_vf * Ly * Lz / (n_slots * w)
                h = min(h, Ly * 0.85)
                if h < 0.02:
                    continue
                _add(make_vertical_slots(ny, nz, Ly, Lz, w, h, n_slots))

    # 候補が 0 の場合はデフォルトを追加
    if not candidates:
        candidates.append(
            make_grid_circles(ny, nz, Ly, Lz,
                              min(0.03, Ly / 8), 3, 4)
        )

    return candidates[:max_candidates]
