"""
system_info.py
==============
【役割】
psutil を使って PC スペックを検出し、
解析前の警告表示やメッシュ自動設定に利用する。

【公開関数】
- get_system_info()           : CPU・メモリの現在状態を取得
- estimate_memory_gb()        : 解析に必要な推定メモリ量を返す
- recommend_mesh_option()     : 利用可能メモリから推奨メッシュを選択
- check_resources_warning()   : 実行前の警告メッセージを生成
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# PC スペック取得
# ---------------------------------------------------------------------------

def get_system_info() -> dict[str, Any]:
    """psutil で現在の CPU・メモリ状況を取得する。

    Returns
    -------
    dict
        - total_ram_gb     : 総物理メモリ [GB]
        - available_ram_gb : 利用可能メモリ [GB]
        - ram_used_pct     : メモリ使用率 [%]
        - cpu_count        : 論理 CPU コア数
        - cpu_pct          : CPU 使用率 [%]（0.5 秒の平均）
        - cpu_freq_ghz     : CPU 周波数 [GHz]（取得できない場合は 0.0）
    """
    import psutil

    mem = psutil.virtual_memory()
    freq = psutil.cpu_freq()

    return {
        "total_ram_gb": mem.total / 1e9,
        "available_ram_gb": mem.available / 1e9,
        "ram_used_pct": mem.percent,
        "cpu_count": psutil.cpu_count(logical=True) or 1,
        "cpu_pct": psutil.cpu_percent(interval=0.5),
        "cpu_freq_ghz": (freq.current / 1000.0) if freq else 0.0,
    }


def get_live_usage() -> dict[str, float]:
    """解析中の CPU・メモリ使用率を即時取得する（ポーリング用）。

    interval=None にすることで前回計測からの差分を即座に返す。
    """
    import psutil

    mem = psutil.virtual_memory()
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "ram_used_pct": mem.percent,
        "available_ram_gb": mem.available / 1e9,
    }


# ---------------------------------------------------------------------------
# メモリ推定
# ---------------------------------------------------------------------------

def estimate_memory_gb(
    n_cells_per_layer: int,
    n_layers: int,
    t_end_min: float,
    record_interval_s: float = 30.0,
    mode: str = "1D",
    n_cells_y: int = 1,
    n_cells_z: int = 1,
) -> float:
    """解析に必要な推定メモリ量を返す [GB]。

    温度場配列（float32）のサイズを基準に計算する。
    実際のメモリ使用量はスパース行列や Python オーバーヘッドを含むため
    この推定値の 3〜5 倍程度になる場合がある。

    Parameters
    ----------
    n_cells_per_layer : int
        層あたりのセル数。
    n_layers : int
        CLT 層数。
    t_end_min : float
        解析時間 [分]。
    record_interval_s : float
        記録間隔 [s]。
    mode : str
        "1D" または "3D"。
    n_cells_y, n_cells_z : int
        3D モード時の y, z 方向セル数。

    Returns
    -------
    float
        推定メモリ量 [GB]。
    """
    nx = n_cells_per_layer * n_layers
    ny = n_cells_y if mode == "3D" else 1
    nz = n_cells_z if mode == "3D" else 1
    n_spatial = nx * ny * nz

    # 記録タイムステップ数
    n_times = int(t_end_min * 60.0 / record_interval_s) + 2

    # float32 (4 bytes) × Nt × Nx × Ny × Nz
    bytes_temp = 4 * n_times * n_spatial
    # char_depths, times などの補助配列
    bytes_aux = 4 * n_times * 3

    total_bytes = bytes_temp + bytes_aux
    # スパース行列 + Python オーバーヘッド（係数 4 で見積もり）
    return total_bytes * 4 / 1e9


# ---------------------------------------------------------------------------
# 推奨設定
# ---------------------------------------------------------------------------

def recommend_mesh_option(available_ram_gb: float) -> str:
    """利用可能メモリから推奨メッシュ細かさを返す。

    Parameters
    ----------
    available_ram_gb : float
        現在の利用可能メモリ [GB]。

    Returns
    -------
    str
        "粗い（計算速い・精度低め）" / "標準（推奨）" / "細かい（精度高・計算遅め）"
    """
    if available_ram_gb >= 8.0:
        return "細かい（精度高・計算遅め）"
    if available_ram_gb >= 4.0:
        return "標準（推奨）"
    return "粗い（計算速い・精度低め）"


# ---------------------------------------------------------------------------
# 実行前警告
# ---------------------------------------------------------------------------

def check_resources_warning(
    estimated_mem_gb: float,
    available_ram_gb: float,
    mode: str = "1D",
    mesh_option: str = "標準（推奨）",
) -> list[str]:
    """実行前の警告メッセージリストを返す。

    空リストなら警告なし（実行 OK）。

    Parameters
    ----------
    estimated_mem_gb : float
        推定メモリ使用量 [GB]。
    available_ram_gb : float
        現在の利用可能メモリ [GB]。
    mode : str
        解析モード。
    mesh_option : str
        メッシュ選択肢。

    Returns
    -------
    list[str]
        警告メッセージのリスト（空なら警告なし）。
    """
    warnings: list[str] = []

    # メモリ不足チェック
    if estimated_mem_gb > available_ram_gb * 0.8:
        warnings.append(
            f"⚠️ 推定メモリ使用量（{estimated_mem_gb:.1f} GB）が "
            f"利用可能メモリ（{available_ram_gb:.1f} GB）の 80% を超えています。\n"
            f"   → メッシュを「粗い」に変更するか、解析時間を短くしてください。"
        )

    # 3D + 細かいメッシュの組み合わせ
    if mode == "3D" and "細かい" in mesh_option:
        warnings.append(
            "⚠️ 3D モード ＋ 細かいメッシュの組み合わせは非常に計算が重くなります。\n"
            "   → まず 1D または「標準」メッシュで試してください。"
        )

    # メモリが 2GB 未満
    if available_ram_gb < 2.0:
        warnings.append(
            f"⚠️ 利用可能メモリが少なすぎます（{available_ram_gb:.1f} GB）。\n"
            f"   他のアプリを閉じてからもう一度試してください。"
        )

    return warnings
