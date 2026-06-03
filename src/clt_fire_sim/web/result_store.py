"""
result_store.py
===============
【役割】
シミュレーション結果の HDF5 形式での保存・読込と、
outputs/ フォルダのスキャン（履歴一覧表示用）。

【HDF5 構造】
result.h5
├── attrs: version, timestamp, run_id
├── config/   yaml (str)
├── mesh/     x_centers (float32)
├── time/     values (float32)
├── fields/
│   ├── temperature  (float32, shape=(Nt, Nx), gzip=4)
│   └── char_depths  (float32, shape=(Nt,),    gzip=4)
└── results/  evaluation_json (str)

【公開関数】
- save_result_hdf5()  : 結果辞書を HDF5 + metadata.json に保存する
- load_result_hdf5()  : HDF5 から結果を読み込む
- scan_results()      : outputs/ フォルダをスキャンして履歴リストを返す
- make_run_id()       : 実行 ID 文字列を生成する
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# HDF5 ファイルフォーマットバージョン
HDF5_VERSION = "1.0"


# ---------------------------------------------------------------------------
# 実行 ID 生成
# ---------------------------------------------------------------------------

def make_run_id(specimen_name: str) -> str:
    """実行 ID 文字列を生成する。

    形式: YYYYMMDD_HHMMSS_{試験体名（安全文字のみ）}

    Parameters
    ----------
    specimen_name : str
        試験体名（日本語可）。

    Returns
    -------
    str
        例: "20260524_130000_5層スギCLT_150mm"
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ファイルシステムで使えない文字を "_" に置換し、長さを制限
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", specimen_name)[:40]
    return f"{ts}_{safe_name}"


# ---------------------------------------------------------------------------
# HDF5 保存
# ---------------------------------------------------------------------------

def save_result_hdf5(
    result: dict[str, Any],
    config: Any,
    run_id: str,
    out_dir: Path,
) -> Path:
    """シミュレーション結果を HDF5 形式で保存する。

    同時に metadata.json も生成する（履歴スキャン用）。

    Parameters
    ----------
    result : dict
        solve() の返り値（times, temperatures, x_centers, char_depths,
        evaluation キーを含む）。
    config : CLTConfig
        シミュレーション設定。
    run_id : str
        実行 ID（make_run_id() で生成）。
    out_dir : Path
        出力ディレクトリ（存在しない場合は自動作成）。

    Returns
    -------
    Path
        保存した HDF5 ファイルのパス。
    """
    import h5py
    import yaml

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = out_dir / "result.h5"

    with h5py.File(h5_path, "w") as f:
        # ── メタデータ属性 ──────────────────────────────────────────
        f.attrs["version"] = HDF5_VERSION
        f.attrs["timestamp"] = datetime.now().isoformat()
        f.attrs["run_id"] = run_id

        # ── 設定 YAML ──────────────────────────────────────────────
        from clt_fire_sim.web.config_editor import config_to_yaml_str
        cfg_yaml = config_to_yaml_str(config)
        cfg_grp = f.create_group("config")
        cfg_grp.create_dataset("yaml", data=cfg_yaml)

        # ── メッシュ ────────────────────────────────────────────────
        mesh_grp = f.create_group("mesh")
        mesh_grp.create_dataset(
            "x_centers",
            data=result["x_centers"].astype(np.float32),
        )

        # ── 時刻 ────────────────────────────────────────────────────
        time_grp = f.create_group("time")
        time_grp.create_dataset(
            "values",
            data=result["times"].astype(np.float32),
        )

        # ── 温度場・炭化深さ（圧縮保存）─────────────────────────────
        fields_grp = f.create_group("fields")
        T_arr = result["temperatures"]
        # 3D の場合は shape=(Nt, Nx*Ny*Nz)、1D では shape=(Nt, Nx)
        fields_grp.create_dataset(
            "temperature",
            data=T_arr.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        fields_grp.create_dataset(
            "char_depths",
            data=result["char_depths"].astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )

        # ── 性能評価結果 ─────────────────────────────────────────────
        results_grp = f.create_group("results")
        evaluation = result.get("evaluation", {})
        results_grp.create_dataset(
            "evaluation_json",
            data=json.dumps(evaluation, ensure_ascii=False),
        )

    # ── metadata.json（フォルダスキャン用）──────────────────────────
    _save_metadata_json(result, config, run_id, out_dir, h5_path.name)

    return h5_path


def _save_metadata_json(
    result: dict,
    config: Any,
    run_id: str,
    out_dir: Path,
    h5_filename: str,
) -> None:
    """metadata.json を生成する（scan_results() で高速スキャンするため）。"""
    from clt_fire_sim.report import compute_charring_rate

    evaluation = result.get("evaluation", {})
    total_mm = sum(layer.thickness_mm for layer in config.specimen.layers)
    beta = compute_charring_rate(result)

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "specimen_name": config.specimen.name,
        "n_layers": len(config.specimen.layers),
        "total_thickness_mm": total_mm,
        "t_end_min": config.simulation.t_end_min,
        "analysis_mode": "1D",
        "char_rate_beta": round(beta, 4) if not np.isnan(beta) else None,
        "evaluation": evaluation,
        "h5_filename": h5_filename,
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# HDF5 読込
# ---------------------------------------------------------------------------

def load_result_hdf5(h5_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """HDF5 ファイルから結果を読み込む。

    Parameters
    ----------
    h5_path : Path
        HDF5 ファイルのパス。

    Returns
    -------
    tuple[dict, dict]
        (result_dict, metadata_dict)
        result_dict は solve() の返り値と同じ形式。
    """
    import h5py

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        metadata = dict(f.attrs)

        result = {
            "x_centers": f["mesh"]["x_centers"][:].astype(np.float64),
            "times": f["time"]["values"][:].astype(np.float64),
            "temperatures": f["fields"]["temperature"][:].astype(np.float64),
            "char_depths": f["fields"]["char_depths"][:].astype(np.float64),
            "evaluation": json.loads(f["results"]["evaluation_json"][()]),
        }

        config_yaml = f["config"]["yaml"][()]
        if isinstance(config_yaml, (bytes, bytearray)):
            config_yaml = config_yaml.decode("utf-8")
        metadata["config_yaml"] = config_yaml

    # 保存されていた YAML から CLTConfig を復元（失敗しても result は返す）
    try:
        from clt_fire_sim.web.config_editor import load_config_from_yaml_str
        result["config"] = load_config_from_yaml_str(config_yaml)
    except Exception:
        pass  # config キーなしで返す（viewer が sidebar config にフォールバック）

    return result, metadata


# ---------------------------------------------------------------------------
# フォルダスキャン（履歴一覧）
# ---------------------------------------------------------------------------

def scan_results(outputs_dir: Path) -> list[dict[str, Any]]:
    """outputs/ フォルダをスキャンして解析履歴リストを返す。

    各サブフォルダの metadata.json を読み込む。
    metadata.json が無いフォルダはスキップする。

    Parameters
    ----------
    outputs_dir : Path
        スキャン対象のルートディレクトリ。

    Returns
    -------
    list[dict]
        metadata 辞書のリスト（新しい順）。
        各辞書に "run_dir" キー（フォルダの絶対パス文字列）を追加する。
    """
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return []

    results: list[dict] = []
    for subdir in sorted(outputs_dir.iterdir(), reverse=True):
        if not subdir.is_dir():
            continue
        meta_path = subdir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            meta["run_dir"] = str(subdir)
            meta["h5_path"] = str(subdir / meta.get("h5_filename", "result.h5"))
            results.append(meta)
        except Exception:
            continue

    return results
