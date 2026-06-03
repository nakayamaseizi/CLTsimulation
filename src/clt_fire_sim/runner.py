"""
runner.py
=========
【役割】
CLT 耐火シミュレーションの高レベル実行インターフェース。

YAML 設定ファイルまたは CLTConfig オブジェクトからシミュレーションを実行し、
温度場・炭化深さ・性能評価結果を一括して返す。

【使い方（CLIの場合）】
    python -m clt_fire_sim.runner configs/clt_5layer_sugi.yaml

【使い方（Python スクリプトの場合）】
    from clt_fire_sim.runner import run_from_yaml
    result = run_from_yaml("configs/clt_5layer_sugi.yaml")
    print(result["evaluation"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .boundary import ConvRadCoolingBC, ISO834HeatedBC
from .config import CLTConfig, load_config
from .materials import make_properties
from .solver.fvm_1d import FVM1DSolver, MultiLayerProperties, make_multi_layer_mesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# メインランナー
# ---------------------------------------------------------------------------

def run_from_config(config: CLTConfig) -> dict[str, Any]:
    """CLTConfig オブジェクトからシミュレーションを実行する。

    設定に従ってメッシュ・物性値・境界条件を構築し、
    FVM1DSolver で時間積分を実行する。

    Parameters
    ----------
    config : CLTConfig
        シミュレーション設定（load_config() で読み込んだもの）。

    Returns
    -------
    dict
        solve() の返す辞書に以下を追加したもの:
        - "config"     : CLTConfig    使用した設定オブジェクト
        - "evaluation" : dict         各評価時刻での性能判定結果

    Examples
    --------
    >>> config = load_config("configs/clt_5layer_sugi.yaml")
    >>> result = run_from_config(config)
    >>> result["evaluation"]["60min"]["insulation_ok"]
    True
    """
    sim = config.simulation
    bc_cfg = config.boundary
    spec = config.specimen
    eval_cfg = config.evaluation

    logger.info(f"シミュレーション開始: {spec.name}")

    # ---- 1. メッシュ生成 ----
    layer_thicknesses_m = [layer.thickness_mm / 1000.0 for layer in spec.layers]
    total_thickness_mm = sum(layer.thickness_mm for layer in spec.layers)
    logger.info(
        f"  試験体: {len(spec.layers)}層, 総厚={total_thickness_mm:.0f}mm, "
        f"各層={[f'{t*1000:.0f}mm' for t in layer_thicknesses_m]}"
    )

    mesh = make_multi_layer_mesh(
        layer_thicknesses=layer_thicknesses_m,
        n_cells_per_layer=sim.n_cells_per_layer,
        ratio=sim.mesh_ratio,
    )

    # ---- 2. 多層物性値の構築 ----
    layer_props = [
        make_properties(
            material=layer.material,
            rho_0=layer.rho_0_kg_m3,
            moisture_content=layer.moisture_content,
        )
        for layer in spec.layers
    ]

    props = MultiLayerProperties(
        layer_thicknesses=layer_thicknesses_m,
        layer_props=layer_props,
    )
    props.setup(mesh.x_centers)

    # ---- 3. 境界条件の構築 ----
    bc_left = ISO834HeatedBC(
        alpha_c=bc_cfg.heated.alpha_c,
        eps_m=bc_cfg.heated.eps_m,
        eps_f=bc_cfg.heated.eps_f,
    )
    bc_right = ConvRadCoolingBC(
        alpha_c=bc_cfg.unheated.alpha_c,
        eps_m=bc_cfg.unheated.eps_m,
        T_inf=bc_cfg.unheated.T_inf,
    )

    # ---- 4. ソルバー実行 ----
    T_init = bc_cfg.unheated.T_inf  # 初期温度 = 外気温度
    solver = FVM1DSolver(
        mesh=mesh,
        props=props,
        bc_left=bc_left,
        bc_right=bc_right,
        T_init=T_init,
    )

    eval_times_s = [t * 60.0 for t in eval_cfg.eval_times_min]
    result = solver.solve(
        t_end=sim.t_end_min * 60.0,
        dt_base=sim.dt_base_s,
        dt_min=sim.dt_min_s,
        dt_max=sim.dt_max_s,
        n_picard=sim.n_picard,
        record_interval=sim.record_interval_s,
        eval_times=eval_times_s,
    )

    # ---- 5. 性能評価 ----
    result["config"] = config
    result["evaluation"] = _evaluate_performance(result, config)

    logger.info("シミュレーション完了")
    for t_key, judgment in result["evaluation"].items():
        status = "OK" if judgment["insulation_ok"] else "NG"
        logger.info(
            f"  {t_key}: 炭化深さ={judgment['char_depth_mm']:.1f}mm, "
            f"非加熱面={judgment['unheated_face_temp_C']:.1f}°C "
            f"(+{judgment['unheated_face_rise_K']:.1f}K) [{status}]"
        )

    return result


def _evaluate_performance(result: dict, config: CLTConfig) -> dict[str, dict]:
    """各評価時刻での耐火・遮熱性能を判定する。

    【判定基準】
    遮熱性: 非加熱面温度 < unheated_face_temp_limit (= 初期温度 + 140K)
    炭化深さ: 300°C 等温面位置（参考値）

    Parameters
    ----------
    result : dict
        solver.solve() の返り値。
    config : CLTConfig
        評価基準を含む設定オブジェクト。

    Returns
    -------
    dict[str, dict]
        {
          "60min": {
            "char_depth_mm": float,
            "unheated_face_temp_C": float,
            "unheated_face_rise_K": float,
            "insulation_ok": bool,
          },
          "75min": {...},
          "90min": {...},
        }
    """
    eval_cfg = config.evaluation
    T_init = config.boundary.unheated.T_inf
    times_min = result["times"] / 60.0
    judgments: dict[str, dict] = {}

    for t_eval_min in eval_cfg.eval_times_min:
        # 評価時刻に最も近いレコードを選択
        idx = int(np.argmin(np.abs(times_min - t_eval_min)))
        T_profile = result["temperatures"][idx]
        char_d_mm = result["char_depths"][idx] * 1000.0
        T_unheated = float(T_profile[-1])
        delta_T = T_unheated - T_init

        insulation_ok = T_unheated < eval_cfg.unheated_face_temp_limit

        judgments[f"{t_eval_min:.0f}min"] = {
            "char_depth_mm": round(char_d_mm, 2),
            "unheated_face_temp_C": round(T_unheated, 2),
            "unheated_face_rise_K": round(delta_T, 2),
            "insulation_ok": insulation_ok,
        }

    return judgments


def run_from_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """YAML 設定ファイルからシミュレーションを実行する。

    load_config() で設定を読み込み、run_from_config() を呼ぶ簡便関数。

    Parameters
    ----------
    yaml_path : str or Path
        YAML 設定ファイルのパス。

    Returns
    -------
    dict
        run_from_config() と同じ形式の結果辞書。

    Examples
    --------
    >>> result = run_from_yaml("configs/clt_5layer_sugi.yaml")
    >>> result["evaluation"]["60min"]["char_depth_mm"]
    39.9
    """
    config = load_config(yaml_path)
    return run_from_config(config)


# ---------------------------------------------------------------------------
# CLI エントリポイント（直接実行用）
# ---------------------------------------------------------------------------

def _cli_entry() -> None:
    """pyproject.toml の [project.scripts] から呼ばれる CLI エントリポイント。"""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "configs/clt_5layer_sugi.yaml"
    result = run_from_yaml(yaml_path)
    print("\n=== 性能評価結果 ===")
    for t_key, judgment in result["evaluation"].items():
        status = "合格" if judgment["insulation_ok"] else "不合格"
        print(
            f"  {t_key}: "
            f"炭化深さ={judgment['char_depth_mm']:.1f}mm, "
            f"非加熱面={judgment['unheated_face_temp_C']:.1f}°C "
            f"(+{judgment['unheated_face_rise_K']:.1f}K), "
            f"遮熱性={status}"
        )


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "configs/clt_5layer_sugi.yaml"
    result = run_from_yaml(yaml_path)

    print("\n=== 性能評価結果 ===")
    for t_key, judgment in result["evaluation"].items():
        status = "合格" if judgment["insulation_ok"] else "不合格"
        print(
            f"  {t_key}: "
            f"炭化深さ={judgment['char_depth_mm']:.1f}mm, "
            f"非加熱面={judgment['unheated_face_temp_C']:.1f}°C "
            f"(+{judgment['unheated_face_rise_K']:.1f}K), "
            f"遮熱性={status}"
        )
