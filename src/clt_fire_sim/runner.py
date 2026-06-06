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

from .boundary import ConvRadCoolingBC, CoolingFurnaceBC, ISO834HeatedBC
from .config import CLTConfig, load_config
from .materials import (
    ConstantProperties,
    PerforatedWoodAdvanced,
    SlittedWoodProperties,
    WoodProperties,
    make_properties,
)
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
        _build_layer_properties(layer)
        for layer in spec.layers
    ]

    # 各層の接触熱抵抗（その層の非加熱面側界面）
    contact_resistances = [
        getattr(layer, "contact_resistance_m2KW", 0.0)
        for layer in spec.layers
    ]
    props = MultiLayerProperties(
        layer_thicknesses=layer_thicknesses_m,
        layer_props=layer_props,
        contact_resistances=contact_resistances,
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

    # 放冷フェーズ用の境界条件
    t_heat_end_s = sim.t_end_min * 60.0
    bc_left_cooling = None
    t_cooling_s = 0.0
    if sim.cooling_time_h > 0.0:
        from .boundary import iso834_temperature  # 局所インポート（循環回避）
        T_at_heat_end = iso834_temperature(sim.t_end_min)
        bc_left_cooling = CoolingFurnaceBC(
            T_end_C=T_at_heat_end,
            t_heat_end_s=t_heat_end_s,
            tau_s=sim.cooling_tau_min * 60.0,
            alpha_c=bc_cfg.heated.alpha_c,
            eps_m=bc_cfg.heated.eps_m,
            eps_f=bc_cfg.heated.eps_f,
            T_ambient=bc_cfg.unheated.T_inf,
        )
        t_cooling_s = sim.cooling_time_h * 3600.0
        logger.info(
            f"  放冷フェーズ設定: {sim.cooling_time_h:.1f}時間, "
            f"τ={sim.cooling_tau_min:.0f}分, "
            f"加熱終了時炉内温度={T_at_heat_end:.0f}°C"
        )

    eval_times_s = [t * 60.0 for t in eval_cfg.eval_times_min]
    result = solver.solve(
        t_end=t_heat_end_s,
        dt_base=sim.dt_base_s,
        dt_min=sim.dt_min_s,
        dt_max=sim.dt_max_s,
        n_picard=sim.n_picard,
        record_interval=sim.record_interval_s,
        eval_times=eval_times_s,
        t_cooling=t_cooling_s,
        bc_left_cooling=bc_left_cooling,
        cooling_dt_base=max(sim.dt_base_s, 30.0),
    )

    # ---- 5. 性能評価 ----
    result["config"] = config
    result["evaluation"] = _evaluate_performance(result, config)

    # ---- 6. 燃え止まり判定（放冷フェーズがある場合のみ）----
    if eval_cfg.char_stop_enabled and t_cooling_s > 0.0:
        result["char_stop"] = _evaluate_char_stop(result, config, mesh)
    else:
        result["char_stop"] = None

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


def _build_layer_properties(layer):
    """LayerConfig から適切な物性値オブジェクトを生成する。

    material_type に応じて以下のオブジェクトを返す:
    - "wood"                    : WoodProperties（Eurocode 5、デフォルト）
    - "perforated_wood"         : PerforatedWoodProperties（単純並列混合則）
    - "perforated_wood_advanced": PerforatedWoodAdvanced（池畑 2021 実験式）
    - "slitted_wood"            : SlittedWoodProperties（スリット加工モデル）
    - "custom"                  : ConstantProperties（ユーザー定義定数物性値）

    Parameters
    ----------
    layer : LayerConfig
        層の設定オブジェクト。

    Returns
    -------
    物性値オブジェクト（get_k_array, get_rho_cp_array を持つ）。
    """
    mtype = layer.material_type

    if mtype == "custom":
        if layer.k_W_mK is None or layer.cp_J_kgK is None:
            raise ValueError(
                f"material_type='custom' の層には k_W_mK と cp_J_kgK が必要です。"
            )
        return ConstantProperties(
            k=layer.k_W_mK,
            rho=layer.rho_0_kg_m3,
            cp=layer.cp_J_kgK,
        )

    # ベースとなる木材物性値を取得（k_char_factor も反映）
    kcf = getattr(layer, "k_char_factor", 1.0)
    base_props = make_properties(
        material=layer.material,
        rho_0=layer.rho_0_kg_m3,
        moisture_content=layer.moisture_content,
        k_char_factor=kcf,
    )

    if mtype == "wood":
        return base_props

    if mtype == "perforated_wood":
        from .materials import PerforatedWoodProperties
        return PerforatedWoodProperties(base_props, layer.void_fraction)

    if mtype == "perforated_wood_advanced":
        return PerforatedWoodAdvanced(
            base_props=base_props,
            void_fraction=layer.void_fraction,
            hole_diameter_mm=layer.hole_diameter_mm,
            hole_depth_mm=layer.hole_depth_mm,
            layer_thickness_mm=layer.thickness_mm,
        )

    if mtype == "slitted_wood":
        return SlittedWoodProperties(
            base_props=base_props,
            slit_width_mm=layer.slit_width_mm,
            slit_depth_mm=layer.slit_depth_mm,
            slit_pitch_mm=layer.slit_pitch_mm,
            layer_thickness_mm=layer.thickness_mm,
        )

    raise ValueError(
        f"未知の material_type: '{mtype}'. "
        "wood | perforated_wood | perforated_wood_advanced | slitted_wood | custom のいずれかを指定してください。"
    )


def _evaluate_char_stop(result: dict, config: CLTConfig, mesh) -> dict:
    """放冷フェーズ終了時の燃え止まり（自消）を判定する。

    【判定基準（鷹野研究室の実験に基づく）】
    1. 放冷終了時点（simulation.cooling_time_h 後）に：
       a. 構造層表面温度 ≤ evaluation.char_stop_struct_temp_limit (デフォルト 100°C)
       b. 試験体内最高温度 ≤ evaluation.char_stop_max_temp_limit (デフォルト 250°C)
    → 上記を両方満たす場合「燃え止まり」と判定。

    【構造層の特定】
    evaluation.structural_layer_index で指定されたインデックスの層の加熱面側端セルを
    「構造層表面」と定義する。デフォルトは最終層（-1 = 非加熱面側）の最初のセル。

    Parameters
    ----------
    result : dict
        solver.solve() の返り値。
    config : CLTConfig
        設定オブジェクト。
    mesh : Mesh1D
        計算格子。

    Returns
    -------
    dict
        {
          "char_stop_ok": bool,      # 燃え止まりか否か
          "struct_temp_C": float,    # 放冷終了時の構造層表面温度 [°C]
          "max_temp_C": float,       # 放冷終了時の試験体内最高温度 [°C]
          "t_eval_s": float,         # 評価時刻 [s]（=総解析時間）
          "reason": str,             # 判定理由の説明文
        }
    """
    eval_cfg = config.evaluation
    sim_cfg = config.simulation
    spec = config.specimen

    # 評価時刻 = 最後のレコード
    times = result["times"]
    temps = result["temperatures"]  # shape (n_times, n_cells)
    t_eval = float(times[-1])
    T_final = temps[-1]

    # 構造層の先頭セルインデックスを特定
    layer_idx = eval_cfg.structural_layer_index
    n_layers = len(spec.layers)
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx  # -1 → 最終層
    layer_idx = int(np.clip(layer_idx, 0, n_layers - 1))

    # 層の境界座標から対応セルを特定
    layer_starts_m = np.cumsum([0.0] + [la.thickness_mm / 1000.0 for la in spec.layers])
    struct_x_start = layer_starts_m[layer_idx]  # 構造層の加熱面側境界
    x_centers = result["x_centers"]
    # 構造層の加熱面側端（最初）のセル
    struct_cell_idx = int(np.argmin(np.abs(x_centers - struct_x_start)))

    T_struct = float(T_final[struct_cell_idx])
    T_max = float(T_final.max())

    ok_struct = T_struct <= eval_cfg.char_stop_struct_temp_limit
    ok_max = T_max <= eval_cfg.char_stop_max_temp_limit
    char_stop_ok = ok_struct and ok_max

    if char_stop_ok:
        reason = (
            f"燃え止まり OK: "
            f"構造層表面 {T_struct:.1f}°C ≤ {eval_cfg.char_stop_struct_temp_limit:.0f}°C, "
            f"最高温度 {T_max:.1f}°C ≤ {eval_cfg.char_stop_max_temp_limit:.0f}°C"
        )
    else:
        reasons = []
        if not ok_struct:
            reasons.append(
                f"構造層表面 {T_struct:.1f}°C > {eval_cfg.char_stop_struct_temp_limit:.0f}°C"
            )
        if not ok_max:
            reasons.append(
                f"最高温度 {T_max:.1f}°C > {eval_cfg.char_stop_max_temp_limit:.0f}°C"
            )
        reason = "燃え止まり NG: " + ", ".join(reasons)

    logger.info(f"  燃え止まり判定（t={t_eval/60:.0f}分）: {reason}")

    return {
        "char_stop_ok": char_stop_ok,
        "struct_temp_C": round(T_struct, 2),
        "max_temp_C": round(T_max, 2),
        "t_eval_s": t_eval,
        "reason": reason,
    }


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
