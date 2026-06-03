"""
presets.py
==========
【役割】
プリセット CLT 設定の定義。
サイドバーの「プリセット読込」ボタンから呼び出される。

各プリセットは CLTConfig と互換性のある辞書形式で定義する。
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# プリセット定義
# ---------------------------------------------------------------------------
# 構造: {表示名: CLTConfig 互換辞書}
# 材料キーは materials.MATERIAL_DB のキーと一致させること。

PRESETS: dict[str, dict[str, Any]] = {

    "5層スギCLT 150mm（標準）": {
        "description": "最も一般的な 5 層スギ CLT。建築基準法 90 分準耐火評価向け。",
        "specimen": {
            "name": "5層スギCLT 150mm",
            "layers": [
                {"name": "第1層（加熱面）", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第2層", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第3層（中心）", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第4層", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第5層（非加熱面）", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
            ],
        },
        "simulation": {"t_end_min": 90.0, "n_cells_per_layer": 12},
    },

    "3層スギCLT 90mm（60分耐火）": {
        "description": "3 層スギ CLT 90mm。建築基準法 60 分耐火評価向け。",
        "specimen": {
            "name": "3層スギCLT 90mm",
            "layers": [
                {"name": "第1層（加熱面）", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第2層（中心）", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第3層（非加熱面）", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 12},
    },

    "5層ヒノキCLT 150mm": {
        "description": "ヒノキ（檜）5 層 CLT。スギより密度が高く炭化が遅い。",
        "specimen": {
            "name": "5層ヒノキCLT 150mm",
            "layers": [
                {"name": "第1層（加熱面）", "material": "hinoki", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 430.0, "moisture_content": 0.12},
                {"name": "第2層", "material": "hinoki", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 430.0, "moisture_content": 0.12},
                {"name": "第3層（中心）", "material": "hinoki", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 430.0, "moisture_content": 0.12},
                {"name": "第4層", "material": "hinoki", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 430.0, "moisture_content": 0.12},
                {"name": "第5層（非加熱面）", "material": "hinoki", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 430.0, "moisture_content": 0.12},
            ],
        },
        "simulation": {"t_end_min": 90.0, "n_cells_per_layer": 12},
    },

    "7層スギCLT 210mm（90分準耐火）": {
        "description": "7 層スギ CLT 210mm。より厚い CLT で長時間耐火性を評価。",
        "specimen": {
            "name": "7層スギCLT 210mm",
            "layers": [
                {"name": "第1層（加熱面）", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第2層", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第3層", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第4層（中心）", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第5層", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第6層", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "第7層（非加熱面）", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
            ],
        },
        "simulation": {"t_end_min": 120.0, "n_cells_per_layer": 12},
    },

    "5層ベイマツCLT 150mm": {
        "description": "ベイマツ（ダグラスファー）5 層 CLT。密度高め、北米産 CLT 相当。",
        "specimen": {
            "name": "5層ベイマツCLT 150mm",
            "layers": [
                {"name": "第1層（加熱面）", "material": "douglas_fir", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 500.0, "moisture_content": 0.12},
                {"name": "第2層", "material": "douglas_fir", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 500.0, "moisture_content": 0.12},
                {"name": "第3層（中心）", "material": "douglas_fir", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 500.0, "moisture_content": 0.12},
                {"name": "第4層", "material": "douglas_fir", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 500.0, "moisture_content": 0.12},
                {"name": "第5層（非加熱面）", "material": "douglas_fir", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 500.0, "moisture_content": 0.12},
            ],
        },
        "simulation": {"t_end_min": 90.0, "n_cells_per_layer": 12},
    },
}

# プリセット名リスト（UI でのプルダウン順序）
PRESET_NAMES: list[str] = list(PRESETS.keys())
