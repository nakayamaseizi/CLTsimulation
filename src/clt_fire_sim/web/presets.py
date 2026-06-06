"""
presets.py
==========
【役割】
プリセット CLT 設定の定義。
サイドバーの「プリセット読込」ボタンから呼び出される。

【出典】
各プリセットは以下の論文に基づく実験試験体を再現している：
- 吉原（2017） 鹿児島大学学部本論
- 林田（2018） 鹿児島大学修士本論
- 大脇（2018） 鹿児島大学学部梗概
- 池畑（2019） 鹿児島大学学部本論
- 田村（2019） 鹿児島大学修士本論
- 中村恭子（2020） 鹿児島大学学部本論
- 柴田（2021） 鹿児島大学修士本論
- 池畑（2021） 鹿児島大学修士本論
- 中村（2022） 鹿児島大学修士本論
- 朱（2023） 鹿児島大学学部本論
- 中尾（2024） 鹿児島大学学部本論
- 朱（2025） 鹿児島大学修士本論
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# プリセット定義
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {

    # ═══════════════════════════════════════════════════════════
    # 🏛️ 基本 CLT（標準仕様）
    # ═══════════════════════════════════════════════════════════

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

    "7層スギCLT 210mm（90分準耐火）": {
        "description": "7 層スギ CLT 210mm。より厚い CLT で長時間耐火性を評価。",
        "specimen": {
            "name": "7層スギCLT 210mm",
            "layers": [
                {"name": f"第{i+1}層", "material": "sugi", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 400.0, "moisture_content": 0.12}
                for i in range(7)
            ],
        },
        "simulation": {"t_end_min": 120.0, "n_cells_per_layer": 12},
    },

    "5層ヒノキCLT 150mm": {
        "description": "ヒノキ（檜）5 層 CLT。スギより密度が高く炭化が遅い。",
        "specimen": {
            "name": "5層ヒノキCLT 150mm",
            "layers": [
                {"name": f"第{i+1}層", "material": "hinoki", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 430.0, "moisture_content": 0.12}
                for i in range(5)
            ],
        },
        "simulation": {"t_end_min": 90.0, "n_cells_per_layer": 12},
    },

    "5層ベイマツCLT 150mm": {
        "description": "ベイマツ（ダグラスファー）5 層 CLT。密度高め、北米産 CLT 相当。",
        "specimen": {
            "name": "5層ベイマツCLT 150mm",
            "layers": [
                {"name": f"第{i+1}層", "material": "douglas_fir", "thickness_mm": 30.0,
                 "rho_0_kg_m3": 500.0, "moisture_content": 0.12}
                for i in range(5)
            ],
        },
        "simulation": {"t_end_min": 90.0, "n_cells_per_layer": 12},
    },

    # ═══════════════════════════════════════════════════════════
    # 🔬 研究室試験体 ― 付加型モデル
    #    出典：田村（2019）・林田（2018）
    # ═══════════════════════════════════════════════════════════

    "[田村2019] 付加型-0.35（スギ+IB 66mm + CLT90mm）": {
        "description": (
            "田村修士2019・池畑2019 試験体II。\n"
            "スギ12mm×3＋インシュレーションボード10mm×3（66mm）＋スギCLT90mm。\n"
            "60分準耐火性能あり。CLT面温度：45分52℃、60分85℃。"
        ),
        "specimen": {
            "name": "付加型-0.35 スギ+IB+CLT",
            "layers": [
                {"name": "スギ表層", "material": "sugi",
                 "thickness_mm": 12.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "IB断熱層①", "material": "insulation_board",
                 "thickness_mm": 10.0, "rho_0_kg_m3": 244.0, "moisture_content": 0.12},
                {"name": "スギ②", "material": "sugi",
                 "thickness_mm": 12.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "IB断熱層②", "material": "insulation_board",
                 "thickness_mm": 10.0, "rho_0_kg_m3": 244.0, "moisture_content": 0.12},
                {"name": "スギ③", "material": "sugi",
                 "thickness_mm": 12.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "IB断熱層③", "material": "insulation_board",
                 "thickness_mm": 10.0, "rho_0_kg_m3": 244.0, "moisture_content": 0.12},
                {"name": "CLT第1層（加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    "[田村2019] 付加型-0.53（スギ有孔 40mm + CLT90mm）": {
        "description": (
            "田村修士2019・池畑2019 試験体I。\n"
            "有孔スギ10mm×2＋スギ10mm×2（40mm）＋スギCLT90mm。\n"
            "CLT面温度：45分99℃、60分132℃。"
        ),
        "specimen": {
            "name": "付加型-0.53 有孔スギ+CLT",
            "layers": [
                {"name": "有孔スギ①", "material": "sugi",
                 "thickness_mm": 10.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12,
                 "material_type": "perforated_wood", "void_fraction": 0.34},
                {"name": "スギ無孔①", "material": "sugi",
                 "thickness_mm": 10.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "有孔スギ②", "material": "sugi",
                 "thickness_mm": 10.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12,
                 "material_type": "perforated_wood", "void_fraction": 0.34},
                {"name": "スギ無孔②", "material": "sugi",
                 "thickness_mm": 10.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "CLT第1層（加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    "[林田2018] スギ+炭化コルク+CLT 3層": {
        "description": (
            "林田修士2018 試験体。スギ24mm＋炭化コルク30mm＋CLT90mm（144mm）。\n"
            "断熱性能測定（U値試験）対象試験体。"
        ),
        "specimen": {
            "name": "スギ+炭化コルク+CLT 144mm",
            "layers": [
                {"name": "スギ表層（加熱面）", "material": "sugi",
                 "thickness_mm": 24.0, "rho_0_kg_m3": 390.0, "moisture_content": 0.12},
                {"name": "炭化コルク断熱層", "material": "charred_cork",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 130.0, "moisture_content": 0.0},
                {"name": "CLT第1層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    "[林田2018] スギ+ファルカタ+CLT 3層": {
        "description": (
            "林田修士2018 試験体。スギ24mm＋ファルカタ31mm＋CLT90mm（145mm）。\n"
            "60分加熱後CLT面温度：93.3℃（準耐火性能確認）。"
        ),
        "specimen": {
            "name": "スギ+ファルカタ+CLT 145mm",
            "layers": [
                {"name": "スギ表層（加熱面）", "material": "sugi",
                 "thickness_mm": 24.0, "rho_0_kg_m3": 390.0, "moisture_content": 0.12},
                {"name": "ファルカタ耐火層", "material": "falcata",
                 "thickness_mm": 31.0, "rho_0_kg_m3": 280.0, "moisture_content": 0.12},
                {"name": "CLT第1層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    "[林田2018] スギ+アカガシ+CLT 3層": {
        "description": (
            "林田修士2018 試験体。スギ24mm＋アカガシ30mm＋CLT90mm（144mm）。\n"
            "60分加熱後CLT面温度：53.1℃（最良性能グループ）。"
        ),
        "specimen": {
            "name": "スギ+アカガシ+CLT 144mm",
            "layers": [
                {"name": "スギ表層（加熱面）", "material": "sugi",
                 "thickness_mm": 24.0, "rho_0_kg_m3": 390.0, "moisture_content": 0.12},
                {"name": "アカガシ耐火層", "material": "akagashi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 850.0, "moisture_content": 0.12},
                {"name": "CLT第1層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    "[林田2018] スギ+竹集成材+CLT 3層": {
        "description": (
            "林田修士2018 試験体。スギ24mm＋竹集成材30mm＋CLT90mm（144mm）。\n"
            "60分加熱後CLT面温度：66.5℃（優秀な耐火性能）。"
        ),
        "specimen": {
            "name": "スギ+竹集成材+CLT 144mm",
            "layers": [
                {"name": "スギ表層（加熱面）", "material": "sugi",
                 "thickness_mm": 24.0, "rho_0_kg_m3": 390.0, "moisture_content": 0.12},
                {"name": "竹集成材耐火断熱層", "material": "bamboo_glulam",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 600.0, "moisture_content": 0.12},
                {"name": "CLT第1層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    "[林田2018] スギ+キリ+CLT 3層": {
        "description": (
            "林田修士2018 試験体。スギ24mm＋キリ30mm＋CLT90mm（144mm）。\n"
            "60分加熱後CLT面温度：104.2℃。超軽量木材の耐火性能評価。"
        ),
        "specimen": {
            "name": "スギ+キリ+CLT 144mm",
            "layers": [
                {"name": "スギ表層（加熱面）", "material": "sugi",
                 "thickness_mm": 24.0, "rho_0_kg_m3": 390.0, "moisture_content": 0.12},
                {"name": "キリ断熱層", "material": "kiri",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 296.0, "moisture_content": 0.12},
                {"name": "CLT第1層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    "[林田2018] スギ+炭化コルク+アカガシ+CLT 4層": {
        "description": (
            "林田修士2018 4層試験体。スギ24mm＋炭化コルク30mm＋アカガシ15mm＋CLT90mm。\n"
            "断熱性能と耐火性能を両立する複合構成。"
        ),
        "specimen": {
            "name": "スギ+炭化コルク+アカガシ+CLT 159mm",
            "layers": [
                {"name": "スギ表層（加熱面）", "material": "sugi",
                 "thickness_mm": 24.0, "rho_0_kg_m3": 390.0, "moisture_content": 0.12},
                {"name": "炭化コルク断熱層", "material": "charred_cork",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 130.0, "moisture_content": 0.0},
                {"name": "アカガシ耐火層", "material": "akagashi",
                 "thickness_mm": 15.0, "rho_0_kg_m3": 850.0, "moisture_content": 0.12},
                {"name": "CLT第1層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    # ═══════════════════════════════════════════════════════════
    # 🔬 研究室試験体 ― 一体型モデル
    #    出典：田村（2019）・柴田（2021）
    # ═══════════════════════════════════════════════════════════

    "[田村2019] 一体型-S スギ7層7プライ 168mm": {
        "description": (
            "田村修士2019 一体型-S試験体。スギ24mm×7層7プライ（168mm）。\n"
            "CLT面温度：45分87℃、60分113〜121℃。"
        ),
        "specimen": {
            "name": "一体型-S スギ7層 168mm",
            "layers": [
                {"name": f"第{i+1}層", "material": "sugi",
                 "thickness_mm": 24.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12}
                for i in range(7)
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    "[柴田2021] スリット付加型 スギ13mm×5+CLT90mm": {
        "description": (
            "柴田修士2021 中型試験体（スリット付加型）。\n"
            "スリット加工スギ13mm×5層（65mm）＋スギCLT90mm。\n"
            "スリット幅15mm深3mm。CLT面温度：45分36〜45℃、60分90〜91℃。\n"
            "60分準耐火性能あり。"
        ),
        "specimen": {
            "name": "スリット付加型 スギ×5+CLT90mm",
            "layers": [
                {"name": "スリットスギ①（加熱面）", "material": "sugi",
                 "thickness_mm": 13.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12,
                 "material_type": "perforated_wood", "void_fraction": 0.08},
                {"name": "スリットスギ②", "material": "sugi",
                 "thickness_mm": 13.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12,
                 "material_type": "perforated_wood", "void_fraction": 0.08},
                {"name": "スリットスギ③", "material": "sugi",
                 "thickness_mm": 13.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12,
                 "material_type": "perforated_wood", "void_fraction": 0.08},
                {"name": "スリットスギ④", "material": "sugi",
                 "thickness_mm": 13.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12,
                 "material_type": "perforated_wood", "void_fraction": 0.08},
                {"name": "スリットスギ⑤", "material": "sugi",
                 "thickness_mm": 13.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12,
                 "material_type": "perforated_wood", "void_fraction": 0.08},
                {"name": "CLT第1層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 8},
    },

    "[柴田2021] 異種中型付加型 スギ+炭化コルク+アカガシ+CLT": {
        "description": (
            "柴田修士2021 中型試験体（異種付加型）。\n"
            "スギ15mm＋炭化コルク25mm＋アカガシ20mm＋CLT90mm（150mm）。\n"
            "U値0.53 W/m²K。60分時CLT面温度 100℃前後。"
        ),
        "specimen": {
            "name": "異種中型付加型 スギ+CC+アカガシ+CLT",
            "layers": [
                {"name": "スギ表層（加熱面）", "material": "sugi",
                 "thickness_mm": 15.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "炭化コルク断熱層", "material": "charred_cork",
                 "thickness_mm": 25.0, "rho_0_kg_m3": 130.0, "moisture_content": 0.0},
                {"name": "アカガシ耐火層", "material": "akagashi",
                 "thickness_mm": 20.0, "rho_0_kg_m3": 850.0, "moisture_content": 0.12},
                {"name": "CLT第1層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    # ═══════════════════════════════════════════════════════════
    # 🔬 研究室試験体 ― 燃え止まり型
    #    出典：朱（2023）
    # ═══════════════════════════════════════════════════════════

    "[朱2023] 燃え止まり型 スギ21mm+炭化コルク75mm+CLT90mm（燃え止まり確認）": {
        "description": (
            "朱学部2023 試験体 S21+CC75。\n"
            "スギ21mm＋炭化コルク75mm＋CLT90mm（186mm）。\n"
            "自消（燃え止まり）を実験で確認した構成。\n"
            "炭化コルク75mmが遅燃断熱層として機能し放冷期に自消。"
        ),
        "specimen": {
            "name": "燃え止まり型 S21+CC75+CLT90",
            "layers": [
                {"name": "スギ表層（加熱面）", "material": "sugi",
                 "thickness_mm": 21.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "炭化コルク遅燃断熱層", "material": "charred_cork",
                 "thickness_mm": 75.0, "rho_0_kg_m3": 130.0, "moisture_content": 0.0},
                {"name": "CLT第1層（構造層）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    "[朱2023] スギ15mm+炭化コルク75mm+CLT90mm": {
        "description": (
            "朱学部2023 試験体 S15+CC75。\n"
            "スギ15mm＋炭化コルク75mm＋CLT90mm（180mm）。\n"
            "自消（燃え止まり）を確認。"
        ),
        "specimen": {
            "name": "S15+CC75+CLT90",
            "layers": [
                {"name": "スギ表層（加熱面）", "material": "sugi",
                 "thickness_mm": 15.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "炭化コルク遅燃断熱層", "material": "charred_cork",
                 "thickness_mm": 75.0, "rho_0_kg_m3": 130.0, "moisture_content": 0.0},
                {"name": "CLT第1層（構造層）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 410.0, "moisture_content": 0.11},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

    # ═══════════════════════════════════════════════════════════
    # 🔬 研究室試験体 ― 吉原2017 断熱性能最高構成
    # ═══════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════
    # 🔥 燃え止まり型CLT ― 不燃木板（難燃処理スギ）シリーズ
    #    出典: 伯耆原ら2019 / 中村2022 / 中尾2024 / 朱2025
    # ═══════════════════════════════════════════════════════════

    "[中村2022/中尾2024] 標準試験体 S24+FR50+CLT90mm（燃え止まり型基準）": {
        "description": (
            "燃え止まり型CLTの標準試験体（基準構成）。\n"
            "燃えしろ層: スギ 24mm ／ 燃え止まり層: 不燃処理スギ 50mm（薬剤注入180 kg/m³）"
            "／ 構造CLT: スギ 90mm。\n\n"
            "【文献実験結果】中尾2024・朱2023 で1時間耐火性能（燃え止まり）を確認。\n\n"
            "【シミュレーターの注意】難燃薬剤の吸熱反応（195°C分解→自消）は純熱伝導モデルで"
            "は再現できません。このため燃え止まり判定は「保守側（NG方向）」に評価されます。\n"
            "温度分布・炭化深さの参照には使えますが、燃え止まりの合否は実験値を参照してください。"
        ),
        "specimen": {
            "name": "標準試験体 S24+FR50+CLT90",
            "layers": [
                {"name": "燃えしろ層（スギ）", "material": "sugi",
                 "thickness_mm": 24.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "燃え止まり層（不燃処理スギ 50mm）", "material": "fr_sugi",
                 "thickness_mm": 50.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "構造CLT 第1層（スギ）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "構造CLT 第2層（スギ）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "構造CLT 第3層（スギ・非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
            ],
        },
        "simulation": {
            "t_end_min": 60.0,
            "n_cells_per_layer": 10,
            "cooling_time_h": 4.0,
            "cooling_tau_min": 45.0,
        },
        "evaluation": {
            "char_stop_enabled": False,   # FR化学反応未モデル化のため参考値のみ
            "structural_layer_index": 2,
        },
    },

    "[中村2022] 基準構成（薄め）S20+FR50+CLT90mm": {
        "description": (
            "中村2022の基準構成バリエーション。燃えしろ層を20mmに薄めた検証ケース。\n"
            "燃えしろ20mm：燃え止まり25mm（FR25）で1時間耐火性能確認（中村2022）。\n"
            "【注意】難燃薬剤の吸熱反応は熱伝導モデル外。燃え止まり判定は参考値。"
        ),
        "specimen": {
            "name": "基準構成バリ S20+FR50+CLT90",
            "layers": [
                {"name": "燃えしろ層（スギ）", "material": "sugi",
                 "thickness_mm": 20.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "燃え止まり層（不燃処理スギ 50mm）", "material": "fr_sugi",
                 "thickness_mm": 50.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "構造CLT 第1層（スギ）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "構造CLT 第2層（スギ）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "構造CLT 第3層（スギ・非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
            ],
        },
        "simulation": {
            "t_end_min": 60.0,
            "n_cells_per_layer": 10,
            "cooling_time_h": 4.0,
            "cooling_tau_min": 45.0,
        },
        "evaluation": {
            "char_stop_enabled": False,
            "structural_layer_index": 2,
        },
    },

    "[中尾2024] FR標準 vs 小片ラミナ 比較（S24+FR50 基準）": {
        "description": (
            "中尾2024の比較実験における標準試験体。\n"
            "不燃木小片ラミナ（FRC）や無機物小片ラミナとの性能比較の基準として用いられた。\n"
            "燃えしろ層スギ24mm + 不燃処理スギ50mm + CLT90mm（合計164mm）。\n"
            "実験では燃え止まり（自消）を確認。シミュレーターは温度分布の参照用途向け。"
        ),
        "specimen": {
            "name": "S24+FR50+CLT90（中尾2024標準）",
            "layers": [
                {"name": "燃えしろ層（スギ 24mm）", "material": "sugi",
                 "thickness_mm": 24.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "燃え止まり層（不燃処理スギ 50mm）", "material": "fr_sugi",
                 "thickness_mm": 50.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "構造CLT 第1層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "構造CLT 第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "構造CLT 第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
            ],
        },
        "simulation": {
            "t_end_min": 60.0,
            "n_cells_per_layer": 10,
            "cooling_time_h": 4.0,
            "cooling_tau_min": 45.0,
        },
        "evaluation": {
            "char_stop_enabled": False,
            "structural_layer_index": 2,
        },
    },

    "[吉原2017] スギ+炭化コルク30mm+CLT 断熱最高（U=0.46）": {
        "description": (
            "吉原学部2017 最良断熱試験体。スギ30mm＋炭化コルク30mm＋CLT90mm（150mm）。\n"
            "実測U値0.464 W/m²K（試験体中最高断熱性能）。\n"
            "炭化コルクは着火後急速燃焼に注意（継目対策が必要）。"
        ),
        "specimen": {
            "name": "スギ+炭化コルク30+CLT 150mm",
            "layers": [
                {"name": "スギ表層（加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "炭化コルク断熱層", "material": "charred_cork",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 130.0, "moisture_content": 0.0},
                {"name": "CLT第1層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "CLT第2層", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
                {"name": "CLT第3層（非加熱面）", "material": "sugi",
                 "thickness_mm": 30.0, "rho_0_kg_m3": 400.0, "moisture_content": 0.12},
            ],
        },
        "simulation": {"t_end_min": 60.0, "n_cells_per_layer": 10},
    },

}

# プリセット名リスト（UI でのプルダウン順序）
PRESET_NAMES: list[str] = list(PRESETS.keys())
