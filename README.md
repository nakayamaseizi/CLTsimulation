# CLT耐火性能 3次元熱シミュレータ

CLT（Cross-Laminated Timber）の耐火・準耐火性能を評価するための
3次元非定常熱伝導シミュレータです。

## 開発フェーズ

### Phase 1：1D・一定物性での基礎ソルバー ✅
- 単層・一定物性（k, ρ, cp 定数）で1D熱伝導を実装
- 表面温度ステップ境界条件
- 半無限固体の解析解との比較検証

### Phase 2：温度依存物性値（Eurocode 5）の導入
- `materials.py` 実装、線形補間関数
- ピカード反復で物性値更新
- 5層スギCLTで炭化速度検証

### Phase 3：境界条件（ISO 834 + 対流輻射）の正式実装
- `boundary.py` 実装
- 輻射項のピカード反復

### Phase 4：多層構成への対応
- 設定ファイル（YAML）読込
- 材料データベース実装

### Phase 5：3D化
- 3D FVM ソルバー
- 側面境界条件

### Phase 6：可視化・自動判定レポート
- matplotlib グラフ
- PyVista 3D可視化
- 判定レポート自動生成

## インストール

```bash
pip install -r requirements.txt
pip install -e .
```

## 使い方

```bash
clt-fire-sim run config/default.yaml
```
