# Pointcloud Cross-Dataset Builder

## 概要 / Purpose

本リポジトリは **ModelNet40 (.off)** と **ScanObjectNN (.h5)** という性質の異なる2つの点群データセットを、

* 共通の前処理（中心化・正規化・サンプリング）
* 共通フォーマット（`.npy`）
* **class_id ベースの train / test 構造**

に統一し、**クロスデータセット実験**（例: ModelNet→ScanObjectNN, ScanObjectNN→ModelNet）を再現性高く構築することを目的としています。

継続学習リポジトリ本体とは **完全に分離したデータセット生成専用リポジトリ** であり、

* 再生成可能
* 実験条件の明示化
* データリーク防止
  を重視した設計になっています。

---

## 入力データ / Required Datasets

### 1. ModelNet40

* 形式: `.off`
* 単位: 1ファイル = 1サンプル
* 各ファイルは `(N, 3)` の頂点座標を含む

期待ディレクトリ構造:

```
raw_datasets/
  modelnet40/
    airplane/
      train/*.off
      test/*.off
    chair/
      train/*.off
      test/*.off
    ... (40 classes)
```

### 2. ScanObjectNN

* 形式: `.h5`
* 各 `.h5` に以下を含む:

  * `data`: `(num_samples, num_points, 3)`
  * `label`: `(num_samples,)`

期待ディレクトリ構造:

```
raw_datasets/
  scanobjectnn/
    h5_files/
      main_split_nobg/
        training_objectdataset.h5
        test_objectdataset.h5
```

※ クラス順は **公式定義順を固定で使用** します。

---

## 前処理パイプライン / Preprocessing Pipeline

実装上の重要な点（コードの現状）:

- **サンプリング（FPS を含む）→ 中心化 → 正規化** の順で処理が行われます。
  - `pc_normalize_unified(...)` は内部でまず FPS（`use_fps=True`）を実行し、その後に重心計算と正規化を行います（つまり FPS の前後で正規化は「後」です）。
  - ModelNet はメッシュからまず `sample_surface_n` 点を抽出し、続けて FPS（`fps_k`）で所望の点数に落とします。
  - ScanObjectNN は HDF5 から読み出した点群に対して直接 FPS（`num_points`）を適用します。

- **軸変換**: ModelNet のメッシュ由来処理では `openshape=True` により Y/Z 軸を入れ替える処理が実装されています。
- **FPS の実装とシード**: `scripts/utils/fps.py` がバックエンド選択（`pytorch3d` → `modelnet2_ops` → NumPy フォールバック）を行い、デフォルトシードは `42` です。

注記: `configs/preprocessing.yaml` に `percentile_scale: 99` が定義されていますが、現在の正規化実装（`pc_normalize_unified`）では使用されていません。`scripts/utils/normalize.py` 内には距離の上位パーセンタイルでスケールを取る補助関数 `compute_centroid_and_scale` が存在しますが、実際のフローでは使われず、代わりに各点の重心からの**最大距離 (`maxd`)** で割る実装になっています。

必要であれば `compute_centroid_and_scale` を正規化フローで利用するよう修正できます（たとえば外れ値を無視してスケールを決めたい場合など）。

---

## 中間生成データ / Intermediate Outputs

### ModelNet40 → `.npy`

各 `.off` ファイルは以下に変換されます:

* 保存単位: 1サンプル = 1ファイル
* 形状: `(P, 3)`
* dtype: `float32`

```
out_root/
  ModelNet/
    airplane/
      train/*.npy
      test/*.npy
    chair/
      train/*.npy
      test/*.npy
    ... (40 classes)
```

### ScanObjectNN → `.npy`

* `.h5` 内の各サンプルを分解して保存
* 保存単位: 1サンプル = 1ファイル
* 形状: `(P, 3)`

```
out_root/
  ScanObjectNN/
    Bag/
      train/*.npy
      test/*.npy
    Bed/
      train/*.npy
      test/*.npy
    ... (15 classes)
```

---
## クロスデータセット構築 / Cross-Dataset (class_id)

`.npy` 化された両データセットを用いて、
**class_id 単位の train / test 構造** を生成します。

### 出力データ

```
out_root/
  modelnet_scanobjectnn/
    0/
      train/*.npy
      test/*.npy
    1/
      train/*.npy
      test/*.npy
    ...
    36/
      train/*.npy
      test/*.npy
```

### class_id 対応表

* ModelNet
  * 0: airplane
  * 1: bathtub
  * 2: bottle
  * 3: bowl
  * 4: car
  * 5: cone
  * 6: cup
  * 7: curtain
  * 8: flower pot
  * 9: glass box
  * 10: guitar
  * 11: keyboard
  * 12: lamp
  * 13: laptop
  * 14: mantel
  * 15: night stand
  * 16: person
  * 17: piano
  * 18: plant
  * 19: radio
  * 20: range hood
  * 21: stairs
  * 22: tent
  * 23: tv stand
  * 24: vase
  * 25: xbox
* ScanObjectNN
  * 26: Cabinet
  * 27: Chair
  * 28: Desk
  * 29: Display
  * 30: Door
  * 31: Shelf
  * 32: Table
  * 33: Bed
  * 34: Sink
  * 35: Sofa
  * 36: Toilet

---

## 環境構築 / Environment Setup

```bash
conda create -n pc_builder python=3.9
conda activate pc_builder
pip install -r requirements.txt
```

CUDA を使用する場合:

* PyTorch >= 1.8
* CUDA >= 11.x

---

## 実行手順 / How to Run

### 1. ModelNet40 → `.npy`

```bash
python -m scripts/build_modelnet \
  --modelnet_root raw_datasets/modelnet40 \
  --out_root outputs
```

主なオプション:

* `--modelnet_root`（必須）: ModelNet40 のルート
* `--out_root`（必須）: 出力先（例: outputs）
* `--sample_surface_n`: サンプリング元点数（既定: 10000）
* `--workers`: 並列数（既定: 4）

### 2. ScanObjectNN → `.npy`

```bash
python -m scripts/build_scanobjectnn \
  --h5_root raw_datasets/scanobjectnn/h5_files \
  --out_root outputs
```

主なオプション:

* `--h5_root`（必須）: `h5_files` のルート
* `--out_root`（必須）: 出力先（例: outputs）
* `--split_dir`: 既定 `main_split_nobg`
* `--train_h5`: 既定 `training_objectdataset.h5`
* `--test_h5`: 既定 `test_objectdataset.h5`
* `--workers`: 並列数（既定: 4）

### 3. クロスデータセット生成（class_id 単位）

```bash
python -m scripts/build_cross_sessions \
  --modelnet_root outputs/ModelNet \
  --scanobjectnn_root outputs/ScanObjectNN \
  --out_root outputs
```

主なオプション:

* `--modelnet_root`（必須）: ModelNet の `.npy` 出力ルート
* `--scanobjectnn_root`（必須）: ScanObjectNN の `.npy` 出力ルート
* `--out_root`（必須）: 出力先（例: outputs）
ログは `logs/` 以下に自動保存されます。

---

## 想定用途 / Intended Use

* Continual Learning (class-incremental / domain-incremental)
* クロスデータセット汎化性能評価
* 点群バックボーン（PointNet++, PointTransformer, Uni3D 等）の共通ベンチマーク

---

## 設計方針まとめ

* データ生成と学習コードを完全分離
* 再実験可能性を最優先
* shape / class / session の曖昧さを排除

研究用途での長期運用を前提とした、
**壊れにくく・説明可能なデータセット生成基盤**です。

---

## Visualization / クロスデータセット構造の可視化

生成されたクロスデータセットが期待どおりの構造・クラス分布になっているかを確認するための可視化スクリプトを同梱しています。

### 使い方（例）

リポジトリルートから実行します。`out_root` は `modelnet_scanobjectnn/` が置かれている出力先を指定してください。

```bash
python scripts/visualize_cross_sessions.py \
  --out_root outputs
```

### 出力

`out_root/modelnet_scanobjectnn/visualization/` に次のファイルが保存されます:

* `train_class_counts.png` — train のクラスごとのサンプル数の棒グラフ
* `test_class_counts.png` — test のクラスごとのサンプル数の棒グラフ

また、コンソールには簡潔なサマリが表示されます（例）：

```
=== Dataset Summary ===
Train samples : 00000
Test samples  : 00000
```

### 依存パッケージ

可視化には `matplotlib` を使用します。`requirements.txt` に `matplotlib` を追加し、環境にインストールしてください。

```bash
pip install -r requirements.txt
```

### 注意点

* スクリプトは `out_root/modelnet_scanobjectnn/{class_id}/{train|test}/*.npy` が存在することを前提とします。これらが存在しない場合は先に `build_cross_sessions.py` を実行してください。
* 必要なら `--out_dir` オプションで可視化画像の出力先を明示できます。

---
