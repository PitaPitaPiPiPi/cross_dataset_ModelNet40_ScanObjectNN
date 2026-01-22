# Pointcloud Cross-Dataset Builder

## 概要 / Purpose

本リポジトリは **ModelNet40 (.off)** と **ScanObjectNN (.h5)** という性質の異なる2つの点群データセットを、

* 共通の前処理（中心化・正規化・サンプリング）
* 共通フォーマット（`.npy`）
* **継続学習（Continual Learning）を前提とした session / train / test 構造**

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
    main_split/
      training_objectdataset.h5
      test_objectdataset.h5
```

※ クラス順は **公式定義順を固定で使用** します。

---

## 前処理パイプライン / Preprocessing Pipeline

各サンプルに対して、以下を **必ずこの順序で** 適用します:

1. **中心化 (Centering)**

   * 重心を原点に移動
2. **正規化 (Normalization)**

   * 最大半径で割り、単位球内に収める
3. **サンプリング (Sampling)**

   * FPS (Farthest Point Sampling)
   * CUDA 前提実装（CPU fallback あり）

---

## 中間生成データ / Intermediate Outputs

### ModelNet40 → `.npy`

各 `.off` ファイルは以下に変換されます:

* 保存単位: 1サンプル = 1ファイル
* 形状: `(P, 3)`
* dtype: `float32`

```
processed/modelnet40/
  train/
    airplane/xxx.npy
    chair/yyy.npy
  test/
    airplane/zzz.npy
```

### ScanObjectNN → `.npy`

* `.h5` 内の各サンプルを分解して保存
* 保存単位: 1サンプル = 1ファイル
* 形状: `(P, 3)`

```
processed/scanobjectnn/
  train/
    class_00/000123.npy
  test/
    class_00/000987.npy
```

---

## クロスデータセット構築 / Cross-Dataset Sessions

`.npy` 化された両データセットを用いて、
**継続学習用 session データ** を生成します。

### 出力データ

```
cross_dataset/
  sessions/
    session_00/
      train.npy
      test.npy
    session_01/
      train.npy
      test.npy
```

#### train.npy / test.npy の中身

* 型: `dict`
* 内容:

```python
{
  "points": np.ndarray (N, P, 3),
  "labels": np.ndarray (N,),
  "dataset_ids": np.ndarray (N,)  # modelnet or scanobjectnn
}
```

* `dataset_ids` により、クロスドメイン実験時の識別が可能
* `cross_sessions.npy` は **session単位で自動生成** されます

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

### 1. `.off` / `.h5` → `.npy`

```bash
python scripts/build_modelnet40.py
python scripts/build_scanobjectnn.py
```

### 2. クロスデータセット生成

```bash
python scripts/build_cross_dataset.py
```

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
