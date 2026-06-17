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

## ShapeNet / CO3D / S2C 追加パイプライン

既存の ModelNet40 / ScanObjectNN / ModelNet→ScanObjectNN (`modelnet_scanobjectnn`) の出力互換性は維持し、別系統として ShapeNet / CO3D / ShapeNet→CO3D (`shapenet_co3d`, S2C) を追加しています。

### 追加出力構造

```text
outputs/
  ShapeNet/<class_name>/train|test/*.npy
  CO3D/<category>/train|test/*.npy
  shapenet_co3d/<class_id>/train|test/*.npy
```

全 `.npy` は原則 `(1024, 3)`, `float32`, finite, 中心化・単位球正規化済みです。保存単位は既存と同じく 1 sample = 1 `.npy` + `.json` metadata です。

`raw_datasets/`, `outputs/`, `external/co3d/`, `*.zip`, `*_in_progress/` は git 管理しません。CO3D は巨大なため、一括 download ではなく category-wise download を推奨します。公式 CO3D コードは vendoring せず、`external/co3d` または `--co3d_repo` で参照します。

### ShapeNet 前処理

```bash
python -m scripts.build_shapenet \
  --shapenet_root /path/to/ShapeNet \
  --out_root outputs \
  --num_points 1024 \
  --workers 4
```

入力は `.obj`, `.off`, `.ply`, `.npy` を自動判定します。ShapeNetCore の `synset_id/model_id/models/model_normalized.obj` に対応します。raw 側に `class_name/train|test/*` がある場合はその split を尊重し、split がない場合は class ごとに deterministic split を作り、`outputs/ShapeNet/split_manifest.json` に保存します。

raw folder 名は環境差があるため、必要に応じて `configs/shapenet_class_map.json` の `aliases` と `synset_id` を編集してください。出力 class dir は canonical name に統一されます。

### CO3D を全量保持せず category-wise に前処理する方法

CO3D 公式 downloader は外部 clone したものを使います。`scripts.download_and_build_co3d_category` は category ごとに download → preprocess → raw cleanup を逐次実行し、前処理済み `.npy` だけを `outputs/CO3D` に蓄積します。

```bash
mkdir -p external
git clone https://github.com/facebookresearch/co3d.git external/co3d

python -m scripts.download_and_build_co3d_category \
  --co3d_repo external/co3d \
  --download_folder raw_datasets/co3d \
  --out_root outputs \
  --all_categories \
  --num_points 1024 \
  --workers 4 \
  --cleanup_raw
```

単一カテゴリだけ処理する場合:

```bash
python -m scripts.download_and_build_co3d_category \
  --co3d_repo external/co3d \
  --download_folder raw_datasets/co3d \
  --out_root outputs \
  --category apple \
  --num_points 1024 \
  --workers 4 \
  --cleanup_raw
```

raw subset を残して確認したい場合:

```bash
python -m scripts.download_and_build_co3d_category \
  --co3d_repo external/co3d \
  --download_folder raw_datasets/co3d \
  --out_root outputs \
  --category apple \
  --num_points 1024 \
  --workers 4 \
  --no_cleanup_raw
```

複数 category は `--categories apple,backpack`、全 category は `--all_categories` で指定します。何も指定しない場合は error になります。既定では `--skip_existing` と `--single_sequence_subset` が有効で、`outputs/CO3D/<category>/train|test/*.npy` が既に存在する category は download せず skip します。再作成したい場合だけ `--overwrite` を指定してください。

既定では preprocess 成功後に `raw_datasets/co3d/<category>/` と category 対応 archive / temporary files を削除します。preprocess 失敗時は再実行・デバッグのため raw subset を残し、`--force_cleanup_on_error` を指定した場合のみ失敗時も cleanup します。`--no_cleanup_raw` 指定時は成功時も raw subset を残します。cleanup 対象は raw subset だけであり、`outputs/CO3D`, `external/co3d`, `configs` は削除しません。

wrapper は `external/co3d/co3d/download_dataset.py --help` を subprocess で確認し、`--download_categories` など category 指定可能な引数が確認できる場合だけ download を実行します。カテゴリ指定できない downloader では、安全のため一括 download を開始しません。

`--download_only` は download のみ、`--process_only` は download 済みデータの前処理のみを実行します。download が中断されて `raw_datasets/co3d/_in_progress` が残った場合は、公式 downloader の再開可否を確認し、必要なら該当 category を整理してから再実行してください。

### CO3D 前処理のみ

```bash
python -m scripts.build_co3d \
  --co3d_root raw_datasets/co3d \
  --out_root outputs \
  --categories apple,backpack \
  --num_points 1024 \
  --workers 4
```

`--categories` 未指定時は `configs/co3d_class_map.json` の 50 categories を対象にします。既定は deterministic sequence split で、`outputs/CO3D/split_manifest.json` に保存します。`--use_set_lists` 指定時のみ `set_lists` の利用を試みますが、CO3D の set_lists は frame-level split の可能性があるため、同一 sequence が train/test/val を跨ぐ場合は deterministic sequence split に fallback します。val は conflict がなければ test に merge します。

### S2C cross dataset 作成

```bash
python -m scripts.build_cross_shapenet_co3d \
  --shapenet_root outputs/ShapeNet \
  --co3d_root outputs/CO3D \
  --out_root outputs
```

出力は `outputs/shapenet_co3d/<class_id>/train|test/*.npy` です。session ディレクトリは作らず、session 情報は `configs/shapenet_co3d_sessions.json`、class map は `configs/shapenet_co3d_class_map.json` に固定しています。class_id は global id `0..88` を保持し、再マッピングしません。CO3D は category-wise に増えるため、`--strict` なしでは missing class を warning として続行し、`--strict` ありでは不足を error にします。

### 前処理済みデータの個別確認

```bash
python -m scripts.smoke_test_dataset --dataset modelnet --data_root outputs/ModelNet --split train
python -m scripts.smoke_test_dataset --dataset scanobjectnn --data_root outputs/ScanObjectNN --split train
python -m scripts.smoke_test_dataset --dataset m2s --data_root outputs/modelnet_scanobjectnn --split train
python -m scripts.smoke_test_dataset --dataset shapenet --data_root outputs/ShapeNet --split train
python -m scripts.smoke_test_dataset --dataset co3d --data_root outputs/CO3D --split train
python -m scripts.smoke_test_dataset --dataset s2c --data_root outputs/shapenet_co3d --session 1 --split train
```

Alias は `ShapeNet`, `shapenet`, `CO3D`, `co3d`, `shapenet_co3d`, `s2c`, `S2C` を受け付けます。ShapeNet と CO3D の single dataset は class-name based directory を読み、config の deterministic class order から label を決めます。S2C は class-id based cross dataset として読み、label は global class_id です。

### 出力検証

```bash
python -m scripts.validate_processed_dataset --root outputs/ShapeNet --mode single --num_points 1024
python -m scripts.validate_processed_dataset --root outputs/CO3D --mode single --num_points 1024
python -m scripts.validate_processed_dataset --root outputs/shapenet_co3d --mode cross --num_points 1024
```

検証では `.npy` の存在、`train`/`test` directory、shape、float 変換可能性、NaN/Inf、中心化・単位球正規化、cross mode の numeric class_id directory を確認します。

### S2C class_id / session 対応表

| session | dataset | class_id: class_name |
| --- | --- | --- |
| 1 | ShapeNet | 0: airplane, 1: trash_bin, 2: basket, 3: bathtub, 4: bed, 5: birdhouse, 6: bookshelf, 7: bus, 8: cabinet, 9: camera, 10: can, 11: cap, 12: clock, 13: dishwasher, 14: display, 15: faucet, 16: file_cabinet, 17: guitar, 18: helmet, 19: jar, 20: knife, 21: lamp, 22: loudspeaker, 23: mailbox, 24: microphone, 25: mug, 26: piano, 27: pillow, 28: pistol, 29: flowerpot, 30: printer, 31: rifle, 32: rocket, 33: stove, 34: table, 35: tower, 36: train, 37: vessel, 38: washer |
| 2 | CO3D | 39: apple, 40: backpack, 41: ball, 42: banana, 43: baseballbat |
| 3 | CO3D | 44: baseballglove, 45: bench, 46: bicycle, 47: book, 48: bottle |
| 4 | CO3D | 49: bowl, 50: broccoli, 51: cake, 52: car, 53: carrot |
| 5 | CO3D | 54: cellphone, 55: chair, 56: couch, 57: cup, 58: donut |
| 6 | CO3D | 59: frisbee, 60: hairdryer, 61: handbag, 62: hotdog, 63: hydrant |
| 7 | CO3D | 64: keyboard, 65: kite, 66: laptop, 67: microwave, 68: motorcycle |
| 8 | CO3D | 69: mouse, 70: orange, 71: parkingmeter, 72: pizza, 73: plant |
| 9 | CO3D | 74: remote, 75: sandwich, 76: skateboard, 77: stopsign |
| 10 | CO3D | 78: suitcase, 79: teddybear, 80: toaster, 81: toilet, 82: toybus, 83: toyplane |
| 11 | CO3D | 84: toytruck, 85: tv, 86: umbrella, 87: vase, 88: wineglass |
