# Pointcloud Cross-Dataset Builder

ModelNet40, ScanObjectNN, ShapeNet, CO3D を共通の `.npy` 形式へ前処理し、必要なら cross dataset も作るためのリポジトリです。

## 何ができるか

- ModelNet40 -> `outputs/ModelNet/<class_name>/train|test/*.npy`
- ScanObjectNN -> `outputs/ScanObjectNN/<class_name>/train|test/*.npy`
- ModelNet + ScanObjectNN cross -> `outputs/modelnet_scanobjectnn/<class_id>/train|test/*.npy`
- ShapeNet -> `outputs/ShapeNet/<class_name>/train|test/*.npy`
- CO3D -> `outputs/CO3D/<category>/train|test/*.npy`
- ShapeNet + CO3D cross -> `outputs/shapenet_co3d/<class_id>/train|test/*.npy`

各 sample は原則 1 ファイルずつ保存され、隣に `.json` metadata が付きます。

## 共通仕様

- 出力点数は原則 `(1024, 3)`
- dtype は `float32`
- NaN / Inf なし
- 中心化・正規化済み
- 保存単位は `1 sample = 1 .npy + 1 .json`

## 1. 環境準備

```bash
conda create -n pc_builder python=3.9
conda activate pc_builder
pip install -r requirements.txt
```

CO3D を扱う場合は公式リポジトリを別途 clone します。

```bash
mkdir -p external
git clone https://github.com/facebookresearch/co3d.git external/co3d
```

## 2. 入力データ配置

### ModelNet40

```text
raw_datasets/modelnet40/<class_name>/train|test/*.off
```

### ScanObjectNN

```text
raw_datasets/scanobjectnn/h5_files/main_split_nobg/
  training_objectdataset.h5
  test_objectdataset.h5
```

### ShapeNet

ShapeNetCore 形式を含む任意の raw 配置を受け付けます。raw folder 名の揺れは `configs/shapenet_class_map.json` で吸収します。

### CO3D

```text
raw_datasets/co3d/<category>/<sequence_name>/pointcloud.ply
```

## 3. ModelNet40 を作る

```bash
python -m scripts.build_modelnet \
  --modelnet_root raw_datasets/modelnet40 \
  --out_root outputs
```

主なオプション:

- `--sample_surface_n 10000`
- `--fps_k 1024`
- `--workers 4`
- `--sample_from_mesh` / `--no_sample_from_mesh`

出力:

```text
outputs/ModelNet/<class_name>/train|test/*.npy
outputs/ModelNet/<class_name>/train|test/*.json
```

## 4. ScanObjectNN を作る

```bash
python -m scripts.build_scanobjectnn \
  --h5_root raw_datasets/scanobjectnn/h5_files \
  --out_root outputs
```

主なオプション:

- `--split_dir main_split_nobg`
- `--train_h5 training_objectdataset.h5`
- `--test_h5 test_objectdataset.h5`
- `--num_points 1024`

出力:

```text
outputs/ScanObjectNN/<class_name>/train|test/*.npy
outputs/ScanObjectNN/<class_name>/train|test/*.json
```

## 5. ModelNet + ScanObjectNN cross を作る

```bash
python -m scripts.build_cross_sessions \
  --modelnet_root outputs/ModelNet \
  --scanobjectnn_root outputs/ScanObjectNN \
  --out_root outputs
```

出力:

```text
outputs/modelnet_scanobjectnn/<class_id>/train|test/*.npy
outputs/modelnet_scanobjectnn/<class_id>/train|test/*.json
```

class_id 対応は `configs/sessions.json` と既存実装に従います。

## 6. ShapeNet を作る

```bash
python -m scripts.build_shapenet \
  --shapenet_root /path/to/ShapeNet \
  --out_root outputs \
  --num_points 1024 \
  --workers 4
```

主なオプション:

- `--sample_surface_n 10000`
- `--train_ratio 0.8`
- `--seed 42`
- `--skip_existing`
- `--overwrite`

出力:

```text
outputs/ShapeNet/<class_name>/train|test/*.npy
outputs/ShapeNet/<class_name>/train|test/*.json
outputs/ShapeNet/split_manifest.json
```

補足:

- `class_name/train|test` が raw にある場合はその split を使います
- split がない場合は class ごとに deterministic split を作ります
- raw folder 名の揺れは `configs/shapenet_class_map.json` の `aliases` と `synset_id` で吸収します

## 7. CO3D を category-wise に作る

CO3D は巨大なので、category ごとに download -> preprocess -> raw cleanup を回します。

### 全カテゴリ

```bash
python -m scripts.download_and_build_co3d_category \
  --co3d_repo external/co3d \
  --download_folder raw_datasets/co3d \
  --out_root outputs \
  --all_categories \
  --num_points 1024 \
  --workers 4 \
  --cleanup_raw
```

### 単一カテゴリ

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

### 複数カテゴリ

```bash
python -m scripts.download_and_build_co3d_category \
  --co3d_repo external/co3d \
  --download_folder raw_datasets/co3d \
  --out_root outputs \
  --categories apple,backpack \
  --num_points 1024 \
  --workers 4 \
  --cleanup_raw
```

### raw を残す

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

この wrapper の既定動作:

- 対象は `--category` / `--categories` / `--all_categories` のどれか 1 つ
- `download_dataset.py --help` を見て、category 指定可能な downloader だけを使う
- category 指定不能なら一括 download はしない
- preprocess 成功後のみ raw subset を削除する
- preprocess 失敗時は raw を残す
- `--force_cleanup_on_error` があるときだけ失敗時 cleanup を許可する
- `outputs/CO3D/<category>` は削除しない

主な CLI オプション:

- `--co3d_repo external/co3d`
- `--download_folder raw_datasets/co3d`
- `--out_root outputs`
- `--category apple`
- `--categories apple,backpack`
- `--all_categories`
- `--num_points 1024`
- `--workers 4`
- `--seed 42`
- `--skip_existing` / `--overwrite`
- `--download_only` / `--process_only`
- `--cleanup_raw` / `--no_cleanup_raw`
- `--force_cleanup_on_error`
- `--single_sequence_subset` / `--no_single_sequence_subset`
- `--strict`

出力:

```text
outputs/CO3D/<category>/train|test/*.npy
outputs/CO3D/<category>/train|test/*.json
outputs/CO3D/split_manifest.json
```

## 8. CO3D 前処理だけを回す

raw が既にある場合はこちらです。

```bash
python -m scripts.build_co3d \
  --co3d_root raw_datasets/co3d \
  --out_root outputs \
  --categories apple,backpack \
  --num_points 1024 \
  --workers 4
```

`--categories` を省略すると `configs/co3d_class_map.json` の全 50 category を使います。

主な CLI オプション:

- `--co3d_root raw_datasets/co3d`
- `--out_root outputs`
- `--categories apple,backpack`
- `--num_points 1024`
- `--train_ratio 0.8`
- `--workers 4`
- `--seed 42`
- `--use_set_lists`
- `--skip_existing`
- `--overwrite`
- `--strict`

出力:

```text
outputs/CO3D/<category>/train|test/*.npy
outputs/CO3D/<category>/train|test/*.json
outputs/CO3D/split_manifest.json
```

## 9. ShapeNet + CO3D cross を作る

```bash
python -m scripts.build_cross_shapenet_co3d \
  --shapenet_root outputs/ShapeNet \
  --co3d_root outputs/CO3D \
  --out_root outputs
```

出力:

```text
outputs/shapenet_co3d/<class_id>/train|test/*.npy
outputs/shapenet_co3d/<class_id>/train|test/*.json
```

class_id は global `0..88` のまま保持します。session 情報は `configs/shapenet_co3d_sessions.json` にあります。

## 10. 生成後の確認

### smoke test

```bash
python -m scripts.smoke_test_dataset --dataset modelnet --data_root outputs/ModelNet --split train
python -m scripts.smoke_test_dataset --dataset scanobjectnn --data_root outputs/ScanObjectNN --split train
python -m scripts.smoke_test_dataset --dataset m2s --data_root outputs/modelnet_scanobjectnn --split train
python -m scripts.smoke_test_dataset --dataset shapenet --data_root outputs/ShapeNet --split train
python -m scripts.smoke_test_dataset --dataset co3d --data_root outputs/CO3D --split train
python -m scripts.smoke_test_dataset --dataset s2c --data_root outputs/shapenet_co3d --session 1 --split train
```

表示される内容:

- dataset 名
- split
- session
- class 数
- sample 数
- 最初の sample shape
- 最初の label
- 最初の metadata

### validate

```bash
python -m scripts.validate_processed_dataset --root outputs/ShapeNet --mode single --num_points 1024
python -m scripts.validate_processed_dataset --root outputs/CO3D --mode single --num_points 1024
python -m scripts.validate_processed_dataset --root outputs/shapenet_co3d --mode cross --num_points 1024
```

確認内容:

- `.npy` が存在する
- `train` / `test` ディレクトリが存在する
- shape が `(num_points, 3)`
- dtype が float に変換可能
- NaN / Inf がない
- おおむね中心化・正規化されている
- cross mode では class_id ディレクトリが数字

## 11. class / session 対応

### ShapeNet

`configs/shapenet_class_map.json`

### CO3D

`configs/co3d_class_map.json`

### S2C

- class map: `configs/shapenet_co3d_class_map.json`
- session map: `configs/shapenet_co3d_sessions.json`

## 12. 重要な注意

- `raw_datasets/` は git 管理しません
- `outputs/` は git 管理しません
- `external/co3d/` は git 管理しません
- `*.zip` と `*_in_progress/` は git 管理しません
- CO3D は一括 download せず category-wise に処理します
- ShapeNet の raw folder 名が違う場合は `configs/shapenet_class_map.json` を調整してください

## 13. いちばん安全な実行順

1. `pip install -r requirements.txt`
2. `git clone https://github.com/facebookresearch/co3d.git external/co3d`
3. raw データを配置する
4. `build_modelnet.py`
5. `build_scanobjectnn.py`
6. `build_cross_sessions.py`
7. `build_shapenet.py`
8. `download_and_build_co3d_category.py`
9. `build_co3d.py` か wrapper で CO3D を追加処理
10. `build_cross_shapenet_co3d.py`
11. `smoke_test_dataset.py`
12. `validate_processed_dataset.py`
