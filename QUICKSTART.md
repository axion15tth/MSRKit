# クイックスタートガイド

すでにデータセットが準備されている別サーバーで、すぐに訓練を開始する手順です。

## 前提条件

- Python 3.8+
- CUDA 11.0+ （GPU訓練の場合）
- データセット: MUSDB18-HQ, MoisesDB, RawStems-48k が `/data` 等に配置済み

## 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/MSRKit.git
cd MSRKit
```

## 2. 環境構築

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt
```

## 3. データセットのセットアップ

データセットが既にある場合、**1コマンド**でセットアップ完了：

```bash
./setup_datasets.sh /data
```

このスクリプトが自動で実行すること：
- `/data/MUSDB18-HQ` → `data/MUSDB18-HQ` へのシンボリックリンク作成
- `/data/MoisesDB` → `data/MoisesDB-48k` へのシンボリックリンク作成
- `/data/RawStems-48k` → `data/RawStems-48k` へのシンボリックリンク作成
- `mixture.wav` と `vocals.wav` のペアを自動検出
- `lists/train_vocals.txt` と `lists/val_vocals.txt` を生成（train/val分割）

### 実行例

```bash
$ ./setup_datasets.sh /data

=== MSRKit Dataset Setup ===

Datasets parent directory: /data

Searching for datasets...

✓ Found MUSDB18-HQ at: /data/MUSDB18-HQ
  → Created symlink: data/MUSDB18-HQ
✓ Found MoisesDB at: /data/MoisesDB
  → Created symlink: data/MoisesDB-48k
✓ Found RawStems at: /data/RawStems-48k
  → Created symlink: data/RawStems-48k

================================
Found 3 dataset(s)

Generating train/val file lists...
pairs: 485 (train=436, val=49) -> lists

✓ File lists generated successfully
  Train: 436 samples
  Val:   49 samples

================================
Setup complete!

Next steps:
  1. Review config.yaml (data paths should be auto-detected)
  2. Start training:
     python train.py --config config.yaml
```

## 4. 訓練の開始

```bash
python train.py --config config.yaml
```

### 訓練の監視

```bash
# 別のターミナルで TensorBoard を起動
tensorboard --logdir runs

# ブラウザで http://localhost:6006 を開く
```

## 5. 推論

```bash
python inference.py \
  --checkpoint runs/msrkit/HTDemucsCUNetGenerator/checkpoints/step_100000.ckpt \
  --input input.wav \
  --output output_vocals.wav
```

---

## トラブルシューティング

### データセットが見つからない

```bash
# データセットのパスを確認
ls /data
# MUSDB18-HQ, MoisesDB, RawStems-48k が存在することを確認

# または異なるパスの場合
./setup_datasets.sh /path/to/your/data
```

### CUDA Out of Memory

config.yamlでバッチサイズを減らす：

```yaml
trainer:
  batch_size: 4  # デフォルトは8
```

### 依存関係エラー

```bash
# すべて再インストール
pip install --upgrade -r requirements.txt
```

---

## config.yaml の主要設定

```yaml
# データ（setup_datasets.sh で自動設定）
data:
  train_file_list: "lists/train_vocals.txt"
  batch_size: 8
  num_workers: 8

# GPU設定
trainer:
  devices: [0]          # 使用するGPU ID（複数指定可: [0,1,2,3]）
  precision: "bf16-mixed"  # 混合精度訓練
  max_steps: 400000

# 損失関数の重み
loss:
  sisnr: 1.0
  l1_wave: 1.0
  mrstft: 0.5
  mixture_reconstruct: 0.5
  gan: 0.05
```

---

## 完了！

これで訓練を開始できます。詳細は [README.md](README.md) と [docs/DATASET_PREPARATION.md](docs/DATASET_PREPARATION.md) を参照してください。
