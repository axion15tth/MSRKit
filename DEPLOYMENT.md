# デプロイメントガイド

別サーバーでの訓練開始までの完全な手順です。

## 前提条件

### ローカル環境（このサーバー）
- Git リポジトリが準備済み
- すべてのコードがcommit済み

### ターゲットサーバー
- Python 3.8+ インストール済み
- CUDA 11.0+ （GPU訓練の場合）
- Git インストール済み
- データセット配置済み:
  - MUSDB18-HQ: `/data/MUSDB18-HQ/`
  - MoisesDB: `/data/MoisesDB/`
  - RawStems: `/data/RawStems-48k/`

---

## Step 1: GitHub へのプッシュ（このサーバー）

### 1.1 コミット

```bash
cd /home/yamahamss/MSRKit

# 全ファイルがステージングされていることを確認
git status

# コミット
git commit -F .commit_message.txt

# コミット確認
git log -1
```

### 1.2 リモートリポジトリの設定

```bash
# リモートリポジトリが設定されていない場合
git remote add origin https://github.com/yourusername/MSRKit.git

# または SSH
git remote add origin git@github.com:yourusername/MSRKit.git

# 確認
git remote -v
```

### 1.3 プッシュ

```bash
# main ブランチにプッシュ
git push -u origin main

# または既存のブランチに
git push origin main
```

---

## Step 2: ターゲットサーバーでのセットアップ

### 2.1 リポジトリのクローン

```bash
# SSH接続でターゲットサーバーにログイン
ssh user@target-server

# 作業ディレクトリに移動
cd /workspace  # または任意のディレクトリ

# クローン
git clone https://github.com/axion15tth/MSRKit.git
cd MSRKit
```

### 2.2 環境構築

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install --upgrade pip
pip install -r requirements.txt

# インストール確認
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch_lightning as pl; print(f'Lightning: {pl.__version__}')"
```

### 2.3 データセットのセットアップ

**ワンコマンドでセットアップ:**

```bash
./setup_datasets.sh /data
```

これで以下が自動実行されます：
- シンボリックリンク作成
- FileList生成（train/val分割）
- データペア数の表示

**実行例:**

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
```

### 2.4 設定の確認

```bash
# config.yaml の確認（必要に応じて編集）
cat config.yaml

# FileList の確認
head -5 lists/train_vocals.txt
wc -l lists/train_vocals.txt lists/val_vocals.txt
```

### 2.5 動作テスト

```bash
# インポートテスト
python -c "
from models.htdemucs_cunet import HTDemucsCUNetGenerator
from data.degradations import forward_degrade
print('✓ All imports successful')
"

# モデル初期化テスト
python -c "
import yaml
import torch
from models.htdemucs_cunet import HTDemucsCUNetGenerator

with open('config.yaml') as f:
    config = yaml.safe_load(f)

model = HTDemucsCUNetGenerator(config)
print(f'✓ Model initialized: {sum(p.numel() for p in model.parameters()):,} params')
"
```

---

## Step 3: 訓練の開始

### 3.1 TensorBoard の起動（別ターミナル）

```bash
# SSH接続（別セッション）
ssh user@target-server
cd /workspace/MSRKit
source venv/bin/activate

# TensorBoard起動
tensorboard --logdir runs --bind_all --port 6006
```

ローカルPCから確認:
```bash
# SSHポートフォワーディング
ssh -L 6006:localhost:6006 user@target-server
```

ブラウザで `http://localhost:6006` を開く

### 3.2 訓練の開始

```bash
# メインターミナルで
python train.py --config config.yaml
```

**オプション: Weights & Biases (WandB) を使用する場合**

WandBで実験を管理する場合：

```bash
# WandBにログイン（初回のみ）
wandb login

# config.yaml を編集してWandBを有効化
# logging.wandb.enabled を true に設定

# 訓練開始
python train.py --config config.yaml
```

WandBを使用すると、ブラウザから以下を確認できます：
- リアルタイムのloss curves
- System metrics (GPU, CPU, メモリ使用率)
- ハイパーパラメータの比較
- 実験ノートと共有

### 3.3 バックグラウンド実行（推奨）

```bash
# tmux または screen を使用
tmux new -s msrkit
python train.py --config config.yaml

# デタッチ: Ctrl+b, d
# 再アタッチ: tmux attach -t msrkit

# または nohup
nohup python train.py --config config.yaml > train.log 2>&1 &
tail -f train.log
```

---

## Step 4: 訓練の監視

### 4.1 TensorBoard で確認

- Loss curves
- Learning rate
- Generated samples（実装されている場合）

### 4.2 ログファイルで確認

```bash
# 訓練ログ
tail -f train.log  # nohup使用の場合

# チェックポイント確認
ls -lh runs/msrkit/HTDemucsCUNetGenerator/checkpoints/
```

### 4.3 GPU使用状況の監視

```bash
# 別ターミナルで
watch -n 1 nvidia-smi
```

---

## Step 5: トラブルシューティング

### CUDA Out of Memory

config.yaml を編集:
```yaml
trainer:
  batch_size: 4  # 8 → 4 に減らす
```

### データセットが見つからない

```bash
# シンボリックリンクを確認
ls -la data/

# FileList を確認
head lists/train_vocals.txt

# 手動で再生成
python tools/build_filelists.py \
  --roots data/MUSDB18-HQ data/MoisesDB-48k data/RawStems-48k \
  --out_dir lists
```

### 依存関係エラー

```bash
# 再インストール
pip install --upgrade --force-reinstall -r requirements.txt

# または個別にインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning
pip install -r requirements.txt
```

### FFmpeg not found（コーデック劣化用）

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# または config.yaml で DT2 を無効化
# （setup_datasets.sh 実行後に自動で調整される）
```

---

## Step 6: 推論とモデル評価

### チェックポイントからの推論

```bash
python inference.py \
  --checkpoint runs/msrkit/HTDemucsCUNetGenerator/checkpoints/step_100000.ckpt \
  --input /path/to/test.wav \
  --output output_vocals.wav
```

### バッチ推論

```bash
# 複数ファイルに対して推論
for file in /path/to/test/*.wav; do
  python inference.py \
    --checkpoint runs/.../step_100000.ckpt \
    --input "$file" \
    --output "outputs/$(basename $file)"
done
```

---

## Quick Reference

### ワンライナー（フルセットアップ）

```bash
# ターゲットサーバーで（データセットが /data 配下にある場合）
git clone https://github.com/axion15tth/MSRKit.git && \
cd MSRKit && \
python -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
./setup_datasets.sh /data && \
python train.py --config config.yaml
```

または、データセットが個別の場所にある場合：

```bash
git clone https://github.com/axion15tth/MSRKit.git && \
cd MSRKit && \
python -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
./setup_datasets.sh --musdb /path/to/MUSDB18 --moisesdb /path/to/MoisesDB --rawstems /path/to/RawStems && \
python train.py --config config.yaml
```

### ディレクトリ構造確認

```bash
tree -L 2 -I '__pycache__|*.pyc|venv|runs'
```

---

## 完了！

これで別サーバーで訓練が開始されました。

**参考ドキュメント:**
- [README.md](README.md) - プロジェクト全体の説明
- [QUICKSTART.md](QUICKSTART.md) - クイックスタート
- [docs/DATASET_PREPARATION.md](docs/DATASET_PREPARATION.md) - データセット詳細
