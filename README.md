# MSRKit - Music Source Restoration with HTDemucs-CUNet

HTDemucs-lite + Complex U-Net による高品質な音楽ソース復元（ボーカル分離）の実装。劣化した音源からクリーンなボーカルステムを抽出します。

## 特徴

- **2段階アーキテクチャ**: HTDemucs-lite (Stage 1) + Complex U-Net Restorer (Stage 2)
- **劣化推定器**: 11の音響特徴から劣化パラメータ (α, direction) を自動推定
- **劣化プロファイル**: スマホ録音、ストリーミングコーデック、重劣化の3種類に対応
- **オンザフライ劣化**: 訓練時に自動で劣化を適用し、実環境に近い学習
- **GAN訓練対応**: Discriminatorによる品質向上
- **マルチデータセット**: MUSDB18, MoisesDB, 自前データを統合可能

## アーキテクチャ

```
Input: wave_mix (B, 2, T)
    ↓
[DegEstimator] → α̂ (strength), d̂ (direction)
    ↓
[Stage 1: HTDemucs-lite] → ŝ_stem (coarse separation)
    ↓
[Stage 2: Complex U-Net] → restored stem (B, 2, T)
```

- **Stage 1**: 時間領域U-Net（28M params）
- **Stage 2**: STFT領域Complex U-Net（86M params）
- **Total**: ~114M parameters

## クイックスタート

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/MSRKit.git
cd MSRKit
```

### 2. 環境構築

```bash
# Python 3.8+ 推奨
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

### 3. データセットの配置

既存のデータセットがある場合、`setup_datasets.sh` を使用：

```bash
# データセットの親ディレクトリを指定
./setup_datasets.sh /path/to/your/datasets

# 例: /data 配下に MUSDB18-HQ/, MoisesDB/, RawStems-48k/ がある場合
./setup_datasets.sh /data
```

このスクリプトは以下を実行：
- データディレクトリへのシンボリックリンク作成
- 自動的に mixture.wav / vocals.wav ペアを検索
- train/val 分割したFileListを生成

### 4. 手動でのデータセット準備（初めての場合）

詳細は [DATASET_PREPARATION.md](DATASET_PREPARATION.md) を参照。

```bash
# MUSDB18-HQ
python tools/resample_to_48k.py --in_root /data/MUSDB18-HQ --out_root data/MUSDB18-48k

# MoisesDB
python tools/export_moisesdb_wavs.py --moises_root /data/moisesdb --out_root data/MoisesDB-48k

# RawStems（自前データ）
python tools/resample_to_48k.py --in_root /data/RawStems --out_root data/RawStems-48k

# FileList生成
python tools/build_filelists.py \
  --roots data/MUSDB18-48k data/MoisesDB-48k data/RawStems-48k \
  --out_dir lists
```

### 5. 訓練の開始

```bash
python train.py --config config.yaml
```

訓練ログは `runs/msrkit/HTDemucsCUNetGenerator/` に保存されます。

```bash
# TensorBoard で確認
tensorboard --logdir runs
```

## 設定

### config.yaml の主要パラメータ

```yaml
# データパス（自動設定される場合は不要）
data:
  train_file_list: "lists/train_vocals.txt"  # setup_datasets.sh で生成
  
# モデル
model:
  name: "HTDemucsCUNetGenerator"
  target: "vocals"  # 分離対象のステム
  
# 訓練
trainer:
  max_steps: 400000
  batch_size: 8
  devices: [0]
  precision: "bf16-mixed"
  
# 損失関数
loss:
  sisnr: 1.0          # SI-SNR損失
  l1_wave: 1.0        # L1損失
  mrstft: 0.5         # Multi-Resolution STFT損失
  mixture_reconstruct: 0.5  # ミックス再構成損失
  gan: 0.05           # GAN損失
```

### 環境変数での設定

```bash
# データセットパスを環境変数で指定することも可能
export MUSDB_PATH=/data/MUSDB18-48k
export MOISESDB_PATH=/data/MoisesDB-48k
export RAWSTEMS_PATH=/data/RawStems-48k

python train.py --config config.yaml
```

## 推論

```bash
python inference.py \
  --checkpoint runs/msrkit/HTDemucsCUNetGenerator/checkpoints/step_100000.ckpt \
  --input input.wav \
  --output output.wav
```

## データセット

サポートされているデータセット：

- **MUSDB18 / MUSDB18-HQ**: 150曲、4ステム（vocals, bass, drums, other）
- **MoisesDB**: 240曲、階層化ステム
- **RawStems**: 自前の高品質ステムコレクション

### ディレクトリ構造

```
data/
├── MUSDB18-48k/
│   ├── train/
│   │   ├── song1/
│   │   │   ├── mixture.wav
│   │   │   └── vocals.wav
│   │   └── ...
│   └── test/
├── MoisesDB-48k/
└── RawStems-48k/

lists/
├── train_vocals.txt  # mix_path|target_path のリスト
└── val_vocals.txt
```

## 劣化プロファイル

訓練時に自動適用される劣化（オンザフライ）：

| Profile | 想定シーン | 処理内容 |
|---------|----------|---------|
| **DT1** | スマホ/環境録音 | HPF, LPF, ダウンサンプリング, コンプ, リバーブ, ノイズ |
| **DT2** | ストリーミング/コーデック | EQ傾斜, AAC/OPUS/MP3往復, サチュレーション |
| **DT3** | ブートレグ/重劣化 | 帯域制限, 強コンプ, 長リバーブ, ノイズ, クリップ |

実装: `data/degradations.py`

## プロジェクト構成

```
MSRKit/
├── config.yaml              # 訓練設定
├── train.py                 # 訓練スクリプト
├── inference.py             # 推論スクリプト
├── requirements.txt         # 依存関係
├── setup_datasets.sh        # データセット自動設定
│
├── models/
│   ├── htdmucs_cunet.py    # メインモデル
│   └── __init__.py
│
├── modules/
│   ├── condition/
│   │   └── deg_estimator.py       # 劣化推定器
│   ├── generator/
│   │   ├── htdemucs.py           # Stage 1: HTDemucs-lite
│   │   ├── complex_unet.py       # Stage 2: Complex U-Net
│   │   └── complex_ops.py        # 複素数演算
│   └── discriminator/
│
├── losses/
│   ├── sisnr.py            # SI-SNR損失
│   ├── mrstft.py           # Multi-Resolution STFT損失
│   ├── complex_loss.py     # Complex損失
│   ├── consistency.py      # Mixture再構成損失
│   ├── fad_clap_approx.py  # CLAP/FAD損失
│   └── gan_loss.py         # GAN損失
│
├── data/
│   ├── dataset.py          # データローダー
│   ├── degradations.py     # 劣化プロファイル
│   ├── augment.py          # データ拡張
│   └── stft.py             # STFT処理
│
├── tools/
│   ├── export_moisesdb_wavs.py    # MoisesDB変換
│   ├── resample_to_48k.py         # リサンプル
│   ├── build_filelists.py         # FileList生成
│   └── render_degraded_corpus.py  # オフライン劣化生成
│
└── docs/
    └── DATASET_PREPARATION.md     # データセット準備詳細
```

## トラブルシューティング

### CUDA Out of Memory

```yaml
# config.yaml で batch_size を削減
trainer:
  batch_size: 4  # 8 → 4
```

### 依存関係エラー

```bash
# pedalboard（データ拡張用）
pip install pedalboard

# laion-clap（CLAP/FAD損失用、オプション）
pip install laion-clap
# または config.yaml で無効化
loss:
  clap_embed: 0.0
  fad_proxy: 0.0
```

### FFmpeg not found

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# または DT2 プロファイル（コーデック劣化）を無効化
```

## パフォーマンス

NVIDIA A100 40GB での参考値：
- **Batch size 8**: ~15 sec/step
- **メモリ使用量**: ~20 GB
- **訓練時間**: 400k steps ≈ 7 days

## 引用

このコードを使用する場合は、以下を引用してください：

```bibtex
@software{msrkit2025,
  title={MSRKit: Music Source Restoration with HTDemucs-CUNet},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/MSRKit}
}
```

## ライセンス

MIT License

## 謝辞

- **HTDemucs**: [facebookresearch/demucs](https://github.com/facebookresearch/demucs)
- **MUSDB18**: [sigsep/sigsep-mus-db](https://github.com/sigsep/sigsep-mus-db)
- **MoisesDB**: [moises-ai/moises-db](https://github.com/moises-ai/moises-db)

## サポート

Issue: https://github.com/yourusername/MSRKit/issues
