# データセット準備ガイド

HTDemucs-CUNet訓練用のデータセット準備手順です。

## 概要

3系統のデータソースを統合し、劣化プロファイルを適用して訓練します：

1. **RawStems**: 自前の高品質ステムコレクション
2. **MUSDB18(HQ)**: 150曲・4ステム（vocals, bass, drums, other）
3. **MoisesDB**: 240曲・階層化ステム

すべて **48 kHz / ステレオ / 10秒チャンク** に統一して使用します。

## 劣化プロファイル

| プロファイル | 想定シーン | 処理内容 |
|------------|----------|---------|
| **DT1** | スマホ/環境録音 | HPF→LPF→ダウンサンプリング→コンプ→リバーブ→ノイズ |
| **DT2** | ストリーミング/コーデック | EQ傾斜→コーデック往復→サチュレーション |
| **DT3** | ブートレグ/重劣化 | LPF+HPF→強コンプ→長リバーブ→ノイズ→クリップ |

訓練時は**オンザフライ**で劣化を適用（`data/degradations.py`）。

---

## 準備手順

### 1. 依存関係のインストール

```bash
cd /home/yamahamss/MSRKit
source /home/yamahamss/msrkit_env/bin/activate

# 基本依存
pip install -r requirements.txt

# データセット用
pip install musdb soundfile
pip install git+https://github.com/moises-ai/moises-db.git
```

### 2. MUSDB18(HQ) の準備

```bash
# ダウンロード（Zenodo から取得）
# https://zenodo.org/record/3338373

# STEMSフォーマットの場合、WAVに変換
musdbconvert /data/MUSDB18-STEMS /data/MUSDB18-WAV

# 48 kHz にリサンプル
python tools/resample_to_48k.py \
  --in_root /data/MUSDB18-WAV \
  --out_root /data/MUSDB18-48k
```

### 3. MoisesDB の準備

```bash
# ダウンロード（研究サイトから取得）
# https://github.com/moises-ai/moises-db

# 環境変数設定
export MOISESDB_PATH=/data/moisesdb

# 48 kHz WAV に書き出し
python tools/export_moisesdb_wavs.py \
  --moises_root /data/moisesdb \
  --out_root /data/MoisesDB-48k
```

### 4. RawStems の準備

```bash
# 自前のステムコレクションを 48 kHz に統一
python tools/resample_to_48k.py \
  --in_root /data/RawStems \
  --out_root /data/RawStems-48k
```

各曲ディレクトリに以下のファイルが必要：
```
song_dir/
  ├── mixture.wav  (48kHz, stereo)
  └── vocals.wav   (48kHz, stereo)
```

### 5. FileList の生成

```bash
# 3コーパスを統合して train/val 分割
python tools/build_filelists.py \
  --roots /data/MUSDB18-48k /data/MoisesDB-48k /data/RawStems-48k \
  --out_dir lists \
  --val_ratio 0.1 \
  --seed 1337
```

生成されるファイル：
- `lists/train_vocals.txt` (各行: `mix_path|target_path`)
- `lists/val_vocals.txt`

### 6. config.yaml の更新

```yaml
data:
  sample_rate: 48000
  clip_duration: 10.0
  train_file_list: "lists/train_vocals.txt"  # FileList方式
  # または RawStems方式:
  # train_dataset:
  #   target_stem: "Voc"
  #   root_directory: "/data/RawStems-48k"
  dataloader_params:
    batch_size: 8
    num_workers: 8
```

---

## 訓練方法

### オンザフライ劣化（推奨）

訓練中に自動で劣化適用（`train.py`は既に対応済み）：

```bash
python train.py --config config.yaml
```

- 入力：`mixture_clean` → 劣化適用 → `mixture_in`
- 教師：`target` (クリーンボーカル)
- MixtureReconstructLoss: 同じ劣化presetで一致を取る

### オフライン劣化（オプション）

事前に劣化ミックスを生成：

```bash
python tools/render_degraded_corpus.py \
  --filelist lists/train_vocals.txt \
  --out_root data/deg_dtall \
  --profile dtall  # dt1, dt2, dt3, または dtall (ランダム)
```

生成：
- `data/deg_dtall/XXXXXXX/mixture_deg.wav`
- `data/deg_dtall/train_vocals_deg.txt`

config.yamlで指定：
```yaml
data:
  train_file_list: "data/deg_dtall/train_vocals_deg.txt"
```

---

## スクリプト一覧

| スクリプト | 用途 |
|----------|------|
| `tools/export_moisesdb_wavs.py` | MoisesDB → 48kHz WAV |
| `tools/resample_to_48k.py` | 一括リサンプル |
| `tools/build_filelists.py` | FileList生成 |
| `data/degradations.py` | 劣化チェーン（DT1/2/3） |
| `tools/render_degraded_corpus.py` | オフライン下焼き |

---

## 劣化の詳細

### DT1: スマホ/環境録音
- HPF: 80-150 Hz
- LPF: 8-12 kHz
- ダウンサンプリング: 24k/32k → 48k復帰
- コンプレッサ: ratio 3-6
- リバーブ: RT60 0.3-0.8s
- ノイズ: SNR 15-30 dB

### DT2: ストリーミング/コーデック
- EQ傾斜: ±3-6 dB
- コーデック: AAC/OPUS/MP3 @ 64-128 kbps
- サチュレーション: drive 0-6 dB

### DT3: ブートレグ/重劣化
- HPF: 120-180 Hz
- LPF: 6-10 kHz
- 強コンプ: ratio 6-10
- 長リバーブ: RT60 0.8-1.5s
- ノイズ: SNR 10-25 dB
- サチュレーション: drive 3-9 dB

---

## RawStems スクリーニング（推奨）

品質フィルタ：
- クリップ率 >1% → 除外
- 無音率 >30% (RMS < -60 dBFS) → 除外
- ステム/ミックス オフセット >±500ms → 除外
- L/R バランス差 >6 dB → 警告

スクリーニングツールは別途提供可能です。

---

## 参考リンク

- [MUSDB18](https://sigsep.github.io/datasets/musdb.html) - 150曲・4ステム
- [MUSDB18-HQ (Zenodo)](https://zenodo.org/record/3338373) - 非圧縮版
- [MoisesDB](https://github.com/moises-ai/moises-db) - 240曲・階層化ステム

---

## トラブルシューティング

### `FileListDataset not found`
→ `data/datasets.py` に `FileListDataset` を実装するか、`RawStems` を使用

### `laion-clap import error`
→ config.yamlで `clap_embed: 0.0`, `fad_proxy: 0.0` に設定

### `pedalboard import error`
→ `pip install pedalboard`

### FFmpeg not found (コーデック劣化)
→ `sudo apt install ffmpeg` または劣化プロファイルをDT1/DT3のみに制限
