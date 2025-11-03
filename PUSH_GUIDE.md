# GitHub プッシュガイド

## 現在の状態

✅ すべてのファイルが準備完了
✅ Git commit 準備完了
✅ ドキュメント整備完了

## プッシュ手順

### 1. コミット

```bash
cd /home/yamahamss/MSRKit
git commit -F .commit_message.txt
```

### 2. GitHubリポジトリの作成

ブラウザで GitHub にアクセス:
1. https://github.com/new
2. リポジトリ名: `MSRKit`
3. Description: `HTDemucs-CUNet: Music Source Restoration with Degradation Profiles`
4. Public または Private を選択
5. **Do not** initialize with README (既にローカルにあるため)
6. `Create repository` をクリック

### 3. リモートリポジトリの設定

```bash
# HTTPS の場合
git remote add origin https://github.com/yourusername/MSRKit.git

# SSH の場合（推奨）
git remote add origin git@github.com:yourusername/MSRKit.git

# 確認
git remote -v
```

### 4. プッシュ

```bash
git push -u origin main
```

## 別サーバーでのクローン

```bash
# ターゲットサーバーで
git clone https://github.com/yourusername/MSRKit.git
cd MSRKit

# 環境構築
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# データセットセットアップ（1コマンド）
./setup_datasets.sh /data

# 訓練開始
python train.py --config config.yaml
```

## 詳細手順

詳しくは以下を参照:
- [DEPLOYMENT.md](DEPLOYMENT.md) - 完全なデプロイメントガイド
- [QUICKSTART.md](QUICKSTART.md) - クイックスタート
- [README.md](README.md) - プロジェクト全体の説明
