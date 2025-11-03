# GitHub プッシュ手順

## 現在の状況

✅ コミット完了: `0d3a104` (34 files changed, 6503 insertions)
✅ リモート設定: `https://github.com/axion15tth/MSRKit.git`
⏳ プッシュ: 認証が必要です

## プッシュ方法

### Option 1: Personal Access Token (PAT) を使用（推奨）

#### 1. GitHub で Personal Access Token を作成

1. https://github.com/settings/tokens にアクセス
2. "Generate new token" → "Generate new token (classic)" をクリック
3. Note: `MSRKit deployment`
4. Expiration: 90 days（または任意）
5. Scopes: `repo` をチェック
6. "Generate token" をクリック
7. **トークンをコピー**（このページを閉じると二度と見られません）

#### 2. トークンを使ってプッシュ

```bash
cd /home/yamahamss/MSRKit

# トークンを使ってプッシュ（1回のみ）
git push -u origin main

# Username: axion15tth
# Password: <コピーしたトークンを貼り付け>
```

または、URLに直接トークンを含める（一時的）:

```bash
git remote set-url origin https://YOUR_TOKEN@github.com/axion15tth/MSRKit.git
git push -u origin main
```

### Option 2: SSH を使用（よりセキュア）

#### 1. SSH鍵が既にある場合

```bash
# リモートURLをSSHに変更
git remote set-url origin git@github.com:axion15tth/MSRKit.git

# プッシュ
git push -u origin main
```

#### 2. SSH鍵がない場合

```bash
# SSH鍵を生成
ssh-keygen -t ed25519 -C "axion.muramatsu@gmail.com"
# Enter を3回押す（パスフレーズなし）

# 公開鍵を表示してコピー
cat ~/.ssh/id_ed25519.pub

# GitHubに追加:
# https://github.com/settings/keys → "New SSH key"
# Title: "MSRKit Server"
# Key: コピーした公開鍵を貼り付け

# リモートURLをSSHに変更
git remote set-url origin git@github.com:axion15tth/MSRKit.git

# プッシュ
git push -u origin main
```

### Option 3: GitHub CLI を使用

```bash
# GitHub CLI のインストール（Debian/Ubuntu）
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# 認証
gh auth login
# → GitHub.com
# → HTTPS
# → Yes (authenticate)
# → Login with a web browser

# プッシュ
git push -u origin main
```

## プッシュ確認

```bash
# プッシュ成功の確認
git log -1 --oneline
# 出力: 0d3a104 feat: Complete HTDemucs-CUNet implementation with degradation profiles

# ブラウザで確認
# https://github.com/axion15tth/MSRKit
```

## 次のステップ

プッシュが完了したら、別サーバーで:

```bash
# クローン
git clone https://github.com/axion15tth/MSRKit.git
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

詳細は [DEPLOYMENT.md](DEPLOYMENT.md) を参照してください。
