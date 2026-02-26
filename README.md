# generate_caption

VLM ベースの Hand-Object Interaction (HOI) キャプション生成パイプライン。
VLM によるキャプション生成 → grasp taxonomy 分類 → 要約の3ステップで、
画像から自然言語の HOI 記述を生成する。

## パイプライン概要

```
Step 1: VLM Caption          Step 2: Grasp Taxonomy       Step 3: Summarize
─────────────────────        ────────────────────────      ──────────────────
画像 + bbox                  分類モデル推論                Mistral-7B で要約
  ↓                            ↓                            ↓
Qwen2.5-VL で                予測された taxonomy を       1文の自然言語記述
8項目のキャプション生成       キャプションに埋め込む       を生成
  ↓                            ↓                            ↓
caption/*.txt                pred_taxonomy_caption/*.txt   summary/*.txt
```

## セットアップ

```bash
conda create -n generate_caption python=3.10 -y
conda activate generate_caption

# CUDA バージョンに合わせてインストール
pip install -r requirements-cu118.txt   # CUDA 11.8
pip install -r requirements-cu124.txt   # CUDA 12.4
```

## Step 1: VLM キャプション生成

Qwen2.5-VL を使い、画像から以下の8項目を生成:

1. `hand_caption` -- 手の動作
2. `object_category` -- 物体カテゴリ
3. `object_shape` -- 物体の形状
4. `object_size` -- 手に対する相対サイズ
5. `interaction_type` -- インタラクション種別
6. `intention` -- 推定される意図
7. `grasp_taxonomy` -- 把持タクソノミー (VLM 推定)
8. `hand_pose_description` -- 手の姿勢記述

```bash
# HOGraspNet
python thirdparty/generate_caption/generate_caption.py \
  --dataset hograspnet \
  --data-root $DATA_ROOT \
  --setup s1 --split train \
  --skip-existing

# HO3D
python thirdparty/generate_caption/generate_caption.py \
  --dataset ho3d \
  --data-root /path/to/HO3D/train \
  --skip-existing

# DexYCB
python thirdparty/generate_caption/generate_caption.py \
  --dataset dexycb \
  --data-root $DATA_ROOT \
  --setup s1 --split test \
  --skip-existing

# FreiHAND (hand-only, bbox なし)
python thirdparty/generate_caption/generate_caption.py \
  --dataset freihand \
  --data-root $DATA_ROOT \
  --prompt-type hand_only \
  --skip-existing
```

### 対応データセット

| データセット | bbox | prompt-type | 備考 |
|-------------|------|-------------|------|
| hograspnet | hand + object | hoi (default) | setup/split 指定可 |
| ho3d | hand + object | hoi (default) | `rgb/*.png` を走査 |
| dexycb | hand + object | hoi (default) | setup/split 指定可 |
| freihand | なし | hand_only | 原画像 (0-32559) + evaluation |

### オプション

| フラグ | デフォルト | 説明 |
|--------|-----------|------|
| `--dataset` | (必須) | `hograspnet`, `ho3d`, `dexycb`, `freihand` |
| `--data-root` | `$DATA_ROOT` | データセットのルートディレクトリ |
| `--prompt-type` | `hoi` | `hoi` (構造化), `compact` (1フレーズ), `hand_only` (bbox なし) |
| `--model-name` | `Qwen/Qwen2.5-VL-7B-Instruct` | VLM モデル |
| `--max-new-tokens` | 150 | 最大生成トークン数 |
| `--temperature` | 1.0 | サンプリング温度 |
| `--setup` / `--split` | `s1` / `train` | データセットの setup と split |
| `--skip-existing` | false | 出力ファイルが存在する場合スキップ |

マルチ GPU 推論は `torch.multiprocessing.spawn` で自動分散。

## Step 2: Grasp Taxonomy 分類 → キャプション更新

分類モデルで grasp taxonomy を推論し、VLM が出力した `grasp_taxonomy` フィールドを分類モデルの予測結果に置換する。

```bash
# 2a. 分類モデルで taxonomy 推論
cd classification
python inference.py

# 2b. キャプション内の taxonomy を推論結果に置換
python pred_caption.py
```

出力: `pred_taxonomy_caption/*.txt` (taxonomy が分類モデルの予測に更新されたキャプション)

## Step 3: 要約

Mistral-7B で構造化キャプションを1文に要約する。

```bash
python thirdparty/generate_caption/summarize.py \
  --caption-root /path/to/captions \
  --caption-subdir pred_taxonomy_caption \
  --summary-subdir summary \
  --skip-existing
```

`--visualize` を付けると画像に要約テキストを重畳して保存。

### オプション

| フラグ | デフォルト | 説明 |
|--------|-----------|------|
| `--caption-root` | (必須) | キャプションファイルのルートディレクトリ |
| `--caption-subdir` | `pred_taxonomy_caption` | 入力サブディレクトリ名 |
| `--summary-subdir` | `summary` | 出力サブディレクトリ名 |
| `--model-name` | `mistralai/Mistral-7B-Instruct-v0.2` | 要約モデル |
| `--remove-pose-desc` | true | hand_pose_description を除外 |
| `--remove-intention` | false | intention を除外 |
| `--remove-taxonomy` | false | grasp_taxonomy を除外 |
| `--visualize` | false | 画像に要約テキストを重畳保存 |
| `--skip-existing` | false | 出力ファイルが存在する場合スキップ |

マルチ GPU は `CUDA_VISIBLE_DEVICES` で制御。

## ファイル構成

```
generate_caption/
├── __init__.py            # パッケージ初期化
├── __main__.py            # CLI エントリポイント (caption / summarize)
├── generate_caption.py    # Step 1: マルチ GPU キャプション生成
├── summarize.py           # Step 3: マルチ GPU 要約生成
├── vlm.py                 # Qwen2.5-VL モデル読み込み・推論
├── prompt.py              # プロンプト組み立て (hoi / compact / hand_only)
├── element.py             # プロンプト各項目のテンプレート
├── taxonomy.py            # Grasp taxonomy 定義
├── utils.py               # bbox ユーティリティ・可視化
├── dex_ycb.py             # DexYCB データセットクラス
└── cfg/                   # データセット split キャッシュ (.pkl)
```

## 必要モデル

| Step | モデル | 用途 |
|------|--------|------|
| 1 | Qwen/Qwen2.5-VL-7B-Instruct | キャプション生成 |
| 2 | classification/checkpoints/best.ckpt | Grasp taxonomy 分類 |
| 3 | mistralai/Mistral-7B-Instruct-v0.2 | 要約生成 |
