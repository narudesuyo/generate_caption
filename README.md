# generate_caption

Hand-object interaction (HOI) キャプション生成パイプライン。
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

### Step 1: VLM キャプション生成

Qwen2.5-VL を使い、画像から以下の8項目を生成:

1. `hand_caption` — 手の動作
2. `object_category` — 物体カテゴリ
3. `object_shape` — 物体の形状
4. `object_size` — 手に対する相対サイズ
5. `interaction_type` — インタラクション種別
6. `intention` — 推定される意図
7. `grasp_taxonomy` — 把持タクソノミー (VLM推定)
8. `hand_pose_description` — 手の姿勢記述

```bash
python -m generate_caption caption \
  --dataset hograspnet \
  --data-root /large/naru/data_selected \
  --setup s1 --split train \
  --skip-existing
```

対応データセット: `hograspnet`, `ho3d`, `dexycb`

### Step 2: Grasp Taxonomy 分類 → キャプション更新

分類モデル (`classification/inference.py`) で grasp taxonomy を推論し、
`pred_caption.py` で VLM が出力した `grasp_taxonomy` フィールドを
分類モデルの予測結果に置換する。

```bash
# 2a. 分類モデルで taxonomy 推論
cd classification
python inference.py

# 2b. キャプション内の taxonomy を推論結果に置換
python pred_caption.py
```

出力: `pred_taxonomy_caption/*.txt` (taxonomy が分類モデルの予測に更新されたキャプション)

### Step 3: 要約

Mistral-7B で詳細キャプションを1文に要約する。

```bash
python -m generate_caption summarize \
  --caption-root /large/naru/data_selected \
  --caption-subdir pred_taxonomy_caption \
  --summary-subdir summary \
  --skip-existing
```

`--visualize` を付けると画像に要約テキストを重畳して保存。

## ファイル構成

```
generate_caption/
├── __init__.py            # パッケージ初期化
├── __main__.py            # CLI エントリポイント
├── taxonomy.py            # 33種の grasp taxonomy 定義
├── element.py             # VLM プロンプトの各項目テンプレート
├── prompt.py              # プロンプト組み立て
├── vlm.py                 # Qwen2.5-VL モデル読み込み・推論
├── utils.py               # bbox ユーティリティ・可視化
├── generate_caption.py    # Step 1: マルチGPUキャプション生成
├── summarize.py           # Step 3: マルチGPU要約生成
├── dex_ycb.py             # DexYCB データセットクラス
└── cfg/                   # HOGraspNet データセット split キャッシュ
```

## 必要モデル

| Step | モデル | 用途 |
|------|--------|------|
| 1 | Qwen/Qwen2.5-VL-7B-Instruct | キャプション生成 |
| 2 | classification/checkpoints/best.ckpt | Grasp taxonomy 分類 |
| 3 | mistralai/Mistral-7B-Instruct-v0.2 | 要約生成 |
