from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from glob import glob
import time
from tqdm import tqdm
import os
from PIL import Image, ImageDraw, ImageFont

# モデル準備
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
).eval()

def format_prompt(text):
    return (
        "A right hand or A left hand. "
        "Start your response with one of them appropriately.\n\n"
        "### Instruction:\n"
        "Summarize the following article concisely in a single sentence.\n\n"
        f"### Input:\n{text}\n\n"
        "### Response:"
    )

# def draw_summary_on_image(image_path, summary_text, save_path):
#     image = Image.open(image_path).convert("RGB")
#     width, height = image.size

#     # 新しい高さを確保（60px 下に追加）
#     new_height = height + 60
#     new_img = Image.new("RGB", (width, new_height), (0, 0, 0))
#     new_img.paste(image, (0, 0))

#     draw = ImageDraw.Draw(new_img)
#     try:
#         font = ImageFont.truetype("arial.ttf", 30)
#     except:
#         font = ImageFont.load_default()

#     draw.text((10, height + 10), summary_text, fill=(255, 255, 255), font=font)
#     new_img.save(save_path)
import textwrap

def draw_summary_on_image(image_path, summary_text, save_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    font_size = 56
    padding = 20

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 🔁 wrap the text to fit within image width
    max_text_width = width - 20  # margin
    wrapped_lines = []
    for line in summary_text.splitlines():  # keep manual \n
        wrapped_lines += textwrap.wrap(line, width=40)  # adjust width if needed

    # テキスト全体の高さ計算
    text_height = font_size * len(wrapped_lines) + padding * 2
    new_height = height + text_height
    new_img = Image.new("RGB", (width, new_height), (0, 0, 0))
    new_img.paste(image, (0, 0))

    # 描画
    draw = ImageDraw.Draw(new_img)
    y = height + padding
    for line in wrapped_lines:
        draw.text((10, y), line, fill=(255, 255, 255), font=font)
        y += font_size

    new_img.save(save_path)

# .txt ファイル探索
txt_files = glob("./text/5_22_1/**/*.txt", recursive=True)
print(f"Found {len(txt_files)} .txt files.")

for path in tqdm(txt_files):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # 要約生成
        prompt = format_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.7,
            top_p=0.95
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary_text = summary.split("### Response:")[-1].strip()

        print(f"txt_file: {path}")

        # 対応画像パスの推定（text -> vis、.txt -> .jpg）
        img_path = path.replace("text", "vis").replace(".txt", ".jpg")

        if not os.path.exists(img_path):
            print(f"[!] Skipped: image not found for {path}")
            continue

        # 保存パス
        save_path = img_path.replace(".jpg", "_with_summary.png")

        # summary 描画して保存
        draw_summary_on_image(img_path, summary_text, save_path)
        print(f"[✓] {img_path} → {save_path}")

    except Exception as e:
        print(f"[!] Failed: {path} — {e}")