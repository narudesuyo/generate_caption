"""Unified caption summarization using Mistral-7B (multi-GPU)."""

import argparse
import os
import re
import textwrap
import multiprocessing
from glob import glob

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont

DEFAULT_SUMMARY_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def format_prompt(text, remove_pose_desc=True, remove_intention=False, remove_taxonomy=False):
    """Format caption text into a summarization prompt."""
    cleaned_text = text
    if remove_pose_desc:
        cleaned_text = re.sub(r"^(?:\d+\.\s*)?hand_pose_description:.*(?:\n|$)", "", cleaned_text, flags=re.MULTILINE)
    if remove_intention:
        cleaned_text = re.sub(r"^(?:\d+\.\s*)?intention:.*(?:\n|$)", "", cleaned_text, flags=re.MULTILINE)
    if remove_taxonomy:
        cleaned_text = re.sub(r"^(?:\d+\.\s*)?grasp_taxonomy:.*(?:\n|$)", "", cleaned_text, flags=re.MULTILINE)

    if remove_taxonomy:
        instruction = (
            "You are given a list of bullet points describing an article.\n"
            "Summarize the overall content in a single sentence.\n"
            "Start your response with either 'A right hand' or 'A left hand', "
            "depending on which is more appropriate based on the content.\n"
            "Do not include any information that is not explicitly mentioned in the bullet points.\n\n"
            "### Instruction:\n"
            "Summarize the following bullet points as a single sentence starting with "
            "either 'A right hand' or 'A left hand'. "
            "You must not add any information that is not clearly and explicitly stated "
            "in the bullet points.\n\n"
        )
    else:
        instruction = (
            "You are given a list of bullet points describing an article.\n"
            "Summarize the overall content in a single sentence.\n"
            "Start your response with either 'A right hand' or 'A left hand', "
            "depending on which is more appropriate based on the content.\n"
            "Do not include any information that is not explicitly mentioned in the bullet points.\n"
            "Be sure to include the taxonomy explicitly if it is mentioned in the bullet points.\n\n"
            "### Instruction:\n"
            "Summarize the following bullet points as a single sentence starting with "
            "either 'A right hand' or 'A left hand'. "
            "The summary must include the taxonomy if it appears in the bullet points. "
            "You must not add any information that is not clearly and explicitly stated "
            "in the bullet points.\n\n"
        )

    return instruction + f"### Input:\n{cleaned_text}\n\n### Response:"


def draw_summary_on_image(image_path, summary_text, save_path):
    """Draw summary text below an image and save."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    font_size = 56
    padding = 20

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    wrapped_lines = []
    for line in summary_text.splitlines():
        wrapped_lines += textwrap.wrap(line, width=40)

    text_height = font_size * len(wrapped_lines) + padding * 2
    new_height = height + text_height
    new_img = Image.new("RGB", (width, new_height), (0, 0, 0))
    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)
    y = height + padding
    for line in wrapped_lines:
        draw.text((10, y), line, fill=(255, 255, 255), font=font)
        y += font_size

    new_img.save(save_path)


def process_files(gpu_id, file_list, args):
    """Process a chunk of files on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype="auto"
    ).eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    bs = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(file_list), bs), desc=f"GPU {gpu_id}"):
        chunk = file_list[i:i + bs]
        prompts = []
        out_paths = []

        for path in chunk:
            save_path = path.replace(args.caption_subdir, args.summary_subdir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(path, "r") as f:
                caption = f.read().strip()
            if not caption:
                continue

            prompt = format_prompt(
                caption,
                remove_pose_desc=args.remove_pose_desc,
                remove_intention=args.remove_intention,
                remove_taxonomy=args.remove_taxonomy,
            )
            prompts.append(prompt)
            out_paths.append(path)

        if not prompts:
            continue

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=150, do_sample=False, temperature=0.7, top_p=0.95
        )
        summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for src_path, summary in zip(out_paths, summaries):
            summary_text = summary.split("### Response:")[-1].strip()
            save_path = src_path.replace(args.caption_subdir, args.summary_subdir)
            with open(save_path, "w") as f:
                f.write(summary_text)

            if args.visualize:
                img_path = src_path.replace("text", "vis").replace(".txt", ".jpg")
                if os.path.exists(img_path):
                    vis_save_path = img_path.replace(".jpg", "_with_summary.png")
                    draw_summary_on_image(img_path, summary_text, vis_save_path)


def main():
    parser = argparse.ArgumentParser(description="Summarize captions using Mistral-7B")
    parser.add_argument("--caption-root", required=True, help="Root directory containing caption files")
    parser.add_argument("--caption-subdir", default="pred_taxonomy_caption",
                        help="Subdirectory name for captions")
    parser.add_argument("--summary-subdir", default="summary",
                        help="Subdirectory name for output summaries")
    parser.add_argument("--model-name", default=DEFAULT_SUMMARY_MODEL)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--visualize", action="store_true",
                        help="Also render summaries on images")
    parser.add_argument("--remove-pose-desc", action="store_true", default=True)
    parser.add_argument("--remove-intention", action="store_true")
    parser.add_argument("--remove-taxonomy", action="store_true")
    args = parser.parse_args()

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [int(i) for i in cuda_visible.split(",") if i.strip().isdigit()]
    if not gpu_ids:
        gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        print("No GPUs available.")
        return

    txt_files = glob(os.path.join(args.caption_root, "**", args.caption_subdir, "*.txt"), recursive=True)
    txt_files = [f for f in txt_files if os.path.isfile(f)]

    # Pre-filter completed items so remaining work is distributed evenly
    if args.skip_existing:
        remaining = [f for f in txt_files if not os.path.exists(f.replace(args.caption_subdir, args.summary_subdir))]
        n_skipped = len(txt_files) - len(remaining)
        if n_skipped > 0:
            print(f"{n_skipped}/{len(txt_files)} already done, {len(remaining)} remaining")
    else:
        remaining = txt_files

    print(f"Processing {len(remaining)} caption files across {len(gpu_ids)} GPUs")

    chunk_size = (len(remaining) + len(gpu_ids) - 1) // len(gpu_ids)
    chunks = [remaining[i * chunk_size:(i + 1) * chunk_size] for i in range(len(gpu_ids))]

    processes = []
    for proc_id, gpu_id in enumerate(gpu_ids):
        p = multiprocessing.Process(target=process_files, args=(gpu_id, chunks[proc_id], args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
