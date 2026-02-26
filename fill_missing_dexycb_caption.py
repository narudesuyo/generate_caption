"""Fill missing DexYCB captions only.

Generates `caption_XXXXXX.txt` only for frames where it does not exist.
"""

import argparse
import glob
import os

import numpy as np
import yaml
from tqdm import tqdm

from utils import expand_bbox
from vlm import load_vlm_model, process_image_with_bbox


def bbox_from_joint_2d(joint_2d, img_w=640, img_h=480):
    valid = ~np.isnan(joint_2d).any(axis=1) & (joint_2d >= 0).all(axis=1)
    if valid.sum() == 0:
        return None
    pts = joint_2d[valid]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return [max(0, int(x1)), max(0, int(y1)), min(img_w - 1, int(x2)), min(img_h - 1, int(y2))]


def bbox_from_seg(seg, target_id=None):
    if target_id is not None:
        obj_mask = seg == target_id
    else:
        obj_mask = (seg > 0) & (seg < 255)
    if not obj_mask.any():
        return None
    ys, xs = np.where(obj_mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def main():
    parser = argparse.ArgumentParser(description="Fill missing DexYCB captions")
    parser.add_argument("--dexycb-root", default="/work/narus/DEXYCB_sampled")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    color_paths = sorted(glob.glob(os.path.join(args.dexycb_root, "*", "*", "*", "color_*.jpg")))
    missing = []
    meta_cache = {}
    for color_path in color_paths:
        frame = os.path.basename(color_path).replace("color_", "").replace(".jpg", "")
        cap_path = os.path.join(os.path.dirname(color_path), f"caption_{frame}.txt")
        if os.path.exists(cap_path):
            continue
        seq_dir = os.path.dirname(os.path.dirname(color_path))
        meta = meta_cache.get(seq_dir)
        if meta is None:
            meta_path = os.path.join(seq_dir, "meta.yml")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f)
            meta_cache[seq_dir] = meta
        missing.append((color_path, cap_path, meta))

    print(f"missing captions: {len(missing)}")
    if not missing:
        return

    model, processor = load_vlm_model(model_name=args.model_name, dtype=None)

    for color_path, cap_path, meta in tqdm(missing, desc="fill_dexycb_caption"):
        frame = os.path.basename(color_path).replace("color_", "").replace(".jpg", "")
        label_path = os.path.join(os.path.dirname(color_path), f"labels_{frame}.npz")
        if not os.path.exists(label_path):
            continue
        labels = np.load(label_path)
        seg = labels["seg"]
        h, w = seg.shape[:2]

        hand_bbox = bbox_from_joint_2d(labels["joint_2d"][0], img_w=w, img_h=h)
        if hand_bbox is None:
            hand_bbox = [0, 0, w - 1, h - 1]

        grasp_obj_id = meta["ycb_ids"][meta["ycb_grasp_ind"]]
        obj_bbox = bbox_from_seg(seg, target_id=grasp_obj_id)

        is_right = meta["mano_sides"][0] == "right"
        os.makedirs(os.path.dirname(cap_path), exist_ok=True)
        process_image_with_bbox(
            image_path=color_path,
            hand_bbox=tuple(expand_bbox(hand_bbox)),
            object_bbox=tuple(expand_bbox(obj_bbox)) if obj_bbox else None,
            is_right=is_right,
            model=model,
            processor=processor,
            output_path=cap_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            compact=False,
            prompt_type="hoi",
        )


if __name__ == "__main__":
    main()

