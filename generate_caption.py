"""Unified multi-GPU caption generation for HOGraspNet, HO3D, DexYCB, and FreiHAND."""

import argparse
import ast
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
from huggingface_hub import snapshot_download
from tqdm import tqdm

from vlm import load_vlm_model, process_image_with_bbox
from utils import expand_bbox, expand_and_convert_bbox_xywh


def iter_hograspnet(data_root, setup='s1', split='train', **kwargs):
    """Yield (image_path, hand_bbox, object_bbox, is_right, save_path) for HOGraspNet."""
    sys.path.insert(0, data_root)
    from src.dataset.HOG_dataloader import HOGDataset
    dataset = HOGDataset(
        setup=setup, split=split,
        db_path=os.path.join(data_root, 'HOGraspNet'),
        load_pkl=False, use_aug=False,
    )
    for i in range(len(dataset)):
        data = dataset[i]
        bbox_hand = expand_and_convert_bbox_xywh(data['bbox_hand'])
        bbox_obj = expand_bbox(data['bbox_obj'])
        s, t, c, f = data['subject'], data['trial'], data['camera'], data['frame']
        save_path = os.path.join(data_root, "HOGraspNet", "text_dummy", s, t, c, f"{c}_{f}.txt")
        yield data['rgb_path'], tuple(bbox_hand), tuple(bbox_obj), data['mano_side'] == 'right', save_path


def iter_ho3d(data_root, **kwargs):
    """Yield (image_path, hand_bbox, object_bbox, is_right, save_path) for HO3D."""
    paths = sorted(glob.glob(os.path.join(data_root, "*/rgb/*.png")))
    for image_path in paths:
        bbox_hand_path = image_path.replace("rgb", "hand_bbox").replace(".png", ".txt")
        bbox_obj_path = image_path.replace("rgb", "obj_bbox").replace(".png", ".txt")
        if not os.path.exists(bbox_hand_path) or not os.path.exists(bbox_obj_path):
            continue
        with open(bbox_hand_path, "r") as f:
            hand_bbox = ast.literal_eval(f.read().strip())
        with open(bbox_obj_path, "r") as f:
            object_bbox = ast.literal_eval(f.read().strip())
        save_path = image_path.replace("rgb", "caption").replace(".png", ".txt")
        yield image_path, hand_bbox, object_bbox, True, save_path


def _bbox_from_joint_2d(joint_2d, img_w=640, img_h=480):
    """Compute xyxy bbox from 2D joint coordinates, clipped to image bounds."""
    valid = ~np.isnan(joint_2d).any(axis=1) & (joint_2d >= 0).all(axis=1)
    if valid.sum() == 0:
        return None
    pts = joint_2d[valid]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return [max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(y2))]


def _bbox_from_seg(seg, target_id=None):
    """Compute xyxy bbox from segmentation mask.

    Args:
        seg: (H, W) segmentation mask (0=bg, 255=hand, 1-21=YCB object IDs).
        target_id: If given, only use pixels matching this object ID.
                   If None, use all non-background, non-hand pixels.
    """
    if target_id is not None:
        obj_mask = seg == target_id
    else:
        obj_mask = (seg > 0) & (seg < 255)
    if not obj_mask.any():
        return None
    ys, xs = np.where(obj_mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def iter_dexycb(data_root, setup='s1', split='test', **kwargs):
    """Yield (image_path, hand_bbox, object_bbox, is_right, save_path) for DexYCB.

    Walks the directory structure directly and computes bboxes from label files
    (joint_2d for hand, seg for object). Does not require calibration files.
    """
    import yaml
    subjects = sorted([d for d in os.listdir(data_root)
                       if os.path.isdir(os.path.join(data_root, d)) and 'subject' in d])
    for subject in subjects:
        subject_dir = os.path.join(data_root, subject)
        sequences = sorted([d for d in os.listdir(subject_dir)
                            if os.path.isdir(os.path.join(subject_dir, d))])
        for seq in sequences:
            seq_dir = os.path.join(subject_dir, seq)
            meta_file = os.path.join(seq_dir, "meta.yml")
            if not os.path.exists(meta_file):
                continue
            with open(meta_file, 'r') as f:
                meta = yaml.safe_load(f)
            mano_side = meta['mano_sides'][0]
            is_right = mano_side == 'right'
            grasp_obj_id = meta['ycb_ids'][meta['ycb_grasp_ind']]
            serials = meta['serials']
            for serial in serials:
                cam_dir = os.path.join(seq_dir, serial)
                if not os.path.isdir(cam_dir):
                    continue
                color_files = sorted(glob.glob(os.path.join(cam_dir, "color_*.jpg")))
                for color_path in color_files:
                    frame_id = os.path.basename(color_path).replace("color_", "").replace(".jpg", "")
                    label_path = os.path.join(cam_dir, f"labels_{frame_id}.npz")
                    if not os.path.exists(label_path):
                        continue
                    labels = np.load(label_path)
                    seg_h, seg_w = labels['seg'].shape[:2]
                    hand_bbox = _bbox_from_joint_2d(labels['joint_2d'][0], img_w=seg_w, img_h=seg_h)
                    obj_bbox = _bbox_from_seg(labels['seg'], target_id=grasp_obj_id)
                    if hand_bbox is None:
                        # Fallback to full-image hand box when joints are invalid.
                        hand_bbox = [0, 0, seg_w - 1, seg_h - 1]
                    save_path = color_path.replace("color", "caption").replace(".jpg", ".txt")
                    yield color_path, tuple(expand_bbox(hand_bbox)), \
                          tuple(expand_bbox(obj_bbox)) if obj_bbox else None, \
                          is_right, save_path


def iter_freihand(data_root, **kwargs):
    """Yield (image_path, hand_bbox, object_bbox, is_right, save_path) for FreiHAND.

    Uses original training images (indices 0-32559) and evaluation images.
    data_root should be $DATA_ROOT (parent of FreiHAND/).
    """
    # Training (original only)
    training_dir = os.path.join(data_root, "FreiHAND", "train", "training")
    rgb_dir = os.path.join(training_dir, "rgb")
    n_original = 32560
    for idx in range(n_original):
        image_path = os.path.join(rgb_dir, f"{idx:08d}.jpg")
        if not os.path.exists(image_path):
            continue
        save_path = os.path.join(training_dir, "caption", f"{idx:08d}.txt")
        yield image_path, None, None, True, save_path

    # Evaluation
    eval_dir = os.path.join(data_root, "FreiHAND", "train", "evaluation")
    eval_rgb_dir = os.path.join(eval_dir, "rgb")
    if os.path.isdir(eval_rgb_dir):
        for fname in sorted(os.listdir(eval_rgb_dir)):
            if not fname.endswith(".jpg"):
                continue
            image_path = os.path.join(eval_rgb_dir, fname)
            save_path = os.path.join(eval_dir, "caption", fname.replace(".jpg", ".txt"))
            yield image_path, None, None, True, save_path


DATASET_ITERS = {
    'hograspnet': iter_hograspnet,
    'ho3d': iter_ho3d,
    'dexycb': iter_dexycb,
    'freihand': iter_freihand,
}


def run_on_gpu(rank, world_size, args):
    torch.cuda.set_device(rank)

    iter_fn = DATASET_ITERS[args.dataset]
    samples = list(iter_fn(args.data_root, setup=args.setup, split=args.split))

    # Pre-filter completed items so remaining work is distributed evenly
    if args.skip_existing:
        remaining = [s for s in samples if not os.path.exists(s[4])]
        n_skipped = len(samples) - len(remaining)
        if n_skipped > 0:
            print(f"GPU {rank}: {n_skipped}/{len(samples)} already done, {len(remaining)} remaining")
    else:
        remaining = samples

    my_items = remaining[rank::world_size]
    if not my_items:
        print(f"GPU {rank}: No items to process, skipping model load")
        return

    model, processor = load_vlm_model(
        model_name=args.model_name,
        device_map={"": rank},
        dtype=None,
    )

    for image_path, hand_bbox, object_bbox, is_right, save_path in tqdm(
        my_items, desc=f"GPU {rank}"
    ):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        process_image_with_bbox(
            image_path=image_path,
            hand_bbox=hand_bbox,
            object_bbox=object_bbox,
            is_right=is_right,
            model=model,
            processor=processor,
            output_path=save_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            compact=args.compact,
            prompt_type=args.prompt_type,
        )


def main():
    parser = argparse.ArgumentParser(description="Generate HOI captions using VLM")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_ITERS.keys()))
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""),
                        help="Root directory for the dataset (default: $DATA_ROOT)")
    parser.add_argument("--setup", default="s1", help="Dataset setup (default: s1)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--compact", action="store_true", help="Use compact prompt")
    parser.add_argument("--prompt-type", default="hoi", choices=["hoi", "compact", "hand_only"],
                        help="Prompt type: hoi (default), compact, or hand_only (no bbox)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output file exists")
    args = parser.parse_args()

    iter_fn = DATASET_ITERS[args.dataset]
    samples = list(iter_fn(args.data_root, setup=args.setup, split=args.split))
    print(f"Data root: {args.data_root}")
    print(f"Dataset: {args.dataset}, Total samples: {len(samples)}")
    if samples:
        print(f"First image: {samples[0][0]}")
        print(f"First save:  {samples[0][4]}")

    # Pre-resolve model to local path so spawned workers don't race on hub cache
    args.model_name = snapshot_download(args.model_name)
    print(f"Model path: {args.model_name}")

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for inference")
    mp.spawn(run_on_gpu, args=(world_size, args), nprocs=world_size)


if __name__ == "__main__":
    main()
