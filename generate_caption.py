"""Unified multi-GPU caption generation for HOGraspNet, HO3D, and DexYCB."""

import argparse
import ast
import glob
import json
import os
import sys

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from .vlm import load_vlm_model, process_image_with_bbox
from .utils import expand_bbox, expand_and_convert_bbox_xywh


def iter_hograspnet(data_root, setup='s1', split='train', **kwargs):
    """Yield (image_path, hand_bbox, object_bbox, is_right, save_path) for HOGraspNet."""
    sys.path.insert(0, data_root)
    from src.dataset.HOG_dataloader import HOGDataset
    dataset = HOGDataset(
        setup=setup, split=split,
        db_path=os.path.join(data_root, 'data_selected'),
        load_pkl=False, use_aug=False,
    )
    for i in range(len(dataset)):
        data = dataset[i]
        bbox_hand = expand_and_convert_bbox_xywh(data['bbox_hand'])
        bbox_obj = expand_bbox(data['bbox_obj'])
        s, t, c, f = data['subject'], data['trial'], data['camera'], data['frame']
        save_path = os.path.join(data_root, "data_selected", "text_dummy", s, t, c, f"{c}_{f}.txt")
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


def iter_dexycb(data_root, setup='s1', split='test', **kwargs):
    """Yield (image_path, hand_bbox, object_bbox, is_right, save_path) for DexYCB."""
    from .dex_ycb import DexYCBDataset
    dataset = DexYCBDataset(setup=setup, split=split)
    for i in range(len(dataset)):
        data = dataset[i]
        image_path = data['color_file']
        if not os.path.exists(image_path):
            continue
        bbox_path = image_path.replace("color", "bbox").replace(".jpg", ".json")
        if not os.path.exists(bbox_path):
            continue
        with open(bbox_path, "r") as f:
            bbox_data = json.load(f)
        save_path = image_path.replace("data_selected", "caption").replace("color", "caption").replace(".jpg", ".txt")
        yield image_path, bbox_data['hand_bbox_xyxy'], bbox_data['obj_bbox_xyxy'], \
              data['mano_side'] == 'right', save_path


def iter_freihand(data_root, **kwargs):
    """Yield (image_path, hand_bbox, object_bbox, is_right, save_path) for FreiHAND.

    Uses augmented images only (indices 32560-130239).
    """
    rgb_dir = os.path.join(data_root, "training", "rgb")
    n_original = 32560
    n_total = 130240
    for idx in range(n_original, n_total):
        image_path = os.path.join(rgb_dir, f"{idx:08d}.jpg")
        if not os.path.exists(image_path):
            continue
        save_path = os.path.join(data_root, "training", "caption", f"{idx:08d}.txt")
        yield image_path, None, None, True, save_path


DATASET_ITERS = {
    'hograspnet': iter_hograspnet,
    'ho3d': iter_ho3d,
    'dexycb': iter_dexycb,
    'freihand': iter_freihand,
}


def run_on_gpu(rank, world_size, args):
    torch.cuda.set_device(rank)
    model, processor = load_vlm_model(
        model_name=args.model_name,
        device_map={"": rank},
        dtype=torch.bfloat16,
    )

    iter_fn = DATASET_ITERS[args.dataset]
    samples = list(iter_fn(args.data_root, setup=args.setup, split=args.split))

    for i in reversed(tqdm(range(rank, len(samples), world_size), desc=f"GPU {rank}")):
        image_path, hand_bbox, object_bbox, is_right, save_path = samples[i]
        if args.skip_existing and os.path.exists(save_path):
            continue
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
    parser.add_argument("--data-root", required=True, help="Root directory for the dataset")
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

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for inference")
    mp.spawn(run_on_gpu, args=(world_size, args), nprocs=world_size)


if __name__ == "__main__":
    main()
