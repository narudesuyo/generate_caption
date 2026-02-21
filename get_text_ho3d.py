import glob
import sys
import pickle
from VLM import process_image_with_bbox
from tqdm import tqdm
sys.path.append("/home/narus/hand_reconstruction/HOGraspNet/")
import sys
import os
import torch
import torch.multiprocessing as mp
import pickle
from tqdm import tqdm
from PIL import Image

# sys.path設定（環境依存なし）
hograspnet_root = "/home/narus/hand_reconstruction/HOGraspNet"
sys.path.append(hograspnet_root)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from src.dataset.HOG_dataloader import HOGDataset
from utils import draw_and_save_bbox, expand_and_convert_bbox_xywh, expand_bbox
from VLM import process_image_with_bbox

def run_on_gpu(rank, world_size):
    print(f"Starting process on GPU {rank}")
    
    torch.cuda.set_device(rank)

    # モデルとプロセッサのロード（GPU割り当て）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float32,
        device_map={"": rank}
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # データセットの構築
    selected_root_dir = "/data2/narus/HO3D/train"
    selected_paths = sorted(glob.glob(f"{selected_root_dir}/*/rgb/*.png"))
    start_idx = 0
    for i in reversed(tqdm(range(start_idx + rank, len(selected_paths), world_size), desc=f"GPU {rank}")):
        image_path = selected_paths[i]
       # bbox_path = data['bbox_path']
       # bbox_data = pickle.load(open(bbox_path, 'rb'))
        mano_side = "right"
        is_right = (mano_side == 'right')
        bbox_hand_path = image_path.replace("rgb", "hand_bbox").replace(".png", ".txt")
        bbox_obj_path = image_path.replace("rgb", "obj_bbox").replace(".png", ".txt")
        with open(bbox_hand_path, "r") as f:
            hand_bbox = f.read().strip()
        with open(bbox_obj_path, "r") as f:
            object_bbox = f.read().strip()
        hand_bbox = eval(hand_bbox)
        object_bbox = eval(object_bbox)

        save_txt_path = image_path.replace("rgb", "caption").replace(".png", ".txt")
        if os.path.exists(save_txt_path):
            continue
        os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)

        output, prompt = process_image_with_bbox(
            image_path=image_path,
            hand_bbox=hand_bbox,
            object_bbox=object_bbox,
            is_right=is_right,
            model=model,
            processor=processor,
            output_path=save_txt_path,
            max_new_tokens=150,
            temperature=1,
            compact=False
        )

        if i == rank:
            print(f"[GPU {rank}] Example prompt: {prompt}")

def main():
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for inference")
    mp.spawn(run_on_gpu, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()


