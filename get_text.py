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
    dataset = HOGDataset(
        setup='s3',
        split='train',
        db_path='/data1/narus/HOGraspNet/data',
        load_pkl=False,
        use_aug=False,
    )

    for i in tqdm(range(rank, len(dataset), world_size), desc=f"GPU {rank}"):
        data = dataset[i]
        image_path = data['rgb_path']
        bbox_path = data['bbox_path']
        bbox_data = pickle.load(open(bbox_path, 'rb'))
        mano_side = data['mano_side']
        bbox_hand = data['bbox_hand']
        bbox_obj = expand_bbox(data['bbox_obj'])
        bbox_hand = expand_and_convert_bbox_xywh(bbox_hand)

        is_right = (mano_side == 'right')
        hand_bbox = tuple(bbox_hand)
        object_bbox = tuple(bbox_obj)
        s, t, c, f = data['subject'], data['trial'], data['camera'], data['frame']

        save_txt_path = os.path.join(
            "/data1/narus/HOGraspNet/data/text/", s, t, c, f"{c}_{f}.txt"
        )

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
        )

        if i == rank:
            print(f"[GPU {rank}] Example prompt: {prompt}")

def main():
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for inference")
    mp.spawn(run_on_gpu, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()

