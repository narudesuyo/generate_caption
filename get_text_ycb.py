import sys
import json
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

# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""DexYCB dataset."""

import os
import yaml
import numpy as np

_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

_MANO_JOINTS = [
    'wrist',
    'thumb_mcp',
    'thumb_pip',
    'thumb_dip',
    'thumb_tip',
    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',
    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',
    'ring_mcp',
    'ring_pip',
    'ring_dip',
    'ring_tip',
    'little_mcp',
    'little_pip',
    'little_dip',
    'little_tip'
]

_MANO_JOINT_CONNECT = [
    [0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
    [0,  5], [ 5,  6], [ 6,  7], [ 7,  8],
    [0,  9], [ 9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

_BOP_EVAL_SUBSAMPLING_FACTOR = 4


class DexYCBDataset():
  """DexYCB dataset."""
  ycb_classes = _YCB_CLASSES
  mano_joints = _MANO_JOINTS
  mano_joint_connect = _MANO_JOINT_CONNECT

  def __init__(self, setup, split):
    """Constructor.

    Args:
      setup: Setup name. 's0', 's1', 's2', or 's3'.
      split: Split name. 'train', 'val', or 'test'.
    """
    self._setup = setup
    self._split = split

    assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
    self._data_dir = os.environ['DEX_YCB_DIR']
    self._calib_dir = os.path.join(self._data_dir, "calibration")
    self._model_dir = os.path.join(self._data_dir, "models")

    self._color_format = "color_{:06d}.jpg"
    self._depth_format = "aligned_depth_to_color_{:06d}.png"
    self._label_format = "labels_{:06d}.npz"
    self._h = 480
    self._w = 640

    self._obj_file = {
        k: os.path.join(self._model_dir, v, "textured_simple.obj")
        for k, v in _YCB_CLASSES.items()
    }

    # Seen subjects, camera views, grasped objects.
    if self._setup == 's0':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 != 4]
      if self._split == 'val':
        subject_ind = [0, 1]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 == 4]
      if self._split == 'test':
        subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 == 4]

    # Unseen subjects.
    if self._setup == 's1':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))
      if self._split == 'val':
        subject_ind = [6]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))
      if self._split == 'test':
        subject_ind = [7, 8]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))

    # Unseen camera views.
    if self._setup == 's2':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5]
        sequence_ind = list(range(100))
      if self._split == 'val':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [6]
        sequence_ind = list(range(100))
      if self._split == 'test':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [7]
        sequence_ind = list(range(100))

    # Unseen grasped objects.
    if self._setup == 's3':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [
            i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)
        ]
      if self._split == 'val':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
      if self._split == 'test':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]

    self._subjects = [_SUBJECTS[i] for i in subject_ind]

    self._serials = [_SERIALS[i] for i in serial_ind]
    self._intrinsics = []
    for s in self._serials:
      intr_file = os.path.join(self._calib_dir, "intrinsics",
                               "{}_{}x{}.yml".format(s, self._w, self._h))
      with open(intr_file, 'r') as f:
        intr = yaml.load(f, Loader=yaml.FullLoader)
      intr = intr['color']
      self._intrinsics.append(intr)

    self._sequences = []
    self._mapping = []
    self._ycb_ids = []
    self._ycb_grasp_ind = []
    self._mano_side = []
    self._mano_betas = []
    offset = 0
    for n in self._subjects:
      seq = sorted(os.listdir(os.path.join(self._data_dir, n)))
      seq = [os.path.join(n, s) for s in seq]
      assert len(seq) == 100 # edited by naru
      seq = [seq[i] for i in sequence_ind]
      self._sequences += seq
      for i, q in enumerate(seq):
        meta_file = os.path.join(self._data_dir, q, "meta.yml")
        with open(meta_file, 'r') as f:
          meta = yaml.load(f, Loader=yaml.FullLoader)
        c = np.arange(len(self._serials))
        f = np.arange(meta['num_frames'])
        f, c = np.meshgrid(f, c)
        c = c.ravel()
        f = f.ravel()
        s = (offset + i) * np.ones_like(c)
        m = np.vstack((s, c, f)).T
        self._mapping.append(m)
        self._ycb_ids.append(meta['ycb_ids'])
        self._ycb_grasp_ind.append(meta['ycb_grasp_ind'])
        self._mano_side.append(meta['mano_sides'][0])
        mano_calib_file = os.path.join(self._data_dir, "calibration",
                                       "mano_{}".format(meta['mano_calib'][0]),
                                       "mano.yml")
        with open(mano_calib_file, 'r') as f:
          mano_calib = yaml.load(f, Loader=yaml.FullLoader)
        self._mano_betas.append(mano_calib['betas'])
      offset += len(seq)
    self._mapping = np.vstack(self._mapping)

  def __len__(self):
    return len(self._mapping)

  def __getitem__(self, idx):
    s, c, f = self._mapping[idx]
    d = os.path.join(self._data_dir, self._sequences[s], self._serials[c])
    sample = {
        'color_file': os.path.join(d, self._color_format.format(f)),
        'depth_file': os.path.join(d, self._depth_format.format(f)),
        'label_file': os.path.join(d, self._label_format.format(f)),
        'intrinsics': self._intrinsics[c],
        'ycb_ids': self._ycb_ids[s],
        'ycb_grasp_ind': self._ycb_grasp_ind[s],
        'mano_side': self._mano_side[s],
        'mano_betas': self._mano_betas[s],
    }
    if self._split == 'test':
      sample['is_bop_target'] = (f % _BOP_EVAL_SUBSAMPLING_FACTOR == 0).item()
      id_next = idx + _BOP_EVAL_SUBSAMPLING_FACTOR
      is_last = (id_next >= len(self._mapping) or
                 (np.any(self._mapping[id_next][:2] != [s, c])).item())
      sample['is_grasp_target'] = sample['is_bop_target'] and is_last
    return sample

  @property
  def data_dir(self):
    return self._data_dir

  @property
  def h(self):
    return self._h

  @property
  def w(self):
    return self._w

  @property
  def obj_file(self):
    return self._obj_file

  def get_bop_id_from_idx(self, idx):
    """Returns the BOP scene ID and image ID given an index.

    Args:
      idx: Index of sample.

    Returns:
      scene_id: BOP scene ID.
      im_id: BOP image ID.
    """
    s, c, f = map(lambda x: x.item(), self._mapping[idx])
    scene_id = s * len(self._serials) + c
    im_id = f
    return scene_id, im_id

dataset = DexYCBDataset(setup='s1', split='test')

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

    start_idx = 0
    for i in reversed(tqdm(range(start_idx + rank, len(dataset), world_size), desc=f"GPU {rank}")):
        data = dataset[i]
        image_path = data['color_file']
        if not os.path.exists(image_path):
            continue
       # bbox_path = data['bbox_path']
       # bbox_data = pickle.load(open(bbox_path, 'rb'))
        hand_bbox_xyxy_path = image_path.replace("color", "bbox").replace(".jpg", ".json")
        if not os.path.exists(hand_bbox_xyxy_path):
            continue
        with open(hand_bbox_xyxy_path, "r") as f:
            bbox_data = json.load(f)
            hand_bbox = bbox_data['hand_bbox_xyxy']
            object_bbox = bbox_data['obj_bbox_xyxy']
        is_right = (data['mano_side'] == 'right')
        save_txt_path = image_path.replace("data_selected", "caption").replace("color", "caption").replace(".jpg", ".txt")
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
