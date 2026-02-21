"""generate_caption: VLM-based hand-object interaction caption generation."""

from .taxonomy import taxonomy, taxonomy_less
from .prompt import make_hoi_prompt, make_hoi_prompt_compact
from .vlm import process_image_with_bbox, load_vlm_model
from .utils import draw_and_save_bbox, expand_bbox, expand_and_convert_bbox_xywh, xywh_to_xyxy
