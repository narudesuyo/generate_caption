import os
from PIL import Image

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info
from .prompt import make_hoi_prompt, make_hoi_prompt_compact, make_hand_only_prompt

DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_vlm_model(model_name=DEFAULT_VLM_MODEL, device_map="auto", dtype=torch.bfloat16):
    """Load Qwen2.5-VL model and processor."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def process_image_with_bbox(image_path: str,
                            hand_bbox: tuple = None,
                            object_bbox: tuple = None,
                            is_right: bool = True,
                            model=None,
                            processor=None,
                            output_path: str = None,
                            max_new_tokens: int = 200,
                            temperature: float = 0.7,
                            compact=False,
                            prompt_type="hoi",
                            ):
    """
    Args:
        image_path: Path to input image
        hand_bbox: (x1, y1, x2, y2) or None
        object_bbox: (x1, y1, x2, y2) or None
        model: Qwen2.5-VL model (optional; will load if None)
        processor: Qwen2.5-VL processor (optional; will load if None)
        output_path: Optional output file path to save result
        prompt_type: "hoi" (default), "compact", or "hand_only"
    Returns:
        output_text: Generated text from the model
    """
    if model is None or processor is None:
        model, processor = load_vlm_model()

    if prompt_type == "hand_only":
        prompt = make_hand_only_prompt(is_right)
    elif compact or prompt_type == "compact":
        prompt = make_hoi_prompt_compact(hand_bbox, object_bbox, is_right)
    else:
        prompt = make_hoi_prompt(hand_bbox, object_bbox, is_right)
    base_image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output_text)

    return output_text, prompt
