import os
from PIL import Image, ImageDraw

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info
from prompt import make_hoi_prompt


def draw_bboxes(image: Image.Image, hand_bbox: tuple, object_bbox: tuple) -> Image.Image:
    """
    hand_bbox/object_bbox: (x1, y1, x2, sy2) in pixel coordinates
    """
    draw = ImageDraw.Draw(image)
    draw.rectangle(hand_bbox, outline="red", width=4)
    draw.text((hand_bbox[0], hand_bbox[1] - 10), "hand", fill="red")
    draw.rectangle(object_bbox, outline="blue", width=4)
    draw.text((object_bbox[0], object_bbox[1] - 10), "object", fill="blue")
    return image


def process_image_with_bbox(image_path: str,
                            hand_bbox: tuple,
                            object_bbox: tuple,
                            is_right: bool,
                            model=None,
                            processor=None,
                            output_path: str = None,
                            max_new_tokens: int = 200,
                            temperature: float = 0.7,
                            ):
    """
    Args:
        image_path: Path to input image
        hand_bbox: (x1, y1, x2, y2)
        object_bbox: (x1, y1, x2, y2)
        model: Qwen2.5-VL model (optional; will load if None)
        processor: Qwen2.5-VL processor (optional; will load if None)
        output_path: Optional output file path to save result
    Returns:
        output_text: Generated text from the model
    """
    if model is None or processor is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    prompt = make_hoi_prompt(hand_bbox, object_bbox, is_right)
    base_image = Image.open(image_path).convert("RGB")
    # annotated_image = draw_bboxes(base_image.copy(), hand_bbox, object_bbox)

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