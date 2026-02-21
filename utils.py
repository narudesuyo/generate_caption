import os
from PIL import Image, ImageDraw, ImageFont

def draw_and_save_bbox(image_path, hand_bbox, object_bbox, save_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.rectangle(hand_bbox, outline="red", width=3)
    draw.text((hand_bbox[0], hand_bbox[1] - 10), "hand", fill="red")

    draw.rectangle(object_bbox, outline="blue", width=3)
    draw.text((object_bbox[0], object_bbox[1] - 10), "object", fill="blue")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)

def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def expand_and_convert_bbox_xywh(bbox_xywh, scale=1.2, image_size=(640, 480)):
    x, y, w, h = bbox_xywh
    cx = x + w / 2
    cy = y + h / 2
    new_w = w * scale
    new_h = h * scale

    x1 = max(0, cx - new_w / 2)
    y1 = max(0, cy - new_h / 2)
    x2 = min(image_size[0], cx + new_w / 2)
    y2 = min(image_size[1], cy + new_h / 2)

    return [int(x1), int(y1), int(x2), int(y2)]


def draw_and_save_bbox(image_path, hand_bbox, object_bbox, save_path, text=None):
    original_image = Image.open(image_path).convert("RGB")
    width, height = original_image.size

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=10)
    except:
        font = ImageFont.load_default()

    text_lines = text.strip().split("\n") if text else []
    line_height = font.getbbox("A")[3] - font.getbbox("A")[1] + 4  # 行間込み
    extra_height = max(100, line_height * len(text_lines) + 20)  # 最低100px、行数に応じて調整

    new_height = height + extra_height
    new_image = Image.new("RGB", (width, new_height), (255, 255, 255))
    new_image.paste(original_image, (0, 0))
    draw = ImageDraw.Draw(new_image)

    if hand_bbox is not None:
        draw.rectangle([(hand_bbox[0], hand_bbox[1]), (hand_bbox[2], hand_bbox[3])], outline="red", width=3)
        draw.text((hand_bbox[0], max(0, hand_bbox[1] - 10)), "hand", fill="red", font=font)
    if object_bbox is not None:
        draw.rectangle([(object_bbox[0], object_bbox[1]), (object_bbox[2], object_bbox[3])], outline="blue", width=3)
        draw.text((object_bbox[0], max(0, object_bbox[1] - 10)), "object", fill="blue", font=font)

    y_offset = height + 10
    for line in text_lines:
        draw.text((10, y_offset), line, fill="black", font=font)
        y_offset += line_height

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    new_image.save(save_path)
def expand_bbox(bbox, scale=1.2, image_size=(640, 480)):
    """
    bbox: [x1, y1, x2, y2] 形式のバウンディングボックス
    scale: 拡大率（例：1.2）
    image_size: (幅, 高さ) 画像サイズ → 範囲を超えないようにクリップ
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale

    new_x1 = max(0, int(cx - w / 2))
    new_y1 = max(0, int(cy - h / 2))
    new_x2 = min(image_size[0], int(cx + w / 2))
    new_y2 = min(image_size[1], int(cy + h / 2))

    return [new_x1, new_y1, new_x2, new_y2]
