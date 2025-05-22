
from taxonomy import taxonomy_less
from element import hand_caption, object_category, object_shape, object_size, interaction_type, intention, grasp_taxonomy, hand_pose_description

def make_hoi_prompt(hand_bbox: tuple = None, object_bbox: tuple = None, is_right: bool = True):
    def format_bbox(name, bbox):
        return f"{name} bbox: {bbox}" if bbox else f"{name} bbox: none"
    if is_right:
        hand = format_bbox("Right hand", hand_bbox)
        which_hand = "A right hand"
    else:
        hand = format_bbox("Left hand", hand_bbox)
        which_hand = "A left hand"
    obj = format_bbox("Object", object_bbox)

    header = f"""You are an assistant for describing hand-object interactions in images. For the given image, generate a natural language description based on the visual cues provided. Follow these detailed instructions to produce the output:"""

    instruction = f"""
Instructions:
1. Use the provided detected hand and object bounding boxes (x1, y1, x2, y2) as reference to describe the interaction. If no bounding box is detected, use “none” for that field. The bounding boxes are given below:
• {hand}
• {obj}

2. Focus on the following key aspects of the hand-object interaction:
• Shape and size: Describe the geometric shape of the object and its relative size to the hand.
• Contact and interaction: What physical relationship is happening between the hand and object?
• Pose and action: Describe how the hand is positioned and what action is being performed.

3. Use the following output keys only. Do not include any other fields, explanations, or comments. Each value should be brief (≤ 1–2 sentences) and formatted exactly as specified.

Write your answers in the following format:

1. hand_caption: ...
2. object_category: ...
3. object_shape: ...
4. object_size: ...
5. interaction_type: ...
6. intention: ...
7. grasp_taxonomy: ...
8. hand_pose_description: ...

Do not use JSON, curly brackets, or quotation marks. Only use the format above.
"""

    keys = f"""
Output Keys and Descriptions:

{hand_caption(which_hand, ver=2)}
{object_category()}
{object_shape(ver=2)}
{object_size()}
{interaction_type(ver=2)}
{intention()}
{grasp_taxonomy(less=True)}
{hand_pose_description(ver=1)}
"""
    return f"{header}\n{instruction.strip()}\n{keys.strip()}"