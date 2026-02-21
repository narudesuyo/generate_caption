from .taxonomy import taxonomy_less
from .element import (hand_caption, object_category, object_shape, object_size,
                      interaction_type, intention, grasp_taxonomy, hand_pose_description)

# Current prompt uses: hand_caption v1, object_shape v2, interaction_type v2, hand_pose_description v1

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

{hand_caption(which_hand, ver=1)}
{object_category()}
{object_shape(ver=2)}
{object_size()}
{interaction_type(ver=2)}
{intention()}
{grasp_taxonomy(less=True)}
{hand_pose_description(ver=1)}
"""
    return f"{header}\n{instruction.strip()}\n{keys.strip()}"

def make_hand_only_prompt(is_right: bool = True):
    """Prompt for hand-only images (no object bbox). Asks whether the hand is interacting with an object."""
    which_hand = "A right hand" if is_right else "A left hand"

    return f"""You are an assistant for describing hand poses in images.
The image shows a hand with no bounding box annotations. Describe the hand based on visual cues.

Write your answers in the following format. Each value should be brief (1-2 sentences).

1. hand_caption: A concise description of what the hand is doing. Start with "{which_hand}". Use present tense. Max 25 words.
2. has_object_interaction: Is the hand interacting with an object? Answer "yes" or "no".
3. object_category: If yes, the object category (single noun). If no, write "none".
4. object_shape: If yes, the object's geometric shape ("cuboid", "cylindrical", "spherical", "flatRectangular", "irregular", or "unknown"). If no, write "none".
5. object_size: If yes, the object's size relative to the hand ("tiny", "small", "medium", "large", "huge", or "unknown"). If no, write "none".
6. interaction_type: If yes, describe the interaction (e.g., grabbing, holding). If no, write "none".
7. intention: The inferred high-level goal. Start with "to". Max 10 words. If unclear, write "undetermined".
8. grasp_taxonomy: If the hand is grasping an object, choose one from: {taxonomy_less}. If not grasping, write "none".
9. hand_pose_description: Describe the hand's orientation, finger shape, and contact points. Max 20 words.

Do not use JSON, curly brackets, or quotation marks. Only use the format above."""


def make_hoi_prompt_compact(hand_bbox=None, object_bbox=None, is_right=True):
    def fmt_bbox(name, bbox):
        return f"{name} bbox: {bbox}" if bbox else f"{name} bbox: none"

    which_hand = "right" if is_right else "left"
    hand = fmt_bbox("Hand", hand_bbox)
    obj = fmt_bbox("Object", object_bbox)

    prompt = f"""
You are an assistant that outputs a single short English phrase describing a hand-object interaction.

Use ONLY this format:
The {which_hand} hand [verb phrase] [object phrase]

Rules:
- Use a concise verb phrase (e.g., holding, touching, grasping, pointing).
- Use a concise noun phrase for the object (e.g., a cup, a smartphone).
- If no object is visible, use "nothing".
- Output exactly one short line, no punctuation, no explanations.
- Do NOT use parentheses, colons, or quotation marks.

Context:
- {hand}
- {obj}

Now produce the output line starting with "The {which_hand} hand".
""".strip()

    return prompt

