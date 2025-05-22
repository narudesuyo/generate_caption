from taxonomy import taxonomy_less, taxonomy
def hand_caption(which_hand, ver=1):
    if ver == 1:
        return f"""1. hand_caption
• What to include: A concise description of the primary action the hand is performing.
• Write in the form: {which_hand} [does something]. Use present tense and active voice. Limit to ≤ 20 words.
"""
    elif ver == 2:
        return f"""1. hand_caption
• Describe what the {which_hand} hand is doing in a natural and clear way.
• Focus on the key action or interaction, using present tense. Avoid long or overly technical descriptions.
• You may start with "{which_hand} hand" or not — be flexible. Keep it concise (around 20 words or fewer).
"""

def object_category():
    return """2. object_category
• What to include: The semantic category of the object the hand is interacting with.
• Format: Use a single noun (singular). If the object cannot be identified, use “unknown”.
"""


def object_shape(ver=1):
    if ver == 1:
        return """3. object_shape
• What to include: The object’s perceived geometric shape based on the image.
• Format: Use one of: “cuboid”, “cylindrical”, “spherical”, “flatRectangular”, “irregular”, or “unknown”.
"""


    elif ver == 2:
        return """3. object_shape
• Describe the geometric shape in simple words like “boxy”, “spherical”, “flat”, or others.
• Prefer short phrases. If unclear, use “unknown”.
"""


def object_size():
    return """4. object_size
• What to include: The object’s size relative to the size of the hand. Use: “tiny”, “small”, “medium”, “large”, “huge”, or “unknown”.
"""


def interaction_type(ver=1):
    if ver == 1:
        return """5. interaction_type
• What to include: The type of interaction the hand is performing with the object, using one of:
"grab","tap","push","pull","twist","pressButton","support","point","gesturing","idleTouch","toolUse", or "unknown".
"""
    elif ver == 2:
        return """5. interaction_type
• What to include: Describe the type of interaction the hand is performing with the object in a concise verb phrase.
• Focus on the action itself (e.g., grabbing, tapping, twisting).
• You may use your own wording. If uncertain, use "unknown".

Examples: "grabbing", "tapping", "pushing", "supporting", "pointing", "twisting", "pressing a button", "using as a tool", "resting on", "unknown"
"""

def intention():
    return """6. intention
• What to include: The inferred high-level goal of the hand. Start with “to”. Use ≤ 10 words. If unclear, write “undetermined”.
"""


def grasp_taxonomy(less=True):
    if less:
        return f"""7. grasp_taxonomy
• What to include: The grasp type according to a predefined taxonomy.
• Format: Choose one from the following list exactly as written: {taxonomy_less}
"""
    else:
        return f"""7. grasp_taxonomy
• What to include: The grasp type according to a predefined taxonomy.
• Format: Choose one from the following list exactly as written: {taxonomy}
"""


def hand_pose_description(ver=1):
    if ver == 1:
        return """8. hand_pose_description
• What to include: Describe the hand’s orientation, finger shape, and contact points. Use ≤ 20 words.
• If any aspect is unclear from the image, you may omit it.
"""
    elif ver == 2:
         return """8. hand_pose_description
• What to include: For each finger, describe the pose (e.g. straight, curled, extended) and whether it is in contact with the object.
• Also describe the orientation of the palm (e.g. facing up, down, inward, outward, sideways).
• Include only what is visible or can be reasonably inferred. If unknown, write "unknown".

Format:
palm:
  orientation: ...

thumb:
  pose: ...
  contact: ...
index:
  pose: ...
  contact: ...
middle:
  pose: ...
  contact: ...
ring:
  pose: ...
  contact: ...
little:
  pose: ...
  contact: ...
"""
