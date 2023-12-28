import cv2
import numpy as np
from enum import Enum
import blend_modes as bm


class LayerMode(Enum):
    CLIP = 0
    INVERT_CLIP = 1
    LIGHTEN = 2
    DIFFERENCE = 3


def blend_images(top, bottom, blend_mode_function):
    top_alpha = cv2.cvtColor(top, cv2.COLOR_RGB2RGBA)
    bottom_alpha = cv2.cvtColor(bottom, cv2.COLOR_RGB2RGBA)

    # Convert to float for blend_modes module
    top_alpha_float = top_alpha / 255.0
    bottom_alpha_float = bottom_alpha / 255.0

    # Do the blending with lighten
    blend_frame = blend_mode_function(top_alpha_float, bottom_alpha_float, 1.0)

    # Convert back to int
    final_frame_alpha = (blend_frame * 255).astype(np.uint8)

    # Strip alpha channel
    return cv2.cvtColor(final_frame_alpha, cv2.COLOR_RGBA2RGB)


def layer_images(top, bottom, mode):
    match mode:
        case LayerMode.CLIP:
            return np.clip(
                np.maximum(top, bottom),
                0,
                256
            ).astype(np.uint8)
        case LayerMode.INVERT_CLIP:
            return np.clip(
                        1 - np.multiply(1 - top, 1 - bottom),
                        0,
                        256
            ).astype(np.uint8)
        case LayerMode.LIGHTEN:
            return blend_images(top=top, bottom=bottom, blend_mode_function=bm.lighten_only)
        case LayerMode.DIFFERENCE:
            return blend_images(top=top, bottom=bottom, blend_mode_function=bm.difference)
        case _:
            raise ValueError("Invalid layer mode")
