"""
raid.transforms
---------------
Spatial geometric transformation and artifact-free cropping operators
for translation, rotation, and affine sweeps.
"""

from typing import Tuple, Union, Sequence
import numpy as np
from PIL import Image

CARDINAL_DIRECTIONS = ("right", "left", "up", "down")
ROTATION_DIRECTIONS = ("clockwise", "counter_clockwise")

DEFAULT_ROTATION_ANGLES = tuple(sorted(list(set(np.round(np.concatenate([
    np.arange(0.05, 1.05, 0.05),
    np.arange(1.1, 3.1, 0.1),
    np.arange(3.2, 10.2, 0.2),
    np.arange(10.5, 30.5, 0.5)
]), 3)))))


def get_translated_crops(arr: np.ndarray, shift_px: int, direction: str = "right") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts reference and translated sub-arrays for a spatial shift
    in one of the four cardinal directions ('right', 'left', 'up', 'down').
    
    Returns the exact overlapping interior regions between the reference
    and translated image, avoiding border/padding artifacts.
    """
    direction = str(direction).lower().strip()
    if shift_px <= 0:
        return arr, arr

    if direction == "right":
        return arr[:, :-shift_px], arr[:, shift_px:]
    elif direction == "left":
        return arr[:, shift_px:], arr[:, :-shift_px]
    elif direction == "down":
        return arr[:-shift_px, :], arr[shift_px:, :]
    elif direction == "up":
        return arr[shift_px:, :], arr[:-shift_px, :]
    else:
        raise ValueError(f"Invalid cardinal direction: '{direction}'. Must be one of {CARDINAL_DIRECTIONS}.")


def get_rotated_crops(img: Union[Image.Image, np.ndarray], angle_deg: float,
                      direction: str = "clockwise", crop_size: int = 256) -> Tuple[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]:
    """
    Extracts central reference and rotated sub-arrays for a rotation angle in degrees,
    strictly avoiding black border artifacts by cropping inside the maximum inscribed
    square radius (e.g. 256x256 crop for 384x512 TID2013 images).
    """
    is_numpy = isinstance(img, np.ndarray)
    if is_numpy:
        pil_img = Image.fromarray(img)
    else:
        pil_img = img

    w, h = pil_img.size
    cx, cy = w // 2, h // 2
    half_s = crop_size // 2
    box = (cx - half_s, cy - half_s, cx + half_s, cy + half_s)

    direction = str(direction).lower().strip()
    if direction in ("clockwise", "cw", "right"):
        rot_angle = -abs(float(angle_deg))
    elif direction in ("counter_clockwise", "counterclockwise", "ccw", "left"):
        rot_angle = abs(float(angle_deg))
    else:
        raise ValueError(f"Invalid rotation direction: '{direction}'. Must be one of {ROTATION_DIRECTIONS}.")

    ref_cropped = pil_img.crop(box)
    rot_img = pil_img.rotate(rot_angle, resample=Image.BICUBIC, center=(cx, cy))
    rot_cropped = rot_img.crop(box)

    if is_numpy:
        return np.array(ref_cropped), np.array(rot_cropped)
    return ref_cropped, rot_cropped
