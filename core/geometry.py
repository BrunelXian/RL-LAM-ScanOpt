"""Geometry helpers for letter-shaped scan regions on coarse decision grids."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label


def _candidate_font_paths() -> Iterable[str]:
    """Yield common font paths/names that are likely to exist on Windows."""
    windir = Path.home().drive + "\\Windows\\Fonts"
    yield str(Path(windir) / "arialbd.ttf")
    yield str(Path(windir) / "arial.ttf")
    yield str(Path(windir) / "segoeuib.ttf")
    yield str(Path(windir) / "segoeui.ttf")
    yield "arialbd.ttf"
    yield "arial.ttf"
    yield "DejaVuSans-Bold.ttf"
    yield "DejaVuSans.ttf"


def _load_font(canvas_size: int, text: str, font_scale: float | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a reasonably large font for centered text rendering."""
    size = max(24, int(canvas_size * (font_scale if font_scale is not None else 0.55)))
    image = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(image)

    for font_path in _candidate_font_paths():
        try:
            font = ImageFont.truetype(font_path, size=size)
        except OSError:
            continue

        while size >= 24:
            bbox = draw.textbbox((0, 0), text, font=font, stroke_width=max(1, size // 40))
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width <= int(canvas_size * 0.9) and height <= int(canvas_size * 0.7):
                return font
            size = max(24, int(size * 0.92))
            font = ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()


def generate_text_mask(text: str, canvas_size: int = 1024, font_scale: float | None = None) -> np.ndarray:
    """Return a centered binary mask whose True region is the letter interior."""
    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    if not text.strip():
        raise ValueError("text must not be empty")

    image = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(image)
    font = _load_font(canvas_size=canvas_size, text=text, font_scale=font_scale)

    size = getattr(font, "size", max(24, int(canvas_size * 0.55)))
    stroke_width = max(1, size // 40)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (canvas_size - width) / 2 - bbox[0]
    y = (canvas_size - height) / 2 - bbox[1]

    draw.text((x, y), text, fill=255, font=font, stroke_width=stroke_width, stroke_fill=255)
    return np.asarray(image) > 0


def _extract_foreground(mask: np.ndarray) -> np.ndarray:
    """Crop a binary mask to the tight bounding box of its foreground."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("mask must contain at least one foreground pixel")

    row_min = int(rows.min())
    row_max = int(rows.max()) + 1
    col_min = int(cols.min())
    col_max = int(cols.max()) + 1
    return mask[row_min:row_max, col_min:col_max]


def _square_pad(mask: np.ndarray, pad_fraction: float = 0.08) -> np.ndarray:
    """Pad a cropped mask onto a square canvas with a small safety margin."""
    height, width = mask.shape
    side = max(height, width)
    pad = max(2, int(round(side * pad_fraction)))
    square_side = side + 2 * pad
    square = np.zeros((square_side, square_side), dtype=bool)

    row_offset = (square_side - height) // 2
    col_offset = (square_side - width) // 2
    square[row_offset : row_offset + height, col_offset : col_offset + width] = mask
    return square


def downsample_mask(mask: np.ndarray, grid_size: int = 64, threshold: float = 0.2) -> np.ndarray:
    """Convert a high-resolution letter mask into a coarse legal-scan grid."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    binary_mask = np.asarray(mask, dtype=bool)
    foreground = _extract_foreground(binary_mask)
    padded = _square_pad(foreground)
    image = Image.fromarray(padded.astype(np.uint8) * 255, mode="L")
    resized = image.resize((grid_size, grid_size), Image.Resampling.BOX)
    coarse = np.asarray(resized, dtype=np.float32) / 255.0
    return coarse >= threshold


def render_mask_preview(mask: np.ndarray, scale: int = 8) -> np.ndarray:
    """Upscale a binary grid for quick preview or image export."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    preview = np.kron((mask > 0).astype(np.uint8), np.ones((scale, scale), dtype=np.uint8))
    return preview * 255


def generate_vertical_stripes_in_component(component_mask: np.ndarray, stripe_width: int = 1) -> list[np.ndarray]:
    """Split one connected component into column-wise stripe segments."""
    if stripe_width <= 0:
        raise ValueError("stripe_width must be positive")

    component = np.asarray(component_mask, dtype=bool)
    if component.ndim != 2:
        raise ValueError("component_mask must be a 2D array")

    stripes: list[np.ndarray] = []
    height, width = component.shape
    for col_start in range(0, width, stripe_width):
        col_end = min(width, col_start + stripe_width)
        stripe_slice = component[:, col_start:col_end]
        if not stripe_slice.any():
            continue

        labeled_slice, num_segments = label(stripe_slice)
        for segment_id in range(1, num_segments + 1):
            stripe = np.zeros((height, width), dtype=bool)
            stripe[:, col_start:col_end] = labeled_slice == segment_id
            if stripe.any():
                stripes.append(stripe)

    return stripes


def generate_stripe_segments(
    mask: np.ndarray,
    grid_size: int = 64,
    stripe_width: int = 1,
    threshold: float = 0.2,
) -> list[np.ndarray]:
    """Generate legal vertical stripe segments fully contained inside the letter region."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 2:
        raise ValueError("mask must be a 2D array")

    if mask_array.shape != (grid_size, grid_size):
        coarse_mask = downsample_mask(mask_array, grid_size=grid_size, threshold=threshold)
    else:
        coarse_mask = mask_array

    labeled_mask, num_components = label(coarse_mask)
    all_stripes: list[np.ndarray] = []
    for component_id in range(1, num_components + 1):
        component_mask = labeled_mask == component_id
        all_stripes.extend(generate_vertical_stripes_in_component(component_mask, stripe_width=stripe_width))

    return all_stripes
