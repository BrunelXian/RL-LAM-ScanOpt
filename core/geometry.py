"""Geometry helpers for Stage A grid masks and simple line-based benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class LDEDTrack:
    """One line-based LDED deposition track."""

    track_id: int
    x_start_mm: float
    x_end_mm: float
    y_start_mm: float
    y_end_mm: float
    x_center_mm: float
    y_center_mm: float
    width_mm: float
    length_mm: float
    direction: str = "bottom_to_top"


@dataclass(frozen=True)
class LDEDCouponBenchmark:
    """Compact benchmark description for line-based track-order experiments."""

    benchmark_name: str
    target_name: str
    plane_width_mm: float
    plane_height_mm: float
    patch_x_min_mm: float
    patch_x_max_mm: float
    patch_y_min_mm: float
    patch_y_max_mm: float
    margin_left_mm: float
    margin_right_mm: float
    margin_top_mm: float
    margin_bottom_mm: float
    track_count: int
    track_width_mm: float
    track_length_mm: float
    track_pitch_mm: float
    layer_count: int
    tracks: tuple[LDEDTrack, ...]

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return the logical benchmark shape as (layers, tracks)."""
        return (self.layer_count, self.track_count)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable benchmark payload."""
        payload = asdict(self)
        payload["tracks"] = [asdict(track) for track in self.tracks]
        payload["grid_shape"] = list(self.grid_shape)
        return payload


def build_lded_coupon_32track_v1() -> LDEDCouponBenchmark:
    """Return the first LDED line-order benchmark used for FEA-teacher planning.

    Geometry:
    - plane: 100 mm x 40 mm
    - deposited patch: 96 mm x 36 mm
    - margins: 2 mm on all sides
    - 32 vertical tracks
    - track width/pitch: 3 mm
    - track length: 36 mm
    - one layer, fixed bottom-to-top scan direction
    """
    benchmark_name = "lded_coupon_32track_v1"
    plane_width_mm = 100.0
    plane_height_mm = 40.0
    margin_mm = 2.0
    patch_x_min_mm = margin_mm
    patch_x_max_mm = plane_width_mm - margin_mm
    patch_y_min_mm = margin_mm
    patch_y_max_mm = plane_height_mm - margin_mm
    track_count = 32
    track_width_mm = 3.0
    track_pitch_mm = 3.0
    track_length_mm = 36.0
    layer_count = 1

    tracks: list[LDEDTrack] = []
    for track_id in range(track_count):
        x_start_mm = patch_x_min_mm + track_id * track_pitch_mm
        x_end_mm = x_start_mm + track_width_mm
        x_center_mm = 0.5 * (x_start_mm + x_end_mm)
        y_start_mm = patch_y_min_mm
        y_end_mm = patch_y_max_mm
        y_center_mm = 0.5 * (y_start_mm + y_end_mm)
        tracks.append(
            LDEDTrack(
                track_id=track_id,
                x_start_mm=x_start_mm,
                x_end_mm=x_end_mm,
                y_start_mm=y_start_mm,
                y_end_mm=y_end_mm,
                x_center_mm=x_center_mm,
                y_center_mm=y_center_mm,
                width_mm=track_width_mm,
                length_mm=track_length_mm,
            )
        )

    return LDEDCouponBenchmark(
        benchmark_name=benchmark_name,
        target_name=benchmark_name,
        plane_width_mm=plane_width_mm,
        plane_height_mm=plane_height_mm,
        patch_x_min_mm=patch_x_min_mm,
        patch_x_max_mm=patch_x_max_mm,
        patch_y_min_mm=patch_y_min_mm,
        patch_y_max_mm=patch_y_max_mm,
        margin_left_mm=margin_mm,
        margin_right_mm=margin_mm,
        margin_top_mm=margin_mm,
        margin_bottom_mm=margin_mm,
        track_count=track_count,
        track_width_mm=track_width_mm,
        track_length_mm=track_length_mm,
        track_pitch_mm=track_pitch_mm,
        layer_count=layer_count,
        tracks=tuple(tracks),
    )


def _interleave_sequences(left: list[int], right: list[int]) -> list[int]:
    """Interleave two equal-length or nearly equal-length integer sequences."""
    result: list[int] = []
    for left_item, right_item in zip(left, right):
        result.append(left_item)
        result.append(right_item)
    if len(left) > len(right):
        result.extend(left[len(right) :])
    elif len(right) > len(left):
        result.extend(right[len(left) :])
    return result


def build_lded_coupon_32track_baselines(random_seeds: Iterable[int] = (0, 7, 13, 29)) -> dict[str, list[int]]:
    """Return deterministic line-order benchmark baselines as track-id sequences."""
    benchmark = build_lded_coupon_32track_v1()
    indices = list(range(benchmark.track_count))
    middle_left = benchmark.track_count // 2 - 1
    middle_right = benchmark.track_count // 2

    center_out: list[int] = []
    for offset in range(benchmark.track_count // 2):
        left_index = middle_left - offset
        right_index = middle_right + offset
        center_out.extend([left_index, right_index])

    edge_in = _interleave_sequences(
        list(range(0, benchmark.track_count // 2)),
        list(range(benchmark.track_count - 1, benchmark.track_count // 2 - 1, -1)),
    )

    baselines: dict[str, list[int]] = {
        "raster_left_to_right": [int(track_id) for track_id in indices],
        "raster_right_to_left": [int(track_id) for track_id in reversed(indices)],
        "center_out": [int(track_id) for track_id in center_out],
        "edge_in": [int(track_id) for track_id in edge_in],
        "odd_even_interlaced": [int(track_id) for track_id in (indices[::2] + indices[1::2])],
        "even_odd_interlaced": [int(track_id) for track_id in (indices[1::2] + indices[::2])],
    }

    for seed in random_seeds:
        rng = np.random.default_rng(int(seed))
        baselines[f"random_seed_{int(seed)}"] = [int(track_id) for track_id in rng.permutation(indices)]

    return baselines
