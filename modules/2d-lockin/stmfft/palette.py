from __future__ import annotations

from typing import Iterable

import configparser
from pathlib import Path
import numpy as np
from PIL import Image


ANCHORS = {
    "gray": [
        (0.00, (0, 0, 0)),
        (1.00, (255, 255, 255)),
    ],
    "copper": [
        (0.00, (10, 3, 0)),
        (0.25, (92, 27, 5)),
        (0.55, (185, 79, 18)),
        (0.80, (244, 166, 69)),
        (1.00, (255, 246, 196)),
    ],
    "inferno": [
        (0.00, (0, 0, 4)),
        (0.18, (50, 10, 92)),
        (0.40, (135, 28, 91)),
        (0.62, (211, 73, 55)),
        (0.82, (248, 142, 8)),
        (1.00, (252, 255, 164)),
    ],
    "viridis": [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ],
    "phase": [
        (0.00, (32, 75, 186)),
        (0.17, (70, 190, 210)),
        (0.33, (120, 220, 100)),
        (0.50, (245, 232, 70)),
        (0.67, (240, 115, 45)),
        (0.83, (194, 50, 120)),
        (1.00, (32, 75, 186)),
    ],
}


def make_lut(name: str, size: int = 256) -> np.ndarray:
    if name == "nanox":
        ini=configparser.ConfigParser();ini.read(Path(__file__).resolve().parents[1]/'assets/Nanox.cpal')
        ids=[int(v) for v in ini['ramp']['indexes'].split(',')]
        hexes=ini['ramp']['colors'].split(',')[2:]
        rgb=np.array([[int(h.zfill(6)[j:j+2],16) for j in (0,2,4)] for h in hexes])
        return np.rint(np.array([np.interp(np.linspace(0,252,size),ids,rgb[:,c]) for c in range(3)]).T).astype(np.uint8)
    anchors = ANCHORS.get(name, ANCHORS["gray"])
    positions = np.array([item[0] for item in anchors])
    colors = np.array([item[1] for item in anchors], dtype=float)
    x = np.linspace(0.0, 1.0, size)
    channels = [np.interp(x, positions, colors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=1).clip(0, 255).astype(np.uint8)


def robust_limits(array: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    finite = np.asarray(array)[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    lower, upper = np.percentile(finite, [low, high])
    if upper <= lower:
        upper = lower + max(abs(lower) * 1e-12, np.finfo(float).tiny)
    return float(lower), float(upper)


def colorize(
    array: np.ndarray,
    palette: str = "gray",
    low: float = 1.0,
    high: float = 99.0,
    limits: tuple[float, float] | None = None,
) -> Image.Image:
    values = np.asarray(array, dtype=np.float64)
    if limits is None:
        lower, upper = robust_limits(values, low, high)
    else:
        lower, upper = limits
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    indices = np.rint(normalized * 255).astype(np.uint8)
    rgb = make_lut(palette)[indices]
    return Image.fromarray(rgb, mode="RGB")
