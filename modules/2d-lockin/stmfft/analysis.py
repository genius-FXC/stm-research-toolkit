from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MaskSpec:
    qx: float
    qy: float
    radius: float
    shape: str = "gaussian"
    include_conjugate: bool = True
    enabled: bool = True
    label: str = "Q"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "MaskSpec":
        return cls(**value)


@dataclass
class FFTResult:
    spectrum: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    dx_nm: float
    dy_nm: float
    window: str


@dataclass
class AnalysisState:
    source_path: str = ""
    channel_index: int = 0
    crop: Optional[Tuple[int, int, int, int]] = None
    preprocess: str = "plane"
    window: str = "hann"
    masks: list[MaskSpec] = field(default_factory=list)
    bragg_g1: Optional[Tuple[float, float]] = None
    bragg_g2: Optional[Tuple[float, float]] = None


@dataclass
class CompositeAmplitude:
    first: np.ndarray
    second: np.ndarray
    combined: np.ndarray
    first_scale: float
    second_scale: float
    normalized: bool
    method: str


def combine_local_amplitudes(
    first: np.ndarray,
    second: np.ndarray,
    method: str = "harmonic",
    normalize: bool = True,
) -> CompositeAmplitude:
    """Combine two *magnitudes*, never complex signals at different Q vectors.

    P90 normalization makes the map dimensionless and prevents the brighter
    Fourier peak from dominating solely because of its global intensity.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("两个局域振幅图必须是相同尺寸的二维数组。")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(a < 0) or np.any(b < 0):
        raise ValueError("局域振幅图必须是有限的非负数。")
    if method not in {"harmonic", "geometric", "rms", "minimum"}:
        raise ValueError(f"未知合成振幅方法：{method}")

    first_scale = float(np.percentile(a, 90)) if normalize and a.size else 1.0
    second_scale = float(np.percentile(b, 90)) if normalize and b.size else 1.0
    first_scale = first_scale if first_scale > 0 else 1.0
    second_scale = second_scale if second_scale > 0 else 1.0
    a_scaled, b_scaled = a / first_scale, b / second_scale
    if method == "harmonic":
        combined = np.divide(
            2.0 * a_scaled * b_scaled,
            a_scaled + b_scaled,
            out=np.zeros_like(a_scaled),
            where=(a_scaled + b_scaled) > 0,
        )
    elif method == "geometric":
        combined = np.sqrt(a_scaled * b_scaled)
    elif method == "rms":
        combined = np.hypot(a_scaled, b_scaled) / np.sqrt(2.0)
    else:
        combined = np.minimum(a_scaled, b_scaled)
    return CompositeAmplitude(a_scaled, b_scaled, combined, first_scale, second_scale, normalize, method)


def sanitize_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    array = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(array)
    fraction = float(np.mean(finite))
    if not np.any(finite):
        raise ValueError("当前通道没有有限数值。")
    fill = float(np.nanmedian(np.where(finite, array, np.nan)))
    return np.where(finite, array, fill), fraction


def subtract_plane(image: np.ndarray, robust: bool = True) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    ny, nx = image.shape
    step = max(1, min(nx, ny) // 256)
    yy, xx = np.mgrid[0:ny:step, 0:nx:step]
    values = image[::step, ::step]
    good = np.isfinite(values)
    if robust and np.any(good):
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median)))
        if mad > 0:
            good &= np.abs(values - median) < 8.0 * 1.4826 * mad
    design = np.column_stack((xx[good], yy[good], np.ones(np.count_nonzero(good))))
    if design.shape[0] < 3:
        return image - np.nanmean(image)
    coefficients, *_ = np.linalg.lstsq(design, values[good], rcond=None)
    full_y, full_x = np.mgrid[0:ny, 0:nx]
    return image - (coefficients[0] * full_x + coefficients[1] * full_y + coefficients[2])


def flatten_rows(image: np.ndarray, linear: bool = False) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    ny, nx = source.shape
    x = np.linspace(-1.0, 1.0, nx)
    output = np.empty_like(source)
    for row_index in range(ny):
        row = source[row_index]
        median = float(np.median(row))
        mad = float(np.median(np.abs(row - median)))
        good = np.ones(nx, dtype=bool)
        if mad > 0:
            good = np.abs(row - median) < 6.0 * 1.4826 * mad
        if linear and np.count_nonzero(good) >= 3:
            coefficients = np.polyfit(x[good], row[good], 1)
            output[row_index] = row - np.polyval(coefficients, x)
        else:
            output[row_index] = row - np.median(row[good])
    return output


def preprocess_image(image: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    clean, finite_fraction = sanitize_image(image)
    if method == "none":
        result = clean.copy()
    elif method == "mean":
        result = clean - np.mean(clean)
    elif method == "plane":
        result = subtract_plane(clean)
    elif method == "row_median":
        result = flatten_rows(subtract_plane(clean), linear=False)
    elif method == "row_linear":
        result = flatten_rows(subtract_plane(clean), linear=True)
    else:
        raise ValueError(f"未知预处理方法：{method}")
    return result, finite_fraction


def tukey_window(length: int, alpha: float = 0.35) -> np.ndarray:
    if length <= 1:
        return np.ones(length)
    x = np.linspace(0.0, 1.0, length)
    window = np.ones(length)
    first = x < alpha / 2
    last = x >= 1 - alpha / 2
    window[first] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[first] - alpha / 2)))
    window[last] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[last] - 1 + alpha / 2)))
    return window


def window_2d(shape: tuple[int, int], name: str) -> np.ndarray:
    ny, nx = shape
    if name == "none":
        return np.ones(shape)
    if name == "hann":
        wy, wx = np.hanning(ny), np.hanning(nx)
    elif name == "tukey":
        wy, wx = tukey_window(ny), tukey_window(nx)
    else:
        raise ValueError(f"未知窗函数：{name}")
    return np.outer(wy, wx)


def compute_fft(image: np.ndarray, scan_width_nm: float, scan_height_nm: float, window: str = "hann") -> FFTResult:
    ny, nx = image.shape
    if nx < 2 or ny < 2:
        raise ValueError("图像至少需要 2×2 像素。")
    dx_nm = scan_width_nm / nx
    dy_nm = scan_height_nm / ny
    weighted = (image - np.mean(image)) * window_2d(image.shape, window)
    spectrum = np.fft.fftshift(np.fft.fft2(weighted))
    qx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_nm))
    qy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy_nm))
    return FFTResult(spectrum=spectrum, qx=qx, qy=qy, dx_nm=dx_nm, dy_nm=dy_nm, window=window)


def _single_mask(qxx: np.ndarray, qyy: np.ndarray, spec: MaskSpec, qx: float, qy: float) -> np.ndarray:
    distance = np.hypot(qxx - qx, qyy - qy)
    radius = max(float(spec.radius), np.finfo(float).eps)
    if spec.shape == "hard":
        return (distance <= radius).astype(np.float64)
    if spec.shape == "cosine":
        scaled = np.clip(distance / radius, 0.0, 1.0)
        result = 0.5 * (1.0 + np.cos(np.pi * scaled))
        result[distance > radius] = 0.0
        return result
    if spec.shape == "gaussian":
        sigma = radius / 2.0
        return np.exp(-0.5 * (distance / sigma) ** 2)
    raise ValueError(f"未知 mask 形状：{spec.shape}")


def make_frequency_mask(qx: np.ndarray, qy: np.ndarray, specs: Iterable[MaskSpec]) -> np.ndarray:
    qxx, qyy = np.meshgrid(qx, qy)
    output = np.zeros(qxx.shape, dtype=np.float64)
    for spec in specs:
        if not spec.enabled:
            continue
        current = _single_mask(qxx, qyy, spec, spec.qx, spec.qy)
        if spec.include_conjugate:
            current = np.maximum(current, _single_mask(qxx, qyy, spec, -spec.qx, -spec.qy))
        output = np.maximum(output, current)
    return np.clip(output, 0.0, 1.0)


def reconstruct_from_mask(spectrum: np.ndarray, mask: np.ndarray) -> np.ndarray:
    reconstructed = np.fft.ifft2(np.fft.ifftshift(spectrum * mask))
    return np.real(reconstructed)


def local_complex_component(spectrum: np.ndarray, qx: np.ndarray, qy: np.ndarray, spec: MaskSpec) -> np.ndarray:
    one_sided = MaskSpec(
        qx=spec.qx,
        qy=spec.qy,
        radius=spec.radius,
        shape=spec.shape,
        include_conjugate=False,
    )
    mask = make_frequency_mask(qx, qy, [one_sided])
    analytic_signal = 2.0 * np.fft.ifft2(np.fft.ifftshift(spectrum * mask))

    # Remove the selected carrier exp(+i 2π Q·r), leaving the slowly varying
    # complex envelope. Its magnitude is the local component amplitude and its
    # angle is the phase relative to the selected Q and the current crop origin.
    ny, nx = analytic_signal.shape
    dqx = abs(float(qx[1] - qx[0])) if nx > 1 else 1.0
    dqy = abs(float(qy[1] - qy[0])) if ny > 1 else 1.0
    dx = 1.0 / (nx * dqx)
    dy = 1.0 / (ny * dqy)
    yy, xx = np.meshgrid(np.arange(ny) * dy, np.arange(nx) * dx, indexing="ij")
    carrier = np.exp(-2j * np.pi * (spec.qx * xx + spec.qy * yy))
    return analytic_signal * carrier


def phase_histogram(
    component: np.ndarray,
    amplitude_floor_percentile: float = 15.0,
    bins: int = 36,
    amplitude_weighted: bool = False,
) -> dict:
    """Histogram and circular statistics for a complex local component.

    Pixels below the requested amplitude percentile are excluded because their
    phase is poorly defined.  Circular statistics are essential here: -pi and
    +pi describe the same direction and must not be averaged linearly.
    """
    values = np.asarray(component, dtype=np.complex128)
    amplitude = np.abs(values)
    finite = np.isfinite(amplitude) & np.isfinite(np.real(values)) & np.isfinite(np.imag(values))
    if not np.any(finite):
        raise ValueError("复包络中没有有限像素。")
    percentile = float(np.clip(amplitude_floor_percentile, 0.0, 100.0))
    threshold = float(np.percentile(amplitude[finite], percentile))
    valid = finite & (amplitude >= threshold)
    phases = np.angle(values[valid])
    weights = amplitude[valid] if amplitude_weighted else np.ones(phases.shape, dtype=float)
    bin_count = max(4, int(bins))
    histogram, edges = np.histogram(phases, bins=bin_count, range=(-np.pi, np.pi), weights=weights)
    total_weight = float(np.sum(weights))
    if total_weight <= 0 or phases.size == 0:
        mean_phase = float("nan")
        resultant_length = float("nan")
        circular_std = float("nan")
    else:
        resultant = np.sum(weights * np.exp(1j * phases)) / total_weight
        mean_phase = float(np.angle(resultant))
        resultant_length = float(np.clip(np.abs(resultant), 0.0, 1.0))
        circular_std = float(np.sqrt(max(0.0, -2.0 * np.log(max(resultant_length, np.finfo(float).tiny)))))
    return {
        "histogram": histogram.astype(float),
        "edges": edges,
        "threshold": threshold,
        "pixel_count": int(phases.size),
        "mean_phase": mean_phase,
        "resultant_length": resultant_length,
        "circular_std": circular_std,
        "amplitude_weighted": bool(amplitude_weighted),
    }


def fractional_coordinates(
    q: Sequence[float],
    g1: Sequence[float],
    g2: Sequence[float],
) -> tuple[float, float]:
    basis = np.array([[g1[0], g2[0]], [g1[1], g2[1]]], dtype=float)
    if abs(np.linalg.det(basis)) < 1e-12:
        raise ValueError("G1 与 G2 共线，不能建立二维倒格坐标。")
    result = np.linalg.solve(basis, np.asarray(q, dtype=float))
    return float(result[0]), float(result[1])


def c4x4_allowed_points(limit: int = 4) -> list[tuple[float, float, int, int]]:
    points: list[tuple[float, float, int, int]] = []
    for h in range(-limit, limit + 1):
        for k in range(-limit, limit + 1):
            if h == 0 and k == 0:
                continue
            if (h + k) % 2 == 0:
                points.append((h / 4.0, k / 4.0, h, k))
    return points
