from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SxmChannel:
    name: str
    unit: str
    direction: str
    data: np.ndarray


@dataclass
class SxmFile:
    path: Path
    header: str
    header_fields: Dict[str, List[str]]
    scan_pixels: Tuple[int, int]
    scan_range: Optional[Tuple[float, float]]
    channels: List[SxmChannel]


def _find_header_end(raw: bytes) -> int:
    markers = [b":SCANIT_END:", b"\x1a\x04"]
    hits = [raw.find(marker) + len(marker) for marker in markers if raw.find(marker) >= 0]
    if not hits:
        raise ValueError("Could not find SXM header terminator (:SCANIT_END:).")
    return min(hits)


def _parse_header_fields(header: str) -> Dict[str, List[str]]:
    fields: Dict[str, List[str]] = {}
    current_key: Optional[str] = None
    for raw_line in header.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^:([^:]+):\s*(.*)$", line)
        if match:
            current_key = match.group(1).strip().upper()
            fields.setdefault(current_key, [])
            rest = match.group(2).strip()
            if rest:
                fields[current_key].append(rest)
            continue
        if current_key:
            fields[current_key].append(line)
    return fields


def _numbers_from_field(fields: Dict[str, List[str]], key: str) -> List[float]:
    values = " ".join(fields.get(key.upper(), []))
    return [float(item) for item in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", values)]


def _parse_data_info(fields: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
    rows = fields.get("DATA_INFO", [])
    parsed: List[Tuple[str, str, str]] = []
    for line in rows:
        parts = [part.strip() for part in re.split(r"\t+|\s{2,}", line) if part.strip()]
        if len(parts) < 3:
            continue
        if parts[0].lower() in {"channel", "index", "name"}:
            continue
        if parts[0].isdigit() and len(parts) >= 4:
            name, unit, direction = parts[1], parts[2], parts[3]
        else:
            name, unit, direction = parts[0], parts[1], parts[2]
        parsed.append((name, unit, direction.lower()))
    return parsed


def _candidate_channel_names(fields: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
    info = _parse_data_info(fields)
    if info:
        return info
    names = fields.get("DATA_INFO", []) or fields.get("CHANNELS", [])
    out = []
    for name in names:
        clean = name.strip()
        if clean:
            out.append((clean, "", "both"))
    return out


def read_sxm(path: str | Path) -> SxmFile:
    path = Path(path)
    raw = path.read_bytes()
    header_end = _find_header_end(raw)
    marker_offset = raw.find(b"\x1a\x04", header_end)
    if marker_offset < 0:
        raise ValueError("Missing SXM binary start marker.")
    data_start = marker_offset + 2

    header = raw[:header_end].decode("utf-8", errors="replace")
    fields = _parse_header_fields(header)

    pixels = _numbers_from_field(fields, "SCAN_PIXELS")
    if len(pixels) < 2:
        raise ValueError("Could not parse SCAN_PIXELS from SXM header.")
    nx, ny = int(pixels[0]), int(pixels[1])

    ranges = _numbers_from_field(fields, "SCAN_RANGE")
    scan_range = (ranges[0], ranges[1]) if len(ranges) >= 2 else None

    channel_defs = _candidate_channel_names(fields)
    if not channel_defs:
        raise ValueError("Could not parse channel list from SXM header DATA_INFO.")

    expected_pixels = nx * ny
    binary = raw[data_start:]
    available_arrays = len(binary) // (expected_pixels * 4)
    if available_arrays <= 0:
        raise ValueError("No float32 image data found after SXM header.")

    direction_slots: List[Tuple[str, str, str]] = []
    for name, unit, direction in channel_defs:
        if direction in {"both", "both directions", "forward/backward"}:
            direction_slots.append((name, unit, "forward"))
            direction_slots.append((name, unit, "backward"))
        else:
            direction_slots.append((name, unit, direction or "unknown"))

    if len(direction_slots) > available_arrays:
        direction_slots = direction_slots[:available_arrays]
    elif len(direction_slots) < available_arrays:
        for idx in range(len(direction_slots), available_arrays):
            direction_slots.append((f"channel_{idx + 1}", "", "unknown"))

    data_bytes = expected_pixels * 4 * len(direction_slots)
    data = np.frombuffer(binary[:data_bytes], dtype=">f4")
    if data.size != expected_pixels * len(direction_slots):
        raise ValueError("SXM binary data is shorter than expected for parsed channels.")

    channels = []
    for idx, (name, unit, direction) in enumerate(direction_slots):
        arr = data[idx * expected_pixels : (idx + 1) * expected_pixels].reshape((ny, nx))
        channels.append(SxmChannel(name=name, unit=unit, direction=direction, data=arr.astype(np.float64)))

    return SxmFile(
        path=path,
        header=header,
        header_fields=fields,
        scan_pixels=(nx, ny),
        scan_range=scan_range,
        channels=channels,
    )


def choose_topography_channel(
    sxm: SxmFile,
    channel_name: Optional[str] = None,
    direction: Optional[str] = None,
) -> SxmChannel:
    candidates = sxm.channels
    if channel_name:
        lowered = channel_name.lower()
        candidates = [ch for ch in candidates if lowered in ch.name.lower()]
        if not candidates:
            raise ValueError(f"No SXM channel matches {channel_name!r}.")
    if direction:
        lowered_dir = direction.lower()
        direction_matches = [ch for ch in candidates if lowered_dir in ch.direction.lower()]
        if direction_matches:
            candidates = direction_matches

    def score(ch: SxmChannel) -> int:
        name = ch.name.lower()
        unit = ch.unit.lower()
        value = 0
        if any(key in name for key in ["z", "height", "topo", "topography"]):
            value += 10
        if any(key in unit for key in ["m", "nm", "pm"]):
            value += 3
        if any(key in name for key in ["current", "curr", "i ", "itunnel", "tunnel"]):
            value -= 8
        if "forward" in ch.direction.lower():
            value += 1
        return value

    return sorted(candidates, key=score, reverse=True)[0]
