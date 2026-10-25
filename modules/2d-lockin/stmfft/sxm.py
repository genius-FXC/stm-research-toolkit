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

    @property
    def label(self) -> str:
        unit = f" [{self.unit}]" if self.unit else ""
        return f"{self.name} — {self.direction}{unit}"


@dataclass
class SxmFile:
    path: Path
    header: str
    header_fields: Dict[str, List[str]]
    scan_pixels: Tuple[int, int]
    scan_range_m: Optional[Tuple[float, float]]
    channels: List[SxmChannel]

    @property
    def scan_range_nm(self) -> Optional[Tuple[float, float]]:
        if self.scan_range_m is None:
            return None
        return self.scan_range_m[0] * 1e9, self.scan_range_m[1] * 1e9


def _find_header_end(raw: bytes) -> int:
    marker = b":SCANIT_END:"
    position = raw.find(marker)
    if position < 0:
        raise ValueError("没有找到 Nanonis SXM 的 :SCANIT_END: 标记。")
    position += len(marker)
    while position < len(raw) and raw[position] in b"\r\n\t ":
        position += 1
    if raw[position : position + 2] == b"\x1a\x04":
        position += 2
    while position < len(raw) and raw[position] in b"\r\n\t ":
        position += 1
    return position


def _parse_header_fields(header: str) -> Dict[str, List[str]]:
    fields: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw_line in header.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^:([^:]+):\s*(.*)$", line)
        if match:
            current = match.group(1).strip().upper()
            fields.setdefault(current, [])
            remainder = match.group(2).strip()
            if remainder:
                fields[current].append(remainder)
        elif current:
            fields[current].append(line)
    return fields


def _numbers(fields: Dict[str, List[str]], key: str) -> List[float]:
    text = " ".join(fields.get(key.upper(), []))
    return [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def _parse_channels(fields: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
    definitions: List[Tuple[str, str, str]] = []
    for line in fields.get("DATA_INFO", []):
        parts = [part.strip() for part in re.split(r"\t+|\s{2,}", line) if part.strip()]
        if not parts or parts[0].lower() in {"channel", "index", "name"}:
            continue
        if parts[0].isdigit() and len(parts) >= 4:
            name, unit, direction = parts[1], parts[2], parts[3]
        elif len(parts) >= 3:
            name, unit, direction = parts[0], parts[1], parts[2]
        else:
            continue
        definitions.append((name, unit, direction.lower()))
    return definitions


def read_sxm(path: str | Path) -> SxmFile:
    path = Path(path)
    raw = path.read_bytes()
    data_start = _find_header_end(raw)
    header = raw[:data_start].decode("latin1", errors="replace")
    fields = _parse_header_fields(header)

    pixels = _numbers(fields, "SCAN_PIXELS")
    if len(pixels) < 2:
        raise ValueError("SXM 头中没有有效的 SCAN_PIXELS。")
    nx, ny = int(pixels[0]), int(pixels[1])

    ranges = _numbers(fields, "SCAN_RANGE")
    scan_range = (ranges[0], ranges[1]) if len(ranges) >= 2 else None
    definitions = _parse_channels(fields)
    if not definitions:
        raise ValueError("SXM 头中没有解析到 DATA_INFO 通道表。")

    slots: List[Tuple[str, str, str]] = []
    for name, unit, direction in definitions:
        if direction in {"both", "both directions", "forward/backward"}:
            slots.extend([(name, unit, "forward"), (name, unit, "backward")])
        else:
            slots.append((name, unit, direction or "unknown"))

    pixels_per_channel = nx * ny
    available = (len(raw) - data_start) // (pixels_per_channel * 4)
    if available <= 0:
        raise ValueError("SXM 头后没有找到 float32 图像数据。")
    if available < len(slots):
        slots = slots[:available]
    elif available > len(slots):
        for index in range(len(slots), available):
            slots.append((f"channel_{index + 1}", "", "unknown"))

    count = len(slots) * pixels_per_channel
    values = np.frombuffer(raw, dtype=">f4", count=count, offset=data_start)
    if values.size != count:
        raise ValueError("SXM 图像数据长度小于头信息给出的尺寸。")

    channels: List[SxmChannel] = []
    for index, (name, unit, direction) in enumerate(slots):
        start = index * pixels_per_channel
        image = values[start : start + pixels_per_channel].reshape(ny, nx).astype(np.float64)
        # Nanonis stores the backward trace in acquisition order. Reverse its
        # fast-scan axis so forward and backward images share physical x,y.
        if direction == "backward":
            image = np.fliplr(image).copy()
        channels.append(SxmChannel(name=name, unit=unit, direction=direction, data=image))

    return SxmFile(
        path=path,
        header=header,
        header_fields=fields,
        scan_pixels=(nx, ny),
        scan_range_m=scan_range,
        channels=channels,
    )
