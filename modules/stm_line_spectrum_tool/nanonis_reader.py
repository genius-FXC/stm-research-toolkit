"""Export raw LI Demod 1 X bias slices and Z from a linear Nanonis grid.

Only one recorded spatial direction exists; do not invent a backward image.
Nonlinear sweeps and incomplete grids are rejected rather than interpolated.
Dependencies: numpy, Pillow. Source data are never modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np


def read_grid(path: Path) -> dict:
    raw = path.read_bytes()
    marker = b":HEADER_END:"
    stop = raw.index(marker) + len(marker)
    offset = stop
    # Consume ONLY the textual terminator, never whitespace in binary floats.
    for ending in (b"\r\r\n", b"\r\n", b"\n"):
        if raw[offset:offset + len(ending)] == ending:
            offset += len(ending)
            break
    else:
        raise ValueError("Unsupported 3DS header line ending.")
    header = raw[:stop].decode("latin1")
    fields = {}
    for line in header.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip().strip('"')
    nx, ny = map(int, re.fullmatch(r"(\d+) x (\d+)", fields["Grid dim"]).groups())
    npts = int(fields["Points"])
    npar = int(fields["# Parameters (4 byte)"])
    channels = fields["Channels"].split(";")
    params = fields["Fixed parameters"].split(";") + fields["Experiment parameters"].split(";")
    if len(params) != npar:
        raise ValueError("Parameter names and parameter count disagree.")
    if fields["Filetype"].lower() != "linear":
        raise ValueError("This exporter requires a verified linear bias sweep; MLS is not approximated.")
    if fields["Sweep Signal"] != "Bias (V)":
        raise ValueError("The swept quantity is not Bias (V).")
    if "LI Demod 1 X (A)" not in channels or "Z (m)" not in params:
        raise ValueError("Required LI Demod 1 X (A) or Z (m) data are absent.")
    stride = npar + len(channels) * npts
    payload = raw[offset:]
    expected = nx * ny * stride * 4
    if len(payload) != expected:
        raise ValueError(f"Incomplete/unsupported payload: {len(payload)} bytes, expected {expected}.")
    if int(fields["Experiment size (bytes)"]) != len(channels) * npts * 4:
        raise ValueError("Spectral payload size does not match the header.")
    records = np.frombuffer(payload, dtype=">f4").reshape(ny, nx, stride)
    ip0, ip1 = params.index("Sweep Start"), params.index("Sweep End")
    start, end = float(records[0, 0, ip0]), float(records[0, 0, ip1])
    if not np.all(records[:, :, ip0] == start) or not np.all(records[:, :, ip1] == end):
        raise ValueError("Bias endpoints vary across the grid; a common bias map cannot be assumed.")
    if not np.isfinite(start + end) or npts < 2 or start == end:
        raise ValueError("Invalid bias sweep.")
    # Spectroscopy settings are optional in valid 3DS headers. The per-pixel
    # fixed parameters are already checked for a common, finite linear sweep.
    # Prefer text precision when available, but do not require those settings.
    endpoints = []
    endpoint_sources = []
    for name, recorded in [("Start", start), ("End", end)]:
        key = f"Bias Spectroscopy>Sweep {name} (V)"
        if key in fields:
            value = float(fields[key])
            if not np.isfinite(value) or not np.isclose(recorded, value, rtol=1e-6, atol=1e-9):
                raise ValueError("Binary and header bias endpoints disagree.")
            endpoint_sources.append(key)
        else:
            # Shortest decimal that round-trips to the stored float32 value:
            # 0.029999999329... becomes 0.03 without guessing a bias offset,
            # fixed decimal precision or an unrelated scan setpoint.
            value = float(str(np.float32(recorded)))
            assert np.float32(value) == np.float32(recorded)
            endpoint_sources.append(f"record fixed parameter Sweep {name}")
        endpoints.append(value)
    hstart, hend = endpoints
    bias = np.linspace(hstart, hend, npts)
    channel = channels.index("LI Demod 1 X (A)")
    lix = records[:, :, npar + channel * npts:npar + (channel + 1) * npts]
    current_name = next((name for name in ['Current (A)', 'Current [AVG] (A)'] if name in channels), None)
    current = None
    if current_name is not None:
        ci = channels.index(current_name)
        current = records[:, :, npar + ci * npts:npar + (ci + 1) * npts]
    z = records[:, :, params.index("Z (m)")]
    if not np.all(np.isfinite(lix)) or not np.all(np.isfinite(z)):
        raise ValueError("Selected maps contain invalid pixels; requires explicit incomplete-data handling.")
    settings = tuple(map(float, fields["Grid settings"].split(";")))
    if len(settings) != 5 or settings[2] <= 0 or settings[3] <= 0:
        raise ValueError("Invalid grid geometry.")
    # Confirm file ordering using recorded X/Y, including the rotated grid.
    x = records[:, :, params.index("X (m)")].astype(float)
    y = records[:, :, params.index("Y (m)")].astype(float)
    rotation = np.deg2rad(settings[4])
    u = x * np.cos(rotation) + y * np.sin(rotation)
    v = -x * np.sin(rotation) + y * np.cos(rotation)
    if nx > 1 and not np.all(np.diff(u, axis=1) > 0):
        raise ValueError("Unexpected horizontal/serpentine ordering; export stopped.")
    if ny > 1 and not np.all(np.diff(v, axis=0) > 0):
        raise ValueError("Unexpected vertical ordering; export stopped.")
    scan_z_equal = None
    if "Scan:Z (m)" in params:
        scan_z_equal = bool(np.array_equal(z, records[:, :, params.index("Scan:Z (m)")]))
    return dict(path=path, sha256=hashlib.sha256(raw).hexdigest(), header=header,
                fields=fields, nx=nx, ny=ny, npts=npts, bias=bias, lix=lix, z=z,
                settings=settings, x=x, y=y, scan_z_equal=scan_z_equal,
                npar=npar, params=params, channel_index=channel,current=current,current_channel=current_name,
                bias_endpoint_sources=endpoint_sources,
                z_parameter_index=params.index("Z (m)"), records=records)


