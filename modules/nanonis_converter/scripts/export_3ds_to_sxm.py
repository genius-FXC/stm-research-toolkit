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
    # Header endpoints avoid float32 display noise, only after checking agreement.
    hstart = float(fields["Bias Spectroscopy>Sweep Start (V)"])
    hend = float(fields["Bias Spectroscopy>Sweep End (V)"])
    if not np.allclose([start, end], [hstart, hend], rtol=1e-6, atol=1e-9):
        raise ValueError("Binary and header bias endpoints disagree.")
    bias = np.linspace(hstart, hend, npts)
    channel = channels.index("LI Demod 1 X (A)")
    lix = records[:, :, npar + channel * npts:npar + (channel + 1) * npts]
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
                npar=npar, params=params, channel_index=channel,
                z_parameter_index=params.index("Z (m)"), records=records)


def write_sxm(path: Path, grid: dict, data: np.ndarray, name: str, unit: str,
              bias_v: float, source_index: int | None) -> None:
    fields = grid["fields"]
    acquired = datetime.strptime(fields["Start time"], "%d.%m.%Y %H:%M:%S")
    temp = float(fields.get("Temperature 1>Temperature 1 (K)", "NaN"))
    cx, cy, width, height, angle = grid["settings"]
    current = float(fields.get("Current>Current (A)", "NaN"))
    comment = [
        "Derived from Nanonis grid spectroscopy; not an independently acquired scan.",
        "Source file: " + grid["path"].name,
        "Source SHA256: " + grid["sha256"],
        "Source channel: " + ("Z (m)" if source_index is None else "LI Demod 1 X (A)"),
        "Source sweep index (zero based): " + str(source_index),
        "Source spatial order preserved; scan direction up; forward image only.",
        "Raw calibrated float32 values; no flattening, filtering or normalization.",
        "BIAS records slice bias; topo BIAS records the grid setpoint bias.",
    ]
    tags = [
        ("NANONIS_VERSION", "2"),
        ("SCANIT_TYPE", "FLOAT MSBFIRST"),
        ("REC_DATE", acquired.strftime("%d.%m.%Y")),
        ("REC_TIME", acquired.strftime("%H:%M:%S")),
        ("REC_TEMP", f"{temp:.12E}"),
        ("ACQ_TIME", "N/A"),
        ("SCAN_PIXELS", f"{grid['nx']} {grid['ny']}"),
        ("SCAN_FILE", path.name),
        ("SCAN_TIME", "NaN NaN"),
        ("SCAN_RANGE", f"{width:.12E} {height:.12E}"),
        ("SCAN_OFFSET", f"{cx:.12E} {cy:.12E}"),
        ("SCAN_ANGLE", f"{angle:.12E}"),
        ("SCAN_DIR", "up"),
        ("BIAS", f"{bias_v:.12E}"),
        ("Z-CONTROLLER", "\tName\ton\tSetpoint\tP-gain\tI-gain\tT-const\n"
         f"\tlog Current\t1\t{current:.12E} A\tNaN\tNaN\tNaN"),
        ("COMMENT", "\n".join(comment)),
        # The vendor loadsxm.m reader ends this table at an EMPTY LINE,
        # not at the next tag. Omitting it consumes SCANIT_END and binary data.
        ("DATA_INFO", "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
         f"\t0\t{name}\t{unit}\tforward\t1.000000E+0\t0.000000E+0\n"),
    ]
    header = "".join(f":{key}:\n{value}\n" for key, value in tags)
    # Match native SXM files and the vendor reference loader: LF plus two
    # blank lines after the final tag. Keep Unicode provenance in sidecars;
    # escape non-ASCII filenames in the SXM's documented ASCII header.
    header += ":SCANIT_END:\n\n\n"
    with path.open("xb") as handle:
        handle.write(header.encode("ascii", errors="backslashreplace"))
        handle.write(b"\x1a\x04")
        handle.write(np.asarray(data, dtype=">f4").tobytes(order="C"))


def verify_nanonis_header(path: Path) -> None:
    """Check the vendor's blank-line-terminated DATA_INFO parsing contract.

    Reference: Nanonis SPM Controller Help, Concepts/loadsxm.m DATA_INFO
    case, and Concepts/loadsxm.py _read_header's two blank lines.
    This checks header compatibility, not the Scan Inspector GUI itself.
    """
    raw = path.read_bytes()
    marker = raw.index(b"\x1a\x04")
    header = raw[:marker]
    if b"\r" in header:
        raise ValueError("SXM header must use native LF line endings.")
    lines = header.decode("ascii").split("\n")
    cursor = lines.index(":DATA_INFO:") + 1
    table = []
    # The official MATLAB loader reads until the first empty line.
    while cursor < len(lines) and len(lines[cursor].strip()) > 2:
        row = lines[cursor].strip().split("\t")
        if len(row) != 6 or lines[cursor].startswith(":"):
            raise ValueError("Missing DATA_INFO blank separator: a tag/data entered the channel table.")
        table.append(row)
        cursor += 1
    if len(table) != 2 or table[0][0] != "Channel" or not table[1][0].isdigit():
        raise ValueError("Invalid single-channel DATA_INFO table.")
    if cursor + 1 >= len(lines) or lines[cursor + 1] != ":SCANIT_END:":
        raise ValueError("Missing header end after the DATA_INFO table.")
    if not header.endswith(b":SCANIT_END:\n\n\n"):
        raise ValueError("Missing vendor reference loader's post-header blank lines.")


def verify_sxm(path: Path, grid: dict, expected: np.ndarray, name: str, unit: str,
               bias_v: float) -> dict:
    from _sxm_reader import read_sxm
    verify_nanonis_header(path)
    loaded = read_sxm(path)
    def require(condition, message):
        if not condition:
            raise ValueError(f"SXM verification failed: {path.name}: {message}")
    require(loaded.scan_pixels == (grid["nx"], grid["ny"]), "pixels")
    require(loaded.scan_range == grid["settings"][2:4], "physical range")
    require(len(loaded.channels) == 1, "channel count")
    channel = loaded.channels[0]
    require((channel.name, channel.unit, channel.direction) == (name, unit, "forward"), "channel/unit/direction")
    require(np.array_equal(channel.data, expected.astype(float)), "raw pixel values")
    require(np.isclose(float(loaded.header_fields["BIAS"][0]), bias_v, rtol=1e-12, atol=1e-15), "bias")
    require(float(loaded.header_fields["SCAN_ANGLE"][0]) == grid["settings"][4], "angle")
    require(loaded.header_fields["SCAN_DIR"] == ["up"], "spatial direction")
    require(np.allclose(list(map(float, loaded.header_fields["SCAN_OFFSET"][0].split())), grid["settings"][:2], rtol=1e-12, atol=1e-20), "offset")
    return dict(pixel_values_exact=True, nanonis_header_contract_checked=True, pixels=int(expected.size),
                file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def create_preview(output: Path, grid: dict, rows: list[dict]) -> None:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    indices = np.unique(np.rint(np.linspace(0, grid["npts"] - 1, min(15, grid["npts"]))).astype(int))
    selected = [None] + [float(grid["bias"][i] * 1000) for i in indices]
    canvas = Image.new("RGB", (1000, 1140), "white")
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"Raw Z and LI Demod 1 X | {grid['nx']} x {grid['ny']} | {grid['settings'][2]*1e9:g} x {grid['settings'][3]*1e9:g} nm", font=font, fill="black")
    draw.text((12, 35), "Preview: individual 2-98% display contrast; exported SXM values are unchanged.", font=font, fill="black")
    for tile, energy in enumerate(selected):
        if energy is None:
            arr, label = grid["z"].astype(float), f"000 Topo Z ({float(grid['fields']['Bias>Bias (V)'])*1000:+g} mV)"
        else:
            i = int(np.argmin(np.abs(grid["bias"] * 1000 - energy)))
            arr = grid["lix"][:, :, i].astype(float)
            row = next(r for r in rows if r["source_sweep_index_0based"] == i)
            label = f"{row['order']:03d}  LI X  {energy:+g} mV"
        lo, hi = np.percentile(arr, [2, 98])
        image = Image.fromarray(np.uint8(np.clip((arr - lo) / max(hi - lo, 1e-30), 0, 1) * 255))
        # Display top (+y) at the top, matching a conventional SCAN_DIR=up viewer.
        image = ImageOps.flip(image).resize((224, 224), Image.Resampling.NEAREST)
        image = ImageOps.colorize(image, "#120800", "#ffe5a6")
        x0, y0 = 12 + (tile % 4) * 248, 74 + (tile // 4) * 266
        draw.text((x0, y0), label, font=font, fill="black")
        canvas.paste(image, (x0, y0 + 28))
    canvas.save(output / "偏压图预览.png")


def main() -> None:
    began = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--inspect", action="store_true", help="Print grid metadata without writing files.")
    parser.add_argument("--preview", action="store_true", help="Also generate a contact image (requires Pillow).")
    parser.add_argument("--order", choices=["ascending", "acquisition"], default="ascending")
    args = parser.parse_args()
    grid = read_grid(args.source.resolve())
    if args.inspect:
        print(json.dumps({"source":str(grid["path"]), "sha256":grid["sha256"], "pixels":[grid["nx"],grid["ny"]], "points":grid["npts"], "bias_start_mV":float(grid["bias"][0]*1000), "bias_end_mV":float(grid["bias"][-1]*1000), "exported_channels":["LI Demod 1 X (A)", "Z (m)"], "z_equals_scan_z":grid["scan_z_equal"]}, indent=2, ensure_ascii=False))
        return
    if args.output is None:
        parser.error("output is required unless --inspect is used")
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output folder is not empty; existing data will not be overwritten.")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    checks = []
    match = re.search(r"grid-p\d+-\d+", grid["path"].stem)
    short = match.group() if match else re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", grid["path"].stem)
    short = short.rstrip(". ")[:100] or "grid"
    topo_bias = float(grid["fields"]["Bias>Bias (V)"])
    path = args.output / f"000_{short}_Topo_Z.sxm"
    write_sxm(path, grid, grid["z"], "Z", "m", topo_bias, None)
    checks.append(dict(filename=path.name, **verify_sxm(path, grid, grid["z"], "Z", "m", topo_bias)))
    rows.append(dict(order=0, filename=path.name, channel="Z", unit="m",
                     bias_mV=topo_bias * 1000, bias_role="topography_setpoint",
                     source_sweep_index_0based="", source_parameter="Z (m)"))
    ordered = np.argsort(grid["bias"], kind="stable") if args.order == "ascending" else np.arange(grid["npts"])
    for order, i in enumerate(ordered, 1):
        i = int(i)
        bias_v = float(grid["bias"][i])
        bias_mV = bias_v * 1000
        tag = f"{bias_mV:+08.3f}mV"
        path = args.output / f"{order:03d}_{short}_LI_Demod_1_X_{tag}.sxm"
        data = grid["lix"][:, :, i]
        write_sxm(path, grid, data, "LI Demod 1 X", "A", bias_v, i)
        checks.append(dict(filename=path.name, **verify_sxm(path, grid, data, "LI Demod 1 X", "A", bias_v)))
        rows.append(dict(order=order, filename=path.name, channel="LI Demod 1 X", unit="A",
                         bias_mV=f"{bias_mV:.9f}", bias_role="spectral_slice",
                         source_sweep_index_0based=i, source_parameter=""))
    with (args.output / "编号与偏压对照.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "原始网格坐标.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["grid_row_0based", "grid_col_0based", "X_m", "Y_m"])
        for row in range(grid["ny"]):
            for col in range(grid["nx"]):
                writer.writerow([row, col, f"{grid['x'][row,col]:.17g}", f"{grid['y'][row,col]:.17g}"])
    (args.output / "源文件头.txt").write_text(grid["header"], encoding="utf-8")
    report = dict(source_file=str(grid["path"]), source_sha256=grid["sha256"],
                  grid_pixels=[grid["nx"], grid["ny"]], grid_settings=grid["settings"],
                  bias_count=grid["npts"], file_count=len(rows), order=args.order,
                  z_source_parameter="Z (m)", z_equals_scan_z=grid["scan_z_equal"],
                  exported_channels=["Z (m)", "LI Demod 1 X (A)"],
                  verification="Nanonis vendor header contract and bundled SXM reader: every pixel, channel, unit, bias and geometry checked.",
                  files=checks)
    (args.output / "导出校验记录.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    notes = f"""3DS → SXM 导出说明
源文件：{grid['path']}
SHA-256：{grid['sha256']}
{len(rows)} 个 SXM：000 为 Z 形貌；001–{grid['npts']:03d} 为 LI Demod 1 X；排列规则 {args.order}；偏压范围 {float(np.min(grid['bias'])*1000):g} 至 {float(np.max(grid['bias'])*1000):g} mV。
形貌取自参数 Z (m)，与 Scan:Z (m) 是否逐像素一致：{grid['scan_z_equal']}。
Topo 的 BIAS 为采集设定值 {topo_bias*1000:g} mV；偏压切片的 BIAS 为对应谱点。
LI Demod 1 X 单位保持原文件的 A；未擅自换算成 S。Current 和 LI Demod 1 Y 未导出。
{grid['nx']} × {grid['ny']} 像素；Grid settings 的 {grid['settings'][2]*1e9:g} × {grid['settings'][3]*1e9:g} nm、中心坐标、{grid['settings'][4]:g}° 均保留。
原始网格记录从低 y 到高 y；SXM 为 SCAN_DIR=up，forward 单方向，不生成虚构 backward 数据。
SXM 是从网格生成的二维数值文件，不是独立采集的扫描；ACQ_TIME/SCAN_TIME 标为 N/A/NaN。
未做平面扣除、逐行平整、滤波、归一化或旋转；每个导出的 float32 像素与源数据一致。
预览图仅独立调节 2–98% 显示对比度，不影响 SXM。
SXM 的扫描范围沿用 Grid settings；精确网格点坐标另保存在原始网格坐标.csv，避免网格端点/扫描像素中心约定差异。
源谱采集顺序为 {grid['bias'][0]*1000:g} 到 {grid['bias'][-1]*1000:g} mV；编号与偏压对照.csv 记录原谱点索引。
所有 SXM 已核对 Nanonis 官方参考读取器的头部约定（LF、DATA_INFO 后空行、SCANIT_END 后两空行），再用技能内独立读取器逐像素检查单位/尺寸/角度/偏压。
头部约定检查不能代替 Scan Inspector 界面的实际打开验证。
格式参考：Gwyddion 的 Nanonis SXM 导入模块（https://raw.githubusercontent.com/cbuehler/gwyddion/master/modules/file/nanonis.c）。
"""
    (args.output / "导出说明.txt").write_text(notes, encoding="utf-8-sig")
    if args.preview:
        create_preview(args.output, grid, rows)
    report["elapsed_seconds"] = round(perf_counter() - began, 3)
    (args.output / "导出校验记录.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
