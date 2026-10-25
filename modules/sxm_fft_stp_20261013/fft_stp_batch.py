"""Plane-subtracted FFT Modulus -> native WSxM double STP, without GUI.

Numerical convention verified against existing WSxM native input/FFT pairs.
Native WSxM disk convention: pixel-area-scaled |FFT|, fftshift then roll(-1,-1).
"""
from __future__ import annotations
import argparse
import csv
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'nanonis_converter/scripts'))
from _sxm_reader import read_sxm

SOURCE = Path(os.environ.get('STM_SXM_INPUT', str(HERE / 'input')))
OUTPUT = SOURCE / 'FFT_STP'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plane_and_fft(source_up, width_m, height_m, signal_unit):
    assert source_up.ndim == 2 and np.all(np.isfinite(source_up))
    ny, nx = source_up.shape
    assert width_m > 0 and height_m > 0 and nx >= 2 and ny >= 2
    # Gwyddion's native STP exporter stores both axes reversed relative to its
    # top-down display. Nanonis up-scan storage -> WSxM disk means x reversal.
    native_input = np.fliplr(source_up).astype(np.float64)
    xx, yy = np.meshgrid((np.arange(nx) - (nx - 1) / 2) / nx, (np.arange(ny) - (ny - 1) / 2) / ny)
    design = np.column_stack([np.ones(nx * ny), xx.ravel(), yy.ravel()])
    coeff = np.linalg.lstsq(design, native_input.ravel(), rcond=None)[0]
    plane = (design @ coeff).reshape(ny, nx)
    residual = native_input - plane
    # Floating cancellation is recorded, not hidden by filtering the source.
    assert np.linalg.norm(design.T @ residual.ravel()) <= max(np.linalg.norm(native_input) * np.linalg.norm(design) * 1e-9, 1e-25)
    p = 1 << (max(nx, ny) - 1).bit_length()
    pixel_area_nm2 = width_m * 1e9 / nx * (height_m * 1e9 / ny)
    assert signal_unit in ('m', 'A')
    display_unit = 'pm' if signal_unit == 'm' else 'pA'
    display_scale = 1e12
    native_fft = np.roll(np.abs(np.fft.fftshift(np.fft.fft2(residual * display_scale, s=(p, p)))) * pixel_area_nm2, (-1, -1), axis=(0, 1))
    assert np.all(np.isfinite(native_fft)) and np.all(native_fft >= 0)
    range_x_nm_inv, range_y_nm_inv = nx / (width_m * 1e9), ny / (height_m * 1e9)
    kx = (np.arange(p) - p // 2) * range_x_nm_inv / p
    ky = (p // 2 - np.arange(p)) * range_y_nm_inv / p
    return dict(native_fft=native_fft, canonical_fft=native_fft[::-1, ::-1], plane=plane, residual=residual,
                plane_coefficients_SI=coeff, source_shape=(ny, nx), fft_shape=(p, p), padded=(p != nx or p != ny),
                display_zunit=f'{display_unit} nm^2', SI_zunit='m^3' if signal_unit == 'm' else 'A m^2', SI_factor=1e-30,
                pixel_area_nm2=pixel_area_nm2, frequency_range_nm_inv=(range_x_nm_inv, range_y_nm_inv),
                kx=kx, ky=ky, residual_mean_SI=float(residual.mean()), residual_rms_SI=float(np.sqrt(np.mean(residual ** 2))))


def write_stp(path, result, provenance):
    data = result['native_fft']
    ny, nx = data.shape
    lo, hi = float(data.min()), float(data.max())
    span = max(hi - lo, np.finfo(float).tiny)
    xr, yr = result['frequency_range_nm_inv']
    comments = [f'Source SXM: {provenance["source_sxm"]}', f'Source 3DS title: {provenance["source_3ds_title"]}',
                f'Bias: {provenance["bias_mV"]} mV' if provenance['bias_mV'] else 'Bias: N/A (source topography acquisition bias not recorded)',
                f'Source SHA256: {provenance["source_sha256"]}',
                'Least-squares 2D plane removed; Linear; Pass; Rectangle; Modulus; 2D-FFT; k=1/wavelength; zoom=1.',
                'FFT = pixel_area_nm2 * abs(DFT(input_in_pm_or_pA)); WSxM disk center convention verified on native samples.',
                f'Source pixels {result["source_shape"][1]} x {result["source_shape"][0]}; FFT pixels {nx} x {ny}; zero padding only, no resampling.',
                'Displayed zero-frequency pixel (0 based) is (N/2,N/2); raw source files unchanged.',
                f'Source scan angle: {provenance["source_scan_angle_deg"]} deg; Fourier axes follow the source grid axes.']
    comment = '\\n'.join(comments)
    lines = ['WSxM file copyright Nanotec Electronica', 'SxM Image file', 'Image header size: 99999', '', '[Control]', '',
             '    Angle: 0', f'    X Amplitude: {xr:.17g} 1/nm', f'    Y Amplitude: {yr:.17g} 1/nm', '', '[General Info]', '',
             '    Image Data Type: double', f'    Acquisition channel: FFT Modulus {provenance["source_channel"]}',
             f'    Acquisition primary channel: {provenance["source_channel"]}', '    Image processes: converted, plane, FFT',
             f'    Number of columns: {nx}', f'    Number of rows: {ny}', '    X scanning direction: forward', '    Y scanning direction: down',
             f'    Z Amplitude: {span:.17g} {result["display_zunit"]}', '', '[Miscellaneous]', '', f'    Comments: {comment}',
             f'    Maximum: {hi:.17g}', f'    Minimum: {lo:.17g}', '    Relative Z value: No', '    Z Scale Factor: 1', '    Z Scale Offset: 0',
             '    Version: 1.0 (December 2003)', '', '[Header end]', '']
    header = '\r\n'.join(lines).encode('ascii', errors='backslashreplace')
    assert len(header) < 100000
    header = header.replace(b'99999', f'{len(header):5d}'.encode('ascii'), 1)
    with path.open('xb') as out:
        out.write(header)
        out.write(np.asarray(data, dtype='<f8').tobytes(order='C'))


def read_stp(path):
    raw = path.read_bytes()
    size = int(re.search(rb'Image header size:\s*(\d+)', raw[:150]).group(1))
    h = raw[:size].decode('ascii')
    fields = {k.strip(): v.strip() for line in h.splitlines() if ':' in line for k, v in [line.split(':', 1)]}
    nx, ny = int(fields['Number of columns']), int(fields['Number of rows'])
    assert fields['Image Data Type'] == 'double' and h.endswith('[Header end]\r\n')
    assert len(raw) == size + nx * ny * 8
    return np.frombuffer(raw, dtype='<f8', offset=size).reshape(ny, nx), fields


class GwyddionReader:
    """Read generated STP through the independently installed native module."""
    def __init__(self):
        root = Path(os.environ.get('GWYDDION_HOME', str(Path(os.environ.get('ProgramFiles', '')) / 'Gwyddion')))
        self.dll_dir = os.add_dll_directory(str(root / 'bin'))
        self.libs = {p: ct.CDLL(str(root / 'bin' / f'lib{p}2-0.dll')) for p in ['gwyddion', 'gwyprocess', 'gwydraw', 'gwydgets', 'gwymodule', 'gwyapp']}
        for part, method in [('gwyddion', 'gwy_type_init'), ('gwyprocess', 'gwy_process_type_init'), ('gwydraw', 'gwy_draw_type_init'), ('gwydgets', 'gwy_widgets_type_init')]:
            if hasattr(self.libs[part], method):
                getattr(self.libs[part], method)()
        self.module = self.libs['gwymodule']
        self.module.gwy_module_register_modules.argtypes = [ct.POINTER(ct.c_char_p)]
        self.module.gwy_module_register_modules.restype = None
        dirs = (ct.c_char_p * 2)(str(root / 'lib/gwyddion/modules/file').encode('utf-8'), None)
        self.module.gwy_module_register_modules(dirs)
        self.module.gwy_file_load.argtypes = [ct.c_char_p, ct.c_int, ct.POINTER(ct.c_void_p)]
        self.module.gwy_file_load.restype = ct.c_void_p
        self.core, self.proc = self.libs['gwyddion'], self.libs['gwyprocess']
        self.glib = ct.CDLL(str(root / 'bin/libglib-2.0-0.dll'))
        self.glib.g_quark_from_string.argtypes = [ct.c_char_p]
        self.glib.g_quark_from_string.restype = ct.c_uint
        self.core.gwy_container_get_object.argtypes = [ct.c_void_p, ct.c_uint]
        self.core.gwy_container_get_object.restype = ct.c_void_p
        for name in ['gwy_data_field_get_xres', 'gwy_data_field_get_yres']:
            getattr(self.proc, name).argtypes = [ct.c_void_p]
            getattr(self.proc, name).restype = ct.c_int
        for name in ['gwy_data_field_get_xreal', 'gwy_data_field_get_yreal']:
            getattr(self.proc, name).argtypes = [ct.c_void_p]
            getattr(self.proc, name).restype = ct.c_double
        self.proc.gwy_data_field_get_data.argtypes = [ct.c_void_p]
        self.proc.gwy_data_field_get_data.restype = ct.POINTER(ct.c_double)
        self.gobject = ct.CDLL(str(root / 'bin/libgobject-2.0-0.dll'))
        self.gobject.g_object_unref.argtypes = [ct.c_void_p]
        self.gobject.g_object_unref.restype = None

    def validate(self, path, expected):
        error = ct.c_void_p()
        container = self.module.gwy_file_load(str(path).encode('utf-8'), 4, ct.byref(error))
        assert container, f'Native STP import failed: {path.name}'
        try:
            field = self.core.gwy_container_get_object(container, self.glib.g_quark_from_string(b'/0/data'))
            assert field
            ny, nx = expected['fft_shape']
            assert self.proc.gwy_data_field_get_xres(field) == nx and self.proc.gwy_data_field_get_yres(field) == ny
            xr, yr = expected['frequency_range_nm_inv']
            actual_range = [self.proc.gwy_data_field_get_xreal(field), self.proc.gwy_data_field_get_yreal(field)]
            assert np.allclose(actual_range, [xr * 1e9, yr * 1e9], rtol=1e-12, atol=0), actual_range
            values = np.ctypeslib.as_array(self.proc.gwy_data_field_get_data(field), shape=(nx * ny,)).reshape(ny, nx)
            target = expected['canonical_fft'] * expected['SI_factor']
            difference = float(np.max(abs(values - target)))
            relative = difference / max(float(np.max(abs(target))), np.finfo(float).tiny)
            assert relative < 1e-12, f'{path.name}: native STP numeric/units mismatch {relative:g}'
            return dict(native_STP_import_passed=True, canonical_orientation_verified=True, frequency_range_and_units_verified=True, max_relative_numeric_error=relative)
        finally:
            self.gobject.g_object_unref(container)


def verify_synthetic():
    # A non-axis-aligned sinusoid tests reciprocal-axis handedness and peak centers.
    nx = ny = 64
    width_nm, height_nm = 8., 4.
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    source = (3e-12 + 2e-12 * np.cos(2 * np.pi * (7 * xx / nx + 5 * yy / ny)))
    r = plane_and_fft(source, width_nm * 1e-9, height_nm * 1e-9, 'A')
    a = r['canonical_fft']
    peaks = np.unravel_index(np.argsort(a.ravel())[-2:], a.shape)
    actual = {(int(y), int(x)) for y, x in zip(*peaks)}
    assert actual == {(32 - 5, 32 + 7), (32 + 5, 32 - 7)}, actual
    error = abs(float(a[27, 39]) - (2e-12 * 1e12 * width_nm * height_nm / 2))
    # Least-squares removal can weakly alter a wave not orthogonal to the plane.
    assert error / 32 < .002
    # A pure affine surface should leave no physical spectral structure.
    p = plane_and_fft(4e-12 + .7e-12 * xx - .2e-12 * yy, width_nm * 1e-9, height_nm * 1e-9, 'A')
    assert float(p['native_fft'].max()) < 1e-9
    return dict(sinusoid_peak_frequency_and_handedness_verified=True, pure_plane_removed=True,
                candidate_peak_cycles_per_nm=[7 / width_nm, 5 / height_nm], max_pure_plane_fft=float(p['native_fft'].max()))


def csvwrite(path, headers, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as out:
        w = csv.writer(out)
        w.writerow(headers)
        w.writerows(rows)


def previews(selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    sys.path.insert(0, str(HERE.parent / 'stm_line_spectrum_tool'))
    from core import PALETTE
    cmap = ListedColormap(PALETTE / 255.)
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    fig, axes = plt.subplots(len(selected), 2, figsize=(8.5, 3.6 * len(selected)), layout='constrained')
    for pair, (row, result, raw_display) in zip(axes, selected):
        lo, hi = np.percentile(raw_display, [2, 98])
        pair[0].imshow(raw_display, cmap=cmap, vmin=lo, vmax=max(hi, lo + 1e-30), origin='upper', interpolation='nearest')
        pair[0].set(title=f'{row["filename"]} 原图', xlabel='像素 x', ylabel='显示像素 y')
        a = result['canonical_fft']
        vmax = max(float(np.percentile(a, 99)), np.finfo(float).tiny)
        kx, ky = result['kx'], result['ky']
        dx, dy = kx[1] - kx[0], ky[0] - ky[1]
        extent = (kx[0] - dx / 2, kx[-1] + dx / 2, ky[-1] - dy / 2, ky[0] + dy / 2)
        pair[1].imshow(a, cmap=cmap, vmin=0, vmax=vmax, origin='upper', extent=extent, interpolation='nearest')
        pair[1].axhline(0, color='#00c9c9', lw=.4)
        pair[1].axvline(0, color='#00c9c9', lw=.4)
        pair[1].set(title=f'FFT{Path(row["filename"]).stem}.stp · 扣平面 + FFT', xlabel='kx（1/nm）', ylabel='ky（1/nm）')
    fig.savefig(OUTPUT / 'FFT_preview.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', action='store_true', help='Only validate one sample under the local project folder.')
    args = parser.parse_args()
    started = perf_counter()
    synthetic = verify_synthetic()
    native_ref = json.loads((HERE / 'native_wsxm_fft_comparison.json').read_text(encoding='utf-8'))
    assert all(t['best_match']['relative_L2_error'] < 1e-12 for t in native_ref)
    reader = GwyddionReader()
    with (SOURCE / 'index_and_bias.csv').open(encoding='utf-8-sig', newline='') as out:
        rows = list(csv.DictReader(out))
    assert len(rows) == 450 and len({r['source_3ds_title'] for r in rows}) == 17
    destination = HERE / 'sample' if args.sample else OUTPUT
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Output is not empty; refusing overwrite: {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    if not args.sample:
        (destination / 'by_3ds').mkdir()
    chosen = rows[:1] if args.sample else rows
    checks, index, selected, group_info, written_axes = [], [], [], {}, set()
    for j, row in enumerate(chosen, 1):
        src = SOURCE / row['filename']
        assert digest(src) == row['sxm_sha256']
        sxm = read_sxm(src)
        assert len(sxm.channels) == 1 and sxm.header_fields['SCAN_DIR'] == ['up']
        ch = sxm.channels[0]
        assert ch.direction == 'forward' and sxm.scan_range is not None
        result = plane_and_fft(ch.data, *sxm.scan_range, ch.unit)
        filename = f'FFT{src.stem}.stp'
        path = destination / filename
        provenance = dict(source_sxm=row['filename'], source_3ds_title=row['source_3ds_title'], source_sha256=row['sxm_sha256'],
                          bias_mV=row['bias_mV'], source_channel=ch.name, source_scan_angle_deg=sxm.header_fields.get('SCAN_ANGLE', ['unknown'])[0])
        write_stp(path, result, provenance)
        own_values, fields = read_stp(path)
        assert np.array_equal(own_values, result['native_fft'])
        native_check = reader.validate(path, result)
        output_hash = digest(path)
        if not args.sample:
            folder = destination / 'by_3ds' / row['source_3ds_title']
            assert folder.parent == destination / 'by_3ds' and re.fullmatch(r'Grid Spectroscopy\d{3}\.3ds', folder.name)
            folder.mkdir(exist_ok=True)
            shutil.copyfile(path, folder / filename)
            assert digest(folder / filename) == output_hash
        assert digest(src) == row['sxm_sha256']
        check = dict(filename=filename, source_sxm=row['filename'], source_sha256=row['sxm_sha256'], STP_sha256=output_hash,
                     source_shape=list(result['source_shape']), fft_shape=list(result['fft_shape']), zero_padded=result['padded'],
                     plane_coefficients_SI=result['plane_coefficients_SI'].tolist(), plane_residual_mean_SI=result['residual_mean_SI'],
                     all_double_values_exact_on_reread=True, **native_check)
        checks.append(check)
        index.append([filename, row['filename'], row['source_3ds_title'], row['image_kind'], row['bias_mV'], *result['source_shape'][::-1], *result['fft_shape'][::-1],
                      *result['frequency_range_nm_inv'], result['display_zunit'], result['padded'], output_hash, row['sxm_sha256']])
        group_info.setdefault(row['source_3ds_title'], []).append(filename)
        if not args.sample:
            axes_key = (*result['fft_shape'], *result['frequency_range_nm_inv'])
            if axes_key not in written_axes:
                written_axes.add(axes_key)
                name = f'frequency_axis_{len(written_axes):02d}.csv'
                csvwrite(destination / name, ['display_pixel_index_0based', 'kx_cycles_per_nm', 'ky_cycles_per_nm'], [[i, x, y] for i, (x, y) in enumerate(zip(result['kx'], result['ky']))])
            if j in [1, 2, 13, 15]:
                selected.append((row, result, np.flipud(ch.data)))
        if j == 1 or j % 50 == 0 or j == len(chosen):
            print(f'FFT/STP verified {j}/{len(chosen)} : {filename}', flush=True)
    if not args.sample:
        assert len(list(destination.glob('*.stp'))) == 450 and len(list((destination / 'by_3ds').rglob('*.stp'))) == 450
        assert len(group_info) == 17
        csvwrite(destination / 'FFT_index.csv', ['STP_filename', 'source_SXM', 'source_3ds_title', 'image_kind', 'bias_mV', 'source_nx', 'source_ny', 'FFT_nx', 'FFT_ny',
                                                'full_kx_range_cycles_per_nm', 'full_ky_range_cycles_per_nm', 'FFT_value_unit', 'zero_padded', 'STP_sha256', 'source_SXM_sha256'], index)
        csvwrite(destination / 'by_3ds/folder_index.csv', ['source_3ds_title', 'STP_count', 'first_filename', 'last_filename'], [[key, len(vals), vals[0], vals[-1]] for key, vals in group_info.items()])
        previews(selected)
    report = dict(source_folder=str(SOURCE), output_folder=str(destination), primary_STP_count=len(chosen), classified_copy_count=0 if args.sample else len(chosen),
                  source_groups=len(group_info), source_files_unchanged=True, source_copies_not_double_counted=True,
                  method='2D least-squares plane; Linear/Pass/Rectangle/Modulus/2D-FFT; k=1/wavelength; pixel-area-scaled continuous-transform amplitude',
                  FFT_dimensions='smallest square power-of-two at least as large as both source dimensions; zero padding only',
                  native_WSxM_reference_crosscheck=native_ref, synthetic_validation=synthetic,
                  all_STP_double_values_exact_on_reread=True, all_native_Gwyddion_STP_imports_verified=True,
                  max_native_import_relative_error=max(t['max_relative_numeric_error'] for t in checks),
                  newly_opened_in_WSxM_GUI=False, elapsed_seconds=round(perf_counter() - started, 3), files=checks)
    (destination / 'FFT_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.sample:
        readme = '''# SXM → 扣平面 → FFT Modulus → WSxM STP

## 输出与名称

本目录包含 450 张独立 FFT 图，文件名仅在原编号前加 FFT，例如 2026_00001.sxm → FFT2026_00001.stp。by_3ds 再保存相同 450 张图的分类副本，17 个子文件夹名称与原 3DS 完整文件名一致（包含 .3ds）。没有把上一级的分类副本当作新输入重复计算。

原始 SXM、平铺版、分类副本和原始 3DS 都没有修改。FFT_index.csv 保存偏压、来源、单位、补零标记和哈希；by_3ds/folder_index.csv 保存各组数量。

## 计算参数

按用户确认，每张实测二维图先独立拟合并减去 a+b*x+c*y 平面。不是逐行平整，没有高斯滤波、空间重采样或旋转。随后 Linear、Pass、Rectangle、Modulus、2D-FFT，k=1/wavelength，zoom=1。存的是复数 DFT 的模，不是功率谱、不做对数或平方。

WSxM 原生样本核对表明，FFT 幅度采用 abs(DFT)*dx*dy；坐标单位 nm，因此输出值为 pm nm^2（Z）或 pA nm^2（LI X），不是绝对 dI/dV。输入的 m/A 仅为显示友好分别换算成 pm/pA，再乘真实像素面积。FFT 原生 on-disk 中心排列与旧 WSxM 样本一致；读取后显示零频位于 (N/2,N/2)，横轴 kx 向右增加、ky 向上增加。

FFT 采用不小于原图尺寸的最小正方形 2 次幂；本批 100×100 补零到 128×128，其他 64、128、256 正方形保持原尺寸。补零仅细化频率采样，不增加真实分辨率，不生成新的实测像素。

完整频率范围为 nx/原空间宽度、ny/原空间高度，单位 1/nm，频率步长分别为完整范围/FFT像素数。所有波矢为原扫描网格轴，扫描角度写在 comment 中，没有旋转到实验室坐标。frequency_axis_*.csv 给出精确中心坐标。

## 文件格式与独立验证

STP 为原生 WSxM Image 格式，准确 header size、CRLF、Image Data Type: double、小端 float64 载荷，不是更改扩展名。使用 Gwyddion 官方 WSxM 模块源码核对格式：
https://svn.code.sf.net/p/gwyddion/code/trunk/gwyddion/modules/file/wsxmfile.c

程序与两组成对的旧原生 WSxM STP/FFT 文件比对：512×512 直接 FFT 的相对 L2 误差约 3.35e-16；208×512 再扣平面、补零到 512×512 的误差约 1.26e-15。对应源路径与完整误差在 FFT_validation.json 中。这个验证针对模值、比例、中心排列及补零约定；不是对每张新图的 WSxM GUI 操作检查。

每个新 STP 均独立重读，double 数值逐点精确相同；又通过本机 Gwyddion 原生 WSxM 导入模块核对全矩阵数值、镜像方向、频率范围和物理单位。非轴向正弦测试验证峰位置和轴方向；纯平面测试验证背景扣除。分类副本的 SHA256 逐个与平铺文件一致。

FFT_preview.png 仅为抽样浏览图，使用 Nanox、线性色阶，上限按各图 P99 设置；它的显示截断不影响 STP 数值。实际 WSxM 颜色可按软件内的色表调整。
'''
        (destination / 'README_中文.md').write_text(readme, encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['files', 'native_WSxM_reference_crosscheck']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
