"""Measured-pixel Se-row spatial FFT and a frozen-scale comparison to prior rows."""
from pathlib import Path
import csv, hashlib, json, shutil, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core import GridDataset, bin_edges
from profile_fft import profile_fft
from plotting import NANOX
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
PREVIOUS = ROOT / 'research_quicklooks/20261011_rows067_073_079_085_091_097/results'
SITE = Path(__import__('os').environ['STM_SITE_COORDINATES'])
INDEX = SITE.parent.parent / 'grid-p0313-001_detailed_analysis_20261010/01_sites_and_regions/atom_spectrum_index.csv'
oldmeta = json.loads((PREVIOUS / 'methods_and_validation.json').read_text(encoding='utf-8'))
old = np.load(PREVIOUS / 'six_rows_frequency_energy.npz')
SOURCE = Path(oldmeta['source_file'])
d = GridDataset(SOURCE)
assert (d.nx, d.ny, len(d.bias)) == (99, 99, 81)
assert d.hash == oldmeta['source_sha256']
sites = list(csv.DictReader(SITE.open(encoding='utf-8-sig')))
index = {t['spectrum_id']: t for t in csv.DictReader(INDEX.open(encoding='utf-8-sig'))}
SE_ATOM_ROWS = np.arange(11, 16)
row_sites = [[t for t in sites if t['species'] == 'Se' and int(t['r']) == r] for r in SE_ATOM_ROWS]
assert all(len(q) == 11 for q in row_sites)
assert all(t['region'] == 'R1_dark' for q in row_sites for t in q)
assert all(float(t[k]) == float(index[t['spectrum_id']][k]) for q in row_sites for t in q for k in ['display_x_px', 'display_y_px'])
rows = np.array([int(np.rint(np.mean([float(t['display_y_px']) for t in q]))) for q in row_sites])
assert rows.tolist() == [70, 76, 82, 88, 94]
refrows = old['pixel_rows'].astype(int)
assert np.all((rows > refrows[:-1]) & (rows < refrows[1:]))
labels = [f'SeR{r:02d}' for r in SE_ATOM_ROWS]
E = d.bias
s = old['distance_nm'].copy()
coords = np.array([np.column_stack([np.arange(99), np.full(99, r)]) for r in rows])
lab = np.array([d.lab(xy) for xy in coords])
signals = d.g[rows].copy()
assert np.array_equal(E, old['bias_mV']) and np.array_equal(d.g[refrows], old['raw_signal_pA'])
step = float(np.linalg.norm(d.transform[1]))
assert np.allclose(s, np.arange(99) * step, atol=1e-9)
direction = d.transform[1] / step
assert np.allclose((lab - lab[:, :1]) @ direction, s[None, :], atol=1e-9)

# Independent binary payload verification, not a second call to GridDataset.
blob = SOURCE.read_bytes()
marker = b':HEADER_END:'
pos = blob.index(marker) + len(marker)
for ending in (b'\r\r\n', b'\r\n', b'\n'):
    if blob[pos:pos + len(ending)] == ending:
        pos += len(ending)
        break
else:
    raise ValueError('Unsupported header terminator')
h = d.header
np0, pts = int(h['# Parameters (4 byte)']), int(h['Points'])
channels = h['Channels'].split(';')
params = h['Fixed parameters'].split(';') + h['Experiment parameters'].split(';')
records = np.frombuffer(blob, dtype='>f4', offset=pos).reshape(99, 99, np0 + len(channels) * pts)
order = np.argsort(np.linspace(float(records[0, 0, params.index('Sweep Start')]), float(records[0, 0, params.index('Sweep End')]), pts))
ci = channels.index('LI Demod 1 X (A)')
direct = records[98 - rows, :, np0 + ci * pts:np0 + (ci + 1) * pts].astype(np.float64)[:, :, order] * 1e12
raw_error = float(np.max(np.abs(direct - signals)))
assert raw_error == 0
design = np.column_stack([np.ones(99), s])
trends = np.array([design @ np.linalg.lstsq(design, g, rcond=None)[0] for g in signals])
processed = signals - trends
all_fft = [[profile_fft(s, g[:, k], 'hann') for k in range(81)] for g in processed]
f = np.asarray(all_fft[0][0]['frequency_cycles_per_nm'])
assert np.array_equal(f, old['frequency_cycles_per_nm'])
assert all(not z['resampled'] and z['zero_padding'] == 0 and np.array_equal(z['frequency_cycles_per_nm'], f) for row in all_fft for z in row)
amp = np.array([[z['amplitude'] for z in row] for row in all_fft])
coeff = np.array([[np.asarray(z['coefficient_real']) + 1j * np.asarray(z['coefficient_imag']) for z in row] for row in all_fft])
mean5 = amp.mean(axis=0)
sd5 = amp.std(axis=0, ddof=1)
line_means = signals.mean(axis=1)
assert np.all(line_means > 0)
relative = amp / line_means[:, :, None]
relative5 = relative.mean(axis=0)
ref6, ref5 = old['mean6_amplitude_pA'], old['dark5_amplitude_pA']
delta = mean5 - ref5
fb, f3 = float(old['q_bragg_cycles_per_nm']), float(old['q_third_cycles_per_nm'])
limits = oldmeta['absolute_color_limits_pA']
rel_limits = oldmeta['relative_color_limits']
full_limits = oldmeta['full_color_limits_pA']
average_error = float(np.max(np.abs(mean5 - sum(amp) / 5)))
assert average_error < 1e-12
reverse_error = max(float(np.max(np.abs(np.asarray(profile_fft(s, g[::-1, k], 'hann')['amplitude']) - amp[j, k]))) for j, g in enumerate(processed) for k in [0, 40, 45, 80])
assert reverse_error < 1e-10

def csvfile(name, headers, values):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as target:
        w = csv.writer(target)
        w.writerow(headers)
        w.writerows(values)

series = {**{label: amp[j] for j, label in enumerate(labels)}, 'SeMean5': mean5, 'RefMean6': ref6, 'RefDark5': ref5}
titles = {**{label: f'Se 第 {r} 排 · 像素行 {y}' for label, r, y in zip(labels, SE_ATOM_ROWS, rows)},
          'SeMean5': '过 Se 五条线 · FFT 幅度平均', 'RefMean6': '此前六行平均 · 67/73/79/85/91/97', 'RefDark5': '此前 dark 五行平均 · 73/79/85/91/97'}
for key, values in {**series, 'SeSD5': sd5, 'SeRelative5': relative5, 'SeMinusRefDark5': delta}.items():
    csvfile(f'{key}_FFT_amplitude.csv', ['bias_mV', *[f'f_{v:.12g}_cycles_per_nm' for v in f]], np.column_stack([E, values]))
for j, label in enumerate(labels):
    csvfile(f'{label}_raw_spectra_pA.csv', ['bias_mV', *[f'{label}_S{k + 1:04d}' for k in range(99)]], np.column_stack([E, signals[j].T]))
csvfile('frequency_axis.csv', ['frequency_index', 'frequency_cycles_per_nm', 'period_nm', 'frequency_over_horizontal_Bragg'], [[j, v, '' if v == 0 else 1 / v, v / fb] for j, v in enumerate(f)])
csvfile('sample_positions.csv', ['line_label', 'sample_name', 'Se_atom_row', 'distance_nm', 'display_x_px', 'display_y_px', 'source_col', 'source_row', 'recorded_x_nm', 'recorded_y_nm', 'region'],
        [[label, f'{label}_S{k + 1:04d}', int(r), s[k], k, y, k, 98 - y, *lab[j, k], 'R1_dark'] for j, (label, r, y) in enumerate(zip(labels, SE_ATOM_ROWS, rows)) for k in range(99)])
atom_rows = []
for j, q in enumerate(row_sites):
    for t in q:
        xx, yy = float(t['display_x_px']), float(t['display_y_px'])
        atom_rows.append([t['spectrum_id'], int(SE_ATOM_ROWS[j]), int(t['c']), xx, yy, labels[j], int(rows[j]), yy - rows[j], int(np.rint(xx)), f'{labels[j]}_S{int(np.rint(xx)) + 1:04d}', yy - rows[j], (yy - rows[j]) * np.linalg.norm(d.transform[2])])
csvfile('selected_Se_atoms_and_line_offsets.csv', ['atom_id', 'Se_atom_row', 'Se_atom_col', 'calibrated_x_px', 'calibrated_y_px', 'line_label', 'line_display_row', 'signed_vertical_offset_px', 'nearest_line_x_px', 'nearest_line_sample_name', 'offset_display_y_px', 'signed_vertical_offset_nm'], atom_rows)
csvfile('complex_FFT_long.csv', ['line_label', 'bias_mV', 'frequency_cycles_per_nm', 'raw_DFT_real_pA', 'raw_DFT_imag_pA', 'amplitude_pA', 'phase_rad'],
        [[label, e, v, c.real, c.imag, amplitude, float(np.angle(c))] for label, cf, af in zip(labels, coeff, amp) for e, cs, amps in zip(E, cf, af) for v, c, amplitude in zip(f, cs, amps)])
np.savez_compressed(OUT / 'Se_rows_frequency_energy.npz', pixel_rows=rows, Se_atom_rows=SE_ATOM_ROWS, bias_mV=E, frequency_cycles_per_nm=f, distance_nm=s, display_xy_px=coords, recorded_lab_xy_nm=lab,
                    raw_signal_pA=signals, linear_background_pA=trends, processed_signal_pA=processed, fft_amplitude_pA=amp, fft_complex_real=coeff.real, fft_complex_imag=coeff.imag,
                    line_mean_pA=line_means, fft_relative_amplitude=relative, SeMean5_amplitude_pA=mean5, SeSD5_amplitude_pA=sd5, SeMean5_relative=relative5,
                    RefMean6_amplitude_pA=ref6, RefDark5_amplitude_pA=ref5, SeMinusRefDark5_amplitude_pA=delta, reference_pixel_rows=refrows,
                    reference_fft_amplitude_pA=old['fft_amplitude_pA'], reference_raw_signal_pA=old['raw_signal_pA'], topo_plane_subtracted_pm=d.topo,
                    Se_calibrated_xy_px=np.array([[float(t['display_x_px']), float(t['display_y_px'])] for q in row_sites for t in q]),
                    q_bragg_cycles_per_nm=fb, q_third_cycles_per_nm=f3, fft_window=np.hanning(99))
shutil.copyfile(PREVIOUS / 'Nanox.pal', OUT / 'Nanox.pal')

row_geometry = [dict(line_label=label, Se_atom_row=int(r), pixel_row=int(y), neighboring_previous_rows=[int(refrows[j]), int(refrows[j + 1])],
                     calibrated_Se_count=11, calibrated_Se_mean_y_px=float(np.mean([float(t['display_y_px']) for t in q])),
                     maximum_vertical_offset_px=float(max(abs(float(t['display_y_px']) - y) for t in q)),
                     original_ids=[t['spectrum_id'] for t in q]) for j, (label, r, y, q) in enumerate(zip(labels, SE_ATOM_ROWS, rows, row_sites))]
meta = dict(source_file=str(SOURCE), source_sha256=d.hash, site_coordinate_file=str(SITE), site_coordinate_sha256=hashlib.sha256(SITE.read_bytes()).hexdigest(),
            site_reference_index_file=str(INDEX), site_reference_index_sha256=hashlib.sha256(INDEX.read_bytes()).hexdigest(), previous_results=str(PREVIOUS),
            previous_metadata_sha256=hashlib.sha256((PREVIOUS / 'methods_and_validation.json').read_bytes()).hexdigest(),
            coordinate_origin='display top-left, zero-based pixel rows; Se atom rows are separate 1-based top-to-bottom IDs',
            Se_atom_rows=SE_ATOM_ROWS.tolist(), display_pixel_rows=rows.tolist(), row_geometry=row_geometry,
            display_x_range=[0, 98], samples_per_row=99, line_length_nm=float(s[-1]), pixel_step_nm=step,
            bias_count=81, bias_range_mV=[float(E[0]), float(E[-1])], bias_step_mV=float(E[1] - E[0]),
            frequency_spacing_cycles_per_nm=float(f[1]), frequency_count_including_DC=len(f),
            line_selection='horizontal integer row nearest mean calibrated Se y; full x=0..98, same measured-pixel convention as previous six lines',
            fft_processing=oldmeta['fft_processing'], amplitude_definition=oldmeta['amplitude_definition'], zero_padding=0, spatial_interpolation=False, energy_interpolation=False,
            mean_definition='five independent one-sided amplitude FFT matrices, then unweighted arithmetic mean; not coherent complex average',
            absolute_color_limits_pA=limits, relative_color_limits=rel_limits, full_color_limits_pA=full_limits,
            color_limits_policy='reused unchanged from previous six-row results, not recomputed from Se data',
            reference_Bragg_frequency_cycles_per_nm=fb, candidate_third_frequency_cycles_per_nm=f3,
            line_composition='Se-connecting horizontal lines also pass same-row FeA; comparison is between paths, not pure chemical-site spectra',
            reference_average_definitions={'RefMean6': refrows.tolist(), 'RefDark5': refrows[1:].tolist()},
            limitations=['Approximate integer-pixel straight lines, not exact interpolated atom centers.', 'Five Se paths are spatially interleaved with the prior paths, not coincident.',
                         'The Se paths include same-row FeA and other pixels; not Se-only atom spectra.', 'Row SD is line heterogeneity, not measurement error.',
                         'RefMean6 includes transition row 67; RefDark5 excludes it.', 'Amplitude mean does not establish common spatial phase.',
                         'Native df stays approximately 0.25 cycles/nm; no proof of exact 3a, significance or PDW.'],
            validation=dict(direct_binary_spectra_max_error_pA=raw_error, mean5_arithmetic_error_pA=average_error, reversed_line_amplitude_max_error_pA=reverse_error,
                            previous_source_spectra_exact_match=True, common_frequency_axis_verified=True, calibrated_Se_positions_match_previous_index=True,
                            fixed_previous_color_limits=True, source_hash_unchanged=True))
(OUT / 'methods_and_validation.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

def freqmap(ax, values, scale=limits, full=False, cmap=NANOX):
    im = ax.pcolormesh(bin_edges(f[1:]), bin_edges(E), values[:, 1:], cmap=cmap, shading='flat', vmin=scale[0], vmax=scale[1], rasterized=True)
    span = (-80, 80) if full else (-30, 30)
    ax.set(xlim=(0, float(f[-1]) if full else 3.5), ylim=span, xlabel='空间频率 f（cycles/nm）', ylabel='偏压（mV）')
    for v, label in [(f3, '1/(3a)'), (2 * f3, '2/(3a)'), (fb, '1/a')]:
        ax.axvline(v, color='#00c9c9', ls='--', lw=.8)
        ax.text(v + .03, span[1] - .06 * (span[1] - span[0]), label, color='#00c9c9', fontsize=8, bbox=dict(fc='black', ec='none', alpha=.55, pad=1))
    ax.axhline(0, color='white', lw=.55, alpha=.55)
    ax.axhline(10, color='white', ls=':', lw=.65, alpha=.7)
    return im

fig, axs = plt.subplots(2, 4, figsize=(15.4, 8.7), layout='constrained', sharex=True, sharey=True)
for ax, (key, values) in zip(axs.flat, series.items()):
    im = freqmap(ax, values)
    ax.set_title(titles[key], fontsize=10)
fig.colorbar(im, ax=axs.ravel().tolist(), label='FFT 幅度（LI X, pA）', fraction=.025, pad=.02)
fig.suptitle('过 Se 的五条线及平均 · 与此前六行 / dark 五行对照\n共同 Nanox 色阶 · 每行 99 个实测像素 · 去线性背景 + Hann', fontsize=12)
for fmt in ['png', 'pdf']:
    fig.savefig(OUT / f'Se_rows_frequency_energy_overview.{fmt}', dpi=180)
plt.close(fig)
for key, values in series.items():
    fig, ax = plt.subplots(figsize=(7, 5.8), layout='constrained')
    im = freqmap(ax, values)
    ax.set_title(titles[key] + '\n与上一组相同处理和色阶', fontsize=10)
    fig.colorbar(im, ax=ax, label='FFT 幅度（LI X, pA）')
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{key}_frequency_energy.{fmt}', dpi=180)
    plt.close(fig)
for key, values, scale, full in [('SeMean5_relative', relative5, rel_limits, False), ('SeMean5_full', mean5, full_limits, True)]:
    fig, ax = plt.subplots(figsize=(7, 5.8), layout='constrained')
    im = freqmap(ax, values, scale, full)
    ax.set_title('过 Se 五线平均 · ' + ('相对幅度' if not full else '完整偏压和正频率'), fontsize=10)
    fig.colorbar(im, ax=ax, label='相对幅度（无量纲）' if not full else 'FFT 幅度（pA）')
    for fmt in ['png', 'pdf']:
        fig.savefig(OUT / f'{key}_frequency_energy.{fmt}', dpi=180)
    plt.close(fig)
fig, ax = plt.subplots(figsize=(7.5, 5.9), layout='constrained')
lo, hi = np.percentile(d.topo, [1, 99])
ax.imshow(d.topo, cmap=NANOX, vmin=lo, vmax=hi, origin='upper', extent=(-.5, 98.5, 98.5, -.5), interpolation='nearest')
for y in refrows:
    ax.plot([0, 98], [y, y], color='#00c9c9' if y != 67 else '#ff9e32', lw=.8, ls='--', alpha=.8)
    ax.text(99.3, y, f'原 {y}', color='#00c9c9' if y != 67 else '#ff9e32', va='center', fontsize=8)
for r, y, q in zip(SE_ATOM_ROWS, rows, row_sites):
    ax.plot([0, 98], [y, y], color='#ffd400', lw=1)
    ax.scatter([float(t['display_x_px']) for t in q], [float(t['display_y_px']) for t in q], s=14, facecolors='none', edgecolors='#ffd400', linewidths=.6, zorder=4)
    ax.text(99.3, y, f'Se{r:02d} / {y}', color='#b98500', va='center', fontsize=8)
ax.set(xlim=(-.5, 119), ylim=(99, 61), xlabel='显示 x（像素）', ylabel='显示 y（向下，像素）', title='实空间位置对照：黄线过 Se，圆圈为原标注 Se 中心\n青色虚线为上一组；橙色虚线为过渡行 67')
for fmt in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'Se_and_previous_topography_paths.{fmt}', dpi=180)
plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(12, 4.7), layout='constrained', sharex=True, sharey=True)
for ax, values, title in zip(axs, [mean5, ref6, ref5], ['过 Se 五线平均', '此前六线平均（含 67）', '此前 dark 五线平均（不含 67）']):
    im = freqmap(ax, values)
    ax.set_title(title, fontsize=10)
fig.colorbar(im, ax=axs.tolist(), label='FFT 幅度（pA）', fraction=.025, pad=.02)
for fmt in ['png', 'pdf']:
    fig.savefig(OUT / f'averages_comparison.{fmt}', dpi=180)
plt.close(fig)

# The candidate reference falls between native bins; retain both, with no interpolation.
js = [np.flatnonzero(f < f3)[-1], np.flatnonzero(f > f3)[0]]
headers, blocks = ['bias_mV'], [E]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), layout='constrained')
for ax, k in zip(axs, js):
    for label, values in zip(labels, amp):
        ax.plot(E, values[:, k], lw=.7, alpha=.45, label=label)
        headers.append(f'{label}_f{f[k]:.9f}_pA')
        blocks.append(values[:, k])
    for key, values, color, ls in [('SeMean5', mean5, '#b98500', '-'), ('RefMean6', ref6, '#6a6a6a', ':'), ('RefDark5', ref5, '#008a96', '--')]:
        ax.plot(E, values[:, k], color=color, ls=ls, lw=1.8, label=key)
        headers.append(f'{key}_f{f[k]:.9f}_pA')
        blocks.append(values[:, k])
    ax.axvspan(0, 10, color='#d9b56e', alpha=.15)
    ax.set(xlim=(-30, 30), ylim=(0, None), xlabel='偏压（mV）', ylabel='FFT 幅度（pA）', title=f'1/(3a) 相邻原生频点 f={f[k]:.3f} cycles/nm')
    ax.grid(alpha=.12)
axs[1].legend(ncol=2, fontsize=8)
csvfile('candidate_frequency_comparison.csv', headers, np.column_stack(blocks))
for fmt in ['png', 'pdf']:
    fig.savefig(OUT / f'candidate_frequency_comparison.{fmt}', dpi=180)
plt.close(fig)

table = '\n'.join(f'| {g["line_label"]} | Se 第 {g["Se_atom_row"]} 排 | {g["pixel_row"]} | {g["neighboring_previous_rows"][0]}–{g["neighboring_previous_rows"][1]} | {g["calibrated_Se_mean_y_px"]:.6f} | {g["maximum_vertical_offset_px"]:.6f} |' for g in row_geometry)
readme = f'''# 过 Se 的五条线：频率—能量图、平均及上一组对照

## 取样位置

用户要求上一组 67、73、79、85、91、97 六条水平线之间，过 Se 原子排的同样频率—能量图。六条线之间共五个间隔，对应已确认的 dark 区域 Se 第 11–15 排，共 55 个标注 Se。不是新增第六条，也没有使用区域外的 Se 排。

Se 原子排编号从上到下为 1 基；像素行为网页左上原点的 0 基，y 向下。依据未修改的人工校准坐标，每排取平均 y 最接近的整像素水平线：

| 线名 | 原子排 | 显示像素行 | 位于上一组两线之间 | 标注 Se 平均 y | 最大垂直偏差 / pixel |
|---|---|---|---|---|---|
{table}

每条线 x=0..98，共 99 个实测像素、单像素宽，长度 {s[-1]:.9f} nm。这些是接近 Se 中心的直线，不对坐标进行双线性插值，不沿各原子中心逐点折线追踪。最大标注中心偏差 {max(g['maximum_vertical_offset_px'] for g in row_geometry):.6f} pixel，55 个 Se 的原始坐标和每条线的偏差完整保存在 selected_Se_atoms_and_line_offsets.csv。Se 坐标与之前详细分析的 atom_spectrum_index.csv 逐项一致。

**过 Se 的方向也会经过同排 FeA 以及其他像素，因此这些是过 Se 排的连续线谱 FFT，不是仅 Se 原子点的频谱。**上一组靠近 FeB 排；两组位置交错，不是完全同一空间取样点。主图仍使用全宽，包括首末标注 Se 外的少量测量像素。

## 算法、平均及公平对照

81 个实测偏压 -80..80 mV，步长 2 mV。每条线、每偏压独立按实际距离减去拟合的常数和线性背景，再减残余均值、乘对称 Hann、N=99 空间 rFFT。单边幅度 abs(DFT)/sum(Hann)，奇数 N 的正频率全乘 2。没有补零、空间插值、能量插值或平滑。原生 Δf={f[1]:.9f} cycles/nm，与上一组六行完全相同。

SeMean5 是五条线先分别 FFT，再将幅度等权算术平均；没有先平均实空间数据，没有复数相干平均。SeSD5 是行间样本 SD（ddof=1），不是测量误差。复数系数另存供后续相位分析。SeMean5_relative 是先按每条线同偏压的原始沿线 LI X 均值归一化，再取五线平均；低信号时会放大噪声。

RefMean6 原样沿用上一组 67、73、79、85、91、97 的平均（含过渡行 67）；RefDark5 原样沿用 73、79、85、91、97 的平均（排除 67）。Se 五条线全部在此前 dark mask 内。RefDark5 的空间高度范围与 Se 五线略错开，不能把差异全部归为化学位点效应。

共同 Nanox 主色阶严格沿用上一组 0..{limits[1]:.9f} pA，没有按 Se 强弱重算、没有逐图自动拉伸。颜色超出上限显示为最亮色，但数据不截断。相对图、完整范围图也沿用上一组对应的独立色阶。横轴 cycles/nm，纵轴 mV，颜色为 LI Demod 1 X FFT 幅度 pA，未做绝对 dI/dV 标定。

竖线 1/(3a)={f3:.9f}、2/(3a)、1/a={fb:.9f} 仅沿用此前横向 Se 列间距的参考导引，不是已拟合峰；白线 0/10 mV 仅为关注窗口，不代表已确定的能隙。

## 文件与图

- Se_rows_frequency_energy_overview.png/pdf：五条 Se 线、SeMean5、RefMean6、RefDark5，共八幅同色阶图。
- SeR11 至 SeR15_frequency_energy.png/pdf/svg：各排单独的频率—能量图。
- SeMean5_frequency_energy：五条过 Se 线的主平均。
- averages_comparison.png/pdf：SeMean5、RefMean6、RefDark5 三图直接对照。
- Se_and_previous_topography_paths.png/pdf/svg：黄色新 Se 线、黄色空心圆为人工标注 Se；青色上一组线，67 用橙色。
- candidate_frequency_comparison.png/pdf/csv：候选 1/(3a) 两侧原生频点的逐行及均值偏压曲线，不对峰频率插值。
- 各行 raw_spectra_pA.csv、sample_positions.csv：495 条点谱及实际像素、源文件行列、实验室坐标。
- *_FFT_amplitude.csv：全部 81 个偏压和 50 个原生频率，包括 DC；SeRelative5 为无量纲，其他幅度、SD、差值为 pA。
- SeMinusRefDark5_FFT_amplitude.csv：Se 五线平均减此前 dark 五线平均，符号保留；只描述线取样差异。
- Se_rows_frequency_energy.npz：完整原始、背景、FFT、平均及上一组参照数组。
- methods_and_validation.json：来源、校准与前次结果的哈希、取样方法、色阶与验证记录。

## 验证和结论边界

独立直接读取二进制 3DS，5×99×81 个谱值最大误差 {raw_error:g} pA；上一组六行原始谱与当前同一文件逐值一致。五行平均算术误差 {average_error:g} pA，反转线方向 FFT 幅度误差 {reverse_error:.3g} pA；所有频率轴一致。原始 3DS、人工标注和此前结果均未修改。

当前仅对这些直线取样，不能视为整块区域平均或纯 Se/FeA/FeB 位点分离。幅度平均不能证明相位一致，平均不提高频率分辨率，Hann 会使频峰展宽。0–10 mV 仅有六个偏压点；本次不确认严格三倍周期、统计显著性、能隙边界或 PDW。
'''
(OUT / 'README_中文.md').write_text(readme, encoding='utf-8')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == d.hash
assert hashlib.sha256(SITE.read_bytes()).hexdigest() == meta['site_coordinate_sha256']
print(json.dumps(dict(folder=str(OUT), rows=rows.tolist(), Se_atom_rows=SE_ATOM_ROWS.tolist(), shape=amp.shape, common_scale=limits, max_Se_offset_px=max(g['maximum_vertical_offset_px'] for g in row_geometry), validation=meta['validation']), ensure_ascii=False), flush=True)
