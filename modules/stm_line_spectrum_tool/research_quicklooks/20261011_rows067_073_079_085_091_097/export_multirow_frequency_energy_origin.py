"""Native editable Origin graphs from frozen six-row arrays; independent instance."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from origin_export import op, heatmap, title_export, sheet_csv

OUT = Path(__file__).resolve().parent / 'results'


def main():
    meta = json.loads((OUT / 'methods_and_validation.json').read_text(encoding='utf-8'))
    a = np.load(OUT / 'six_rows_frequency_energy.npz')
    rows = a['pixel_rows'].astype(int).tolist()
    E, f = a['bias_mV'], a['frequency_cycles_per_nm']
    pal = OUT / 'Nanox.pal'
    op.set_show(False)
    op.new()
    rawbook = op.new_book('w', lname='Six horizontal rows: original measured LI X spectra (pA)')
    rawbook.name = 'RawLines'
    for j, row in enumerate(rows):
        sheet_csv(rawbook, f'R{row:03d}', OUT / f'R{row:03d}_raw_spectra_pA.csv', first=j == 0)
    wb = op.new_book('w', lname='Spatial FFT amplitudes, averages, row SD and provenance')
    wb.name = 'FFTData'
    for j, row in enumerate(rows):
        sheet_csv(wb, f'FFT_R{row:03d}', OUT / f'R{row:03d}_FFT_amplitude_pA.csv', first=j == 0)
    for name, file in [('Mean6', 'Mean6_FFT_amplitude_pA.csv'),
                       ('Dark5', 'Dark5_FFT_amplitude_pA.csv'),
                       ('Relative6', 'Mean6_relative.csv'),
                       ('RowSD6', 'RowSD6_pA.csv'),
                       ('PowerMean6', 'Mean6_power_pA2.csv'),
                       ('FrequencyAxis', 'frequency_axis.csv'),
                       ('Positions', 'sample_positions.csv'),
                       ('ComplexFFT', 'complex_FFT_long.csv'),
                       ('CandidateCurves', 'candidate_adjacent_frequency_energy_curves.csv')]:
        sheet_csv(wb, name, OUT / file)
    refs = wb.add_sheet('ReferenceLines')
    refs.cols = 4
    refs.from_list(0, [float(E[0]), float(E[-1])], lname='Bias', units='mV', axis='Y')
    refvals = [float(a['q_third_cycles_per_nm']), 2 * float(a['q_third_cycles_per_nm']), float(a['q_bragg_cycles_per_nm'])]
    reflabels = ['1/(3a)', '2/(3a)', '1/a']
    for j, (v, label) in enumerate(zip(refvals, reflabels), 1):
        refs.from_list(j, [v, v], lname=label, units='cycles/nm', axis='X')
    paths = wb.add_sheet('PathsCartesian')
    paths.cols = 7
    paths.from_list(0, list(range(99)), lname='Display x', units='pixel', axis='X')
    for j, row in enumerate(rows, 1):
        paths.from_list(j, [98 - row] * 99, lname=f'R{row:03d}: Cartesian y = 98 - display row', units='pixel', axis='Y')
    info = op.new_book('w', lname='Source hash, coordinates, FFT and averaging definitions')
    info.name = 'SourceInfo'
    si = info[0]
    si.name = 'Methods'
    si.cols = 2
    si.from_list(0, list(meta), lname='Item')
    si.from_list(1, [json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v) for v in meta.values()], lname='Value')
    matrices, graphs, graphlimits = {}, [], {}

    def matrix(name, values, xy, label):
        m = op.new_sheet('m', lname=label)
        m.name = name
        m.from_np(values)
        m.xymap = xy
        m.set_label(0, label)
        matrices[name] = (values, xy)
        return m

    xy = (float(f[1]), float(f[-1]), float(E[0]), float(E[-1]))

    def axes(g, decimals=2):
        g[0].lt_exec(f'layer.x.label.type=1;layer.x.label.numFormat=1;layer.x.label.decPlaces={decimals};layer.y.label.type=1;layer.y.label.numFormat=1;layer.y.label.decPlaces=0;')

    def fft_graph(m, title, name, limits, full=False):
        xr = (0, float(f[-1]), 2.5) if full else (0, 3.5, .5)
        yr = (-80, 80, 20) if full else (-30, 30, 10)
        g = heatmap(m, title, name, OUT, xr, yr, 'Spatial frequency f (cycles/nm)', 'Bias (mV)', pal, limits=limits)
        assert g.name == name
        for j, (val, label) in enumerate(zip(refvals, reflabels), 1):
            p = g[0].add_plot(refs, coly=0, colx=j, type='l')
            p.color = '#00c9c9'
            p.layer.SetNumProp(p._format_property('line.width'), .8)
            lab = g[0].add_label(label, val + .03, yr[1] - .08 * (yr[1] - yr[0]))
            if lab:
                lab.set_int('fsize', 9)
                lab.set_int('color', op.ocolor('#00c9c9'))
        legend = g[0].label('Legend')
        if legend:
            legend.remove()
        axes(g)
        title_export(g, OUT)
        graphs.append(name)
        graphlimits[name] = list(limits)
        print(f'Created {name}', flush=True)
        return g

    for j, row in enumerate(rows):
        m = matrix(f'F{row:03d}', a['fft_amplitude_pA'][j, :, 1:], xy, 'FFT amplitude (pA)')
        region = 'transition' if row == 67 else 'dark'
        fft_graph(m, f'Row {row} ({region}) | x=0..98 | one-sided FFT amplitude', f'G_R{row:03d}_FE', meta['absolute_color_limits_pA'])
    m6 = matrix('FMean6', a['mean6_amplitude_pA'][:, 1:], xy, 'FFT amplitude (pA)')
    fft_graph(m6, 'Mean of six per-row FFT amplitudes | rows 67,73,79,85,91,97', 'G_Mean6_FE', meta['absolute_color_limits_pA'])
    md = matrix('FDark5', a['dark5_amplitude_pA'][:, 1:], xy, 'FFT amplitude (pA)')
    fft_graph(md, 'Mean of five dark-row FFT amplitudes | rows 73,79,85,91,97', 'G_Dark5_FE', meta['absolute_color_limits_pA'])
    mr = matrix('FRel6', a['mean6_relative_amplitude'][:, 1:], xy, 'Relative FFT amplitude')
    fft_graph(mr, 'Mean6 | each row FFT / its raw line-mean LI X | relative amplitude', 'G_Mean6_Rel', meta['relative_color_limits'])
    fft_graph(m6, 'Mean6 | full measured bias and positive-frequency range', 'G_Mean6_Full', meta['full_color_limits_pA'], full=True)
    mz = matrix('TopoRows', a['topo_plane_subtracted_pm'], (0., 98., 98., 0.), 'Z (pm)')
    topo_limits = np.percentile(a['topo_plane_subtracted_pm'], [1, 99]).tolist()
    g = heatmap(mz, 'Six measured rows | orange: transition R067; cyan: dark rows', 'G_Rows_Topo', OUT, (0, 98, 20), (0, 98, 20), 'Display x (pixel)', 'Cartesian y (=98-display y)', pal, limits=topo_limits)
    for j, row in enumerate(rows, 1):
        color = '#ff9e32' if row == 67 else '#00c9c9'
        p = g[0].add_plot(paths, coly=j, colx=0, type='l')
        p.color = color
        p.layer.SetNumProp(p._format_property('line.width'), 1.2)
        lab = g[0].add_label(f'R{row:03d}', 3., float(98 - row) + 1.3)
        if lab:
            lab.set_int('fsize', 9)
            lab.set_int('color', op.ocolor(color))
    legend = g[0].label('Legend')
    if legend:
        legend.remove()
    axes(g, 0)
    title_export(g, OUT)
    graphs.append(g.name)
    graphlimits[g.name] = topo_limits
    project = OUT / 'six_rows_frequency_energy_editable.opju'
    assert op.save(str(project))
    assert op.open(str(project), readonly=True, asksave=False)
    saved_m = {layer.name: layer for page in op.pages() if type(page).__name__ == 'MBook' for layer in page}
    errors, xyerrors = {}, {}
    for name, (expected, expected_xy) in matrices.items():
        errors[name] = float(np.max(np.abs(saved_m[name].to_np2d() - expected)))
        xyerrors[name] = float(np.max(np.abs(np.asarray(saved_m[name].xymap) - expected_xy)))
    assert max(errors.values()) == 0 and max(xyerrors.values()) < 1e-9
    rawbook2 = op.find_book('w', 'RawLines')
    rawerrors = {}
    for j, row in enumerate(rows):
        sh = next(sh for sh in rawbook2 if sh.name == f'R{row:03d}')
        rawerrors[str(row)] = max(float(np.max(np.abs(np.asarray(sh.to_list(k + 1)) - a['raw_signal_pA'][j, k]))) for k in range(99))
    assert max(rawerrors.values()) == 0
    saved_avg = np.mean([saved_m[f'F{row:03d}'].to_np2d() for row in rows], axis=0)
    saved_dark = np.mean([saved_m[f'F{row:03d}'].to_np2d() for row in rows[1:]], axis=0)
    avgerror = float(np.max(np.abs(saved_m['FMean6'].to_np2d() - saved_avg)))
    darkerror = float(np.max(np.abs(saved_m['FDark5'].to_np2d() - saved_dark)))
    assert max(avgerror, darkerror) < 1e-10
    for name in graphs:
        gp = op.find_graph(name)
        assert gp and gp.name == name
        assert len(gp[0].plot_list()) == (7 if name == 'G_Rows_Topo' else 4)
        levels = gp[0].plot_list()[0].zlevels['levels']
        assert np.allclose([levels[0], levels[-1]], graphlimits[name], rtol=1e-9, atol=1e-9)
    source_unchanged = hashlib.sha256(Path(meta['source_file']).read_bytes()).hexdigest() == meta['source_sha256']
    assert source_unchanged
    result = dict(saved_project_reopened=True, native_graph_names=graphs, matrix_max_errors=errors,
                  matrix_coordinate_max_errors=xyerrors, raw_spectra_max_errors_pA=rawerrors,
                  saved_mean6_error_pA=avgerror, saved_dark5_error_pA=darkerror,
                  fixed_color_limits_verified=True, positive_frequency_bins=49,
                  independent_origin_instance=True, source_hash_unchanged=source_unchanged)
    (OUT / 'Origin_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    readme = (OUT / 'README_中文.md').read_text(encoding='utf-8').split('\n## 可编辑 Origin 工程')[0]
    readme += '\n## 可编辑 Origin 工程\n\n打开 `six_rows_frequency_energy_editable.opju`。所有图均由原生 Origin 矩阵和曲线绘制，可编辑色阶、坐标、参考线与标注。\n\n'
    readme += '| Origin 图名 | 内容 |\n|---|---|\n'
    for row in rows:
        region = '此前分区的过渡区' if row == 67 else '此前分区的 dark 区'
        readme += f'| `G_R{row:03d}_FE` | 第 {row} 行，{region}；全宽 x=0–98 的逐偏压空间 FFT 幅度 |\n'
    readme += '| `G_Mean6_FE` | 用户指定六行的 FFT 幅度算术平均，主结果 |\n| `G_Dark5_FE` | 去除过渡区第 67 行后，五条 dark 线的幅度平均，对照结果 |\n| `G_Mean6_Rel` | 各行按同偏压原始线均值归一化后，再取六行平均；独立色阶 |\n| `G_Mean6_Full` | 六行幅度平均的全部偏压和正频率范围；独立色阶 |\n| `G_Rows_Topo` | 六条取样线的位置；纵坐标为 98−显示行号，橙色 67 行、青色其余行 |\n\n'
    readme += '`RawLines` 保存六条线的 594 条原始点谱；`FFTData` 保存逐行 FFT、平均值、行间 SD、复数系数、相邻候选频率曲线和像素坐标；`SourceInfo` 保存完整方法和来源。矩阵 `F067` 至 `F097`、`FMean6`、`FDark5` 仅用于绘图去掉了 DC 列，CSV/工作表保留 DC。主图和各行图使用同一固定色阶。\n\n保存后重新打开工程，全部原始点谱与矩阵数值误差为 0；坐标、固定色阶和平均关系通过检查。原始 3DS 哈希保持不变。工程由独立 Origin 实例生成。\n'
    (OUT / 'README_中文.md').write_text(readme, encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            op.exit()
        except Exception:
            pass
