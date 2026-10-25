"""Independent native Origin project for five measured Se paths and references."""
from pathlib import Path
import hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from origin_export import op, heatmap, title_export, sheet_csv

OUT = Path(__file__).resolve().parent / 'results'


def main():
    meta = json.loads((OUT / 'methods_and_validation.json').read_text(encoding='utf-8'))
    a = np.load(OUT / 'Se_rows_frequency_energy.npz')
    previous = Path(meta['previous_results'])
    labels = [f'SeR{int(r):02d}' for r in a['Se_atom_rows']]
    rows = a['pixel_rows'].astype(int).tolist()
    refrows = a['reference_pixel_rows'].astype(int).tolist()
    E, f = a['bias_mV'], a['frequency_cycles_per_nm']
    pal = OUT / 'Nanox.pal'
    op.set_show(False)
    op.new()
    rawbook = op.new_book('w', lname='Measured spectra along Se paths and previous reference paths')
    rawbook.name = 'RawLines'
    for j, label in enumerate(labels):
        sheet_csv(rawbook, label, OUT / f'{label}_raw_spectra_pA.csv', first=j == 0)
    for row in refrows:
        sheet_csv(rawbook, f'Ref{row:03d}', previous / f'R{row:03d}_raw_spectra_pA.csv')
    wb = op.new_book('w', lname='Spatial FFT amplitudes, means, coordinates and phases')
    wb.name = 'FFTData'
    for j, label in enumerate(labels):
        sheet_csv(wb, label, OUT / f'{label}_FFT_amplitude.csv', first=j == 0)
    for key in ['SeMean5', 'RefMean6', 'RefDark5', 'SeSD5', 'SeRelative5', 'SeMinusRefDark5']:
        sheet_csv(wb, key, OUT / f'{key}_FFT_amplitude.csv')
    for name, file in [('FrequencyAxis', 'frequency_axis.csv'), ('Positions', 'sample_positions.csv'),
                       ('SeAtomOffsets', 'selected_Se_atoms_and_line_offsets.csv'),
                       ('ComplexFFT', 'complex_FFT_long.csv'), ('CandidateCurves', 'candidate_frequency_comparison.csv')]:
        sheet_csv(wb, name, OUT / file)
    refs = wb.add_sheet('ReferenceLines')
    refs.cols = 4
    refs.from_list(0, [float(E[0]), float(E[-1])], lname='Bias', units='mV', axis='Y')
    refvals = [float(a['q_third_cycles_per_nm']), 2 * float(a['q_third_cycles_per_nm']), float(a['q_bragg_cycles_per_nm'])]
    for j, (v, label) in enumerate(zip(refvals, ['1/(3a)', '2/(3a)', '1/a']), 1):
        refs.from_list(j, [v, v], lname=label, units='cycles/nm', axis='X')
    paths = wb.add_sheet('PathsCartesian')
    paths.cols = 12
    paths.from_list(0, list(range(99)), lname='Display x', units='pixel', axis='X')
    for j, (y, label) in enumerate(zip(rows + refrows, labels + [f'Ref{r:03d}' for r in refrows]), 1):
        paths.from_list(j, [98 - y] * 99, lname=label, units='pixel', axis='Y')
    atoms = wb.add_sheet('CalibratedSeCartesian')
    atoms.cols = 2
    atoms.from_list(0, a['Se_calibrated_xy_px'][:, 0].tolist(), lname='Calibrated Se x', units='pixel', axis='X')
    atoms.from_list(1, (98 - a['Se_calibrated_xy_px'][:, 1]).tolist(), lname='Cartesian Se y', units='pixel', axis='Y')
    info = op.new_book('w', lname='Source, calibrated Se paths, averaging methods and limitations')
    info.name = 'SourceInfo'
    sh = info[0]
    sh.name = 'Methods'
    sh.cols = 2
    sh.from_list(0, list(meta), lname='Item')
    sh.from_list(1, [json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v) for v in meta.values()], lname='Value')
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
        for j, (v, label) in enumerate(zip(refvals, ['1/(3a)', '2/(3a)', '1/a']), 1):
            p = g[0].add_plot(refs, coly=0, colx=j, type='l')
            p.color = '#00c9c9'
            p.layer.SetNumProp(p._format_property('line.width'), .8)
            lab = g[0].add_label(label, v + .03, yr[1] - .08 * (yr[1] - yr[0]))
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

    for j, label in enumerate(labels):
        m = matrix(f'F{label}', a['fft_amplitude_pA'][j, :, 1:], xy, 'FFT amplitude (pA)')
        fft_graph(m, f'{label} | display row {rows[j]} | x=0..98 | near-Se path', f'G_{label}_FE', meta['absolute_color_limits_pA'])
    mmean = matrix('FSeMean5', a['SeMean5_amplitude_pA'][:, 1:], xy, 'FFT amplitude (pA)')
    fft_graph(mmean, 'Mean of five near-Se path FFT amplitudes | rows 70,76,82,88,94', 'G_SeMean5_FE', meta['absolute_color_limits_pA'])
    for key, title in [('RefMean6', 'Previous mean of six rows | 67,73,79,85,91,97'), ('RefDark5', 'Previous dark mean of five rows | 73,79,85,91,97')]:
        m = matrix(f'F{key}', a[f'{key}_amplitude_pA'][:, 1:], xy, 'FFT amplitude (pA)')
        fft_graph(m, title, f'G_{key}_FE', meta['absolute_color_limits_pA'])
    mr = matrix('FSeRelative', a['SeMean5_relative'][:, 1:], xy, 'Relative FFT amplitude')
    fft_graph(mr, 'SeMean5 | per-line FFT / raw line-mean LI X | relative amplitude', 'G_SeMean5_Rel', meta['relative_color_limits'])
    fft_graph(mmean, 'SeMean5 | full measured bias and positive-frequency range', 'G_SeMean5_Full', meta['full_color_limits_pA'], full=True)
    mz = matrix('TopoPaths', a['topo_plane_subtracted_pm'], (0., 98., 98., 0.), 'Z (pm)')
    topo_limits = np.percentile(a['topo_plane_subtracted_pm'], [1, 99]).tolist()
    g = heatmap(mz, 'Yellow: Se paths; cyan: previous paths; orange: transition row 67', 'G_Se_TopoPaths', OUT, (0, 98, 20), (0, 38, 5), 'Display x (pixel)', 'Cartesian y (=98-display y)', pal, limits=topo_limits)
    for j, (y, label) in enumerate(zip(rows + refrows, labels + [f'Ref{r:03d}' for r in refrows]), 1):
        color = '#ffd400' if j <= 5 else '#ff9e32' if y == 67 else '#00c9c9'
        p = g[0].add_plot(paths, coly=j, colx=0, type='l')
        p.color = color
        p.layer.SetNumProp(p._format_property('line.width'), .9)
        lab = g[0].add_label(f'{label} / y={y}', 2., float(98 - y) + .7)
        if lab:
            lab.set_int('fsize', 8)
            lab.set_int('color', op.ocolor(color))
    legend = g[0].label('Legend')
    if legend:
        legend.remove()
    axes(g, 0)
    title_export(g, OUT)
    graphs.append(g.name)
    graphlimits[g.name] = topo_limits
    project = OUT / 'Se_rows_frequency_energy_comparison_editable.opju'
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
    for prefix, names, raw in [('', labels, a['raw_signal_pA']), ('', [f'Ref{r:03d}' for r in refrows], a['reference_raw_signal_pA'])]:
        for j, name in enumerate(names):
            sh = next(sh for sh in rawbook2 if sh.name == prefix + name)
            rawerrors[name] = max(float(np.max(np.abs(np.asarray(sh.to_list(k + 1)) - raw[j, k]))) for k in range(99))
    assert max(rawerrors.values()) == 0
    saved_mean = np.mean([saved_m[f'F{label}'].to_np2d() for label in labels], axis=0)
    avgerror = float(np.max(np.abs(saved_m['FSeMean5'].to_np2d() - saved_mean)))
    assert avgerror == 0
    for name in graphs:
        gp = op.find_graph(name)
        assert gp and gp.name == name
        assert len(gp[0].plot_list()) == (12 if name == 'G_Se_TopoPaths' else 4)
        levels = gp[0].plot_list()[0].zlevels['levels']
        assert np.allclose([levels[0], levels[-1]], graphlimits[name], rtol=1e-9, atol=1e-9)
    source_unchanged = hashlib.sha256(Path(meta['source_file']).read_bytes()).hexdigest() == meta['source_sha256']
    sites_unchanged = hashlib.sha256(Path(meta['site_coordinate_file']).read_bytes()).hexdigest() == meta['site_coordinate_sha256']
    assert source_unchanged and sites_unchanged
    result = dict(saved_project_reopened=True, native_graph_names=graphs, matrix_max_errors=errors, matrix_coordinate_max_errors=xyerrors,
                  raw_spectra_max_errors_pA=rawerrors, saved_mean5_error_pA=avgerror, fixed_previous_color_limits_verified=True,
                  independent_origin_instance=True, source_hash_unchanged=source_unchanged, manual_Se_coordinates_unchanged=sites_unchanged)
    (OUT / 'Origin_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    readme = (OUT / 'README_中文.md').read_text(encoding='utf-8').split('\n## 可编辑 Origin 工程')[0]
    readme += '\n## 可编辑 Origin 工程\n\n打开 `Se_rows_frequency_energy_comparison_editable.opju`。所有图为原生矩阵热图与参考线，色阶、坐标和标注可编辑。\n\n| 图名 | 内容 |\n|---|---|\n'
    for label, row in zip(labels, rows):
        readme += f'| `G_{label}_FE` | {label}、显示像素行 {row}，过该排 Se 的直线空间 FFT |\n'
    readme += '| `G_SeMean5_FE` | 五条过 Se 线的主幅度平均 |\n| `G_RefMean6_FE` | 上一组指定六行平均，包含过渡行 67 |\n| `G_RefDark5_FE` | 上一组排除 67 的五行平均，便于 dark 内对照 |\n| `G_SeMean5_Rel` | 过 Se 五线的相对幅度平均 |\n| `G_SeMean5_Full` | 过 Se 五线平均的完整偏压和正频率 |\n| `G_Se_TopoPaths` | 新五条线与原六条线的位置；纵坐标为 98−显示像素行 |\n\n'
    readme += '`RawLines` 包含新 495 条和上一组 594 条原始点谱。`FFTData` 包含各行幅度、均值、行间 SD、差值、原生频率、复数系数、像素位置及 55 个标注 Se 的线位置偏差。`SourceInfo` 保存完整方法和来源。\n\n保存后重新打开工程，所有矩阵、1089 条原始点谱和主平均数值误差为 0；频率/偏压坐标和沿用的固定色阶通过检查。原始 3DS 与手工 Se 标注哈希均未改变。\n'
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
