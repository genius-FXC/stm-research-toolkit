"""Editable Origin heat maps from frozen row-85 arrays; separate Origin instance."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from origin_export import op,heatmap,title_export,sheet_csv

OUT=Path(__file__).resolve().parent/'frequency_energy'

def main():
    meta=json.loads((OUT/'row85_methods_and_validation.json').read_text(encoding='utf-8'))
    a=np.load(OUT/'row85_frequency_energy.npz')
    E=a['bias_mV'];f=a['frequency_cycles_per_nm'];s=a['distance_nm'];pal=OUT/'Nanox.pal'
    op.set_show(False);op.new()
    wb=op.new_book('w',lname='Row85 raw measured line, FFT amplitudes and frequency axis');wb.name='Row85Data'
    raw=sheet_csv(wb,'RawPixelSpectra',OUT/'row85_raw_pixel_spectra_pA.csv',first=True)
    sheet_csv(wb,'FFTAmplitude',OUT/'row85_FFT_amplitude_pA.csv')
    sheet_csv(wb,'RelativeAmplitude',OUT/'row85_FFT_relative_amplitude.csv')
    sheet_csv(wb,'FrequencyAxis',OUT/'row85_frequency_axis.csv')
    sheet_csv(wb,'ComplexFFT',OUT/'row85_FFT_complex_coefficients.csv')
    sheet_csv(wb,'PixelCoordinates',OUT/'row85_sample_positions.csv')
    refs=wb.add_sheet('ReferenceLines');refs.cols=4
    refs.from_list(0,[-80.,80.],lname='Bias',units='mV',axis='Y')
    for j,val in enumerate([meta['candidate_third_frequency_cycles_per_nm'],2*meta['candidate_third_frequency_cycles_per_nm'],meta['reference_Bragg_frequency_cycles_per_nm']],1):refs.from_list(j,[val,val],lname=['1/(3a)','2/(3a)','1/a'][j-1],units='cycles/nm',axis='X')
    loc=wb.add_sheet('PathCartesian');loc.cols=2
    loc.from_list(0,a['display_xy_px'][:,0].tolist(),lname='Display x',units='pixel',axis='X')
    loc.from_list(1,(98-a['display_xy_px'][:,1]).tolist(),lname='Cartesian y = 98 - display y',units='pixel',axis='Y')
    si=op.new_book('w',lname='Source, row 85 geometry, FFT methods and limits');si.name='SourceInfo';sh=si[0];sh.name='Methods';sh.cols=2
    sh.from_list(0,list(meta),lname='Item');sh.from_list(1,[json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else str(v) for v in meta.values()],lname='Value')
    matrices={};graphs=[]
    def matrix(name,values,xy,label):
        m=op.new_sheet('m',lname=label);m.name=name;m.from_np(values);m.xymap=xy;m.set_label(0,label);matrices[name]=(m,values,xy);return m
    ma=matrix('FreqAbs',a['fft_amplitude_pA'][:,1:],(float(f[1]),float(f[-1]),float(E[0]),float(E[-1])),'FFT amplitude (pA)')
    mr=matrix('FreqRelative',a['fft_relative_amplitude'][:,1:],(float(f[1]),float(f[-1]),float(E[0]),float(E[-1])),'Relative FFT amplitude')
    ml=matrix('RawLine',a['signal_pA'].T,(0.,float(s[-1]),float(E[0]),float(E[-1])),'LI X (pA)')
    mz=matrix('TopoPath',a['topo_plane_subtracted_pm'],(0.,98.,98.,0.),'Z (pm)')
    def format_axis(g,dx=2):
        g[0].lt_exec(f'layer.x.label.type=1;layer.x.label.numFormat=1;layer.x.label.decPlaces={dx};layer.y.label.type=1;layer.y.label.numFormat=1;layer.y.label.decPlaces=0;')
    def fft_graph(m,title,name,xrange,yrange,limits):
        g=heatmap(m,title,name,OUT,xrange,yrange,'Spatial frequency f (cycles/nm)','Bias (mV)',pal,limits=limits)
        for j,label in enumerate(['1/(3a)','2/(3a)','1/a'],1):
            p=g[0].add_plot(refs,coly=0,colx=j,type='l');p.color='#00c9c9';p.layer.SetNumProp(p._format_property('line.width'),.8)
            xx=refs.to_list(j)[0];lab=g[0].add_label(label,xx+.03,yrange[1]-.08*(yrange[1]-yrange[0]))
            if lab:lab.set_int('fsize',9);lab.set_int('color',op.ocolor('#00c9c9'))
        legend=g[0].label('Legend')
        if legend:legend.remove()
        format_axis(g);title_export(g,OUT);graphs.append(g.name)
        return g
    fft_graph(ma,'Row85 | spatial FFT at each bias | amplitude (pA)','G_Row85_FreqEnergy',(0,3.5,.5),(-30,30,10),meta['absolute_color_limits_pA'])
    fft_graph(mr,'Row85 | FFT / raw line-mean LI X | relative amplitude','G_Row85_FreqRelative',(0,3.5,.5),(-30,30,10),meta['relative_color_limits'])
    full_limits=[0.,float(np.percentile(a['fft_amplitude_pA'][:,1:],99))]
    fft_graph(ma,'Row85 | full measured bias and positive-frequency range','G_Row85_FreqEnergyFull',(0,float(f[-1]),2.5),(-80,80,20),full_limits)
    g=heatmap(ml,'Row85 | input energy-position line | raw LI X (pA)','G_Row85_EnergyPosition',OUT,(0,float(s[-1]),.5),(-30,30,10),'Distance along A to B (nm)','Bias (mV)',pal,limits=[0.,float(np.percentile(a['signal_pA'][:,(E>=-30)&(E<=30)],99))]);format_axis(g);title_export(g,OUT);graphs.append(g.name)
    zlimits=np.percentile(a['topo_plane_subtracted_pm'],[1,99]).tolist()
    g=heatmap(mz,'Row85 | A(0,85) to B(97,85) | single-pixel line','G_Row85_TopoPath',OUT,(0,98,20),(0,98,20),'Display x (pixel)','Cartesian y (=98-display y)',pal,limits=zlimits)
    p=g[0].add_plot(loc,coly=1,colx=0,type='l');p.color='#00c9c9';p.layer.SetNumProp(p._format_property('line.width'),1.7)
    for xx,label in [(0,'A'),(97,'B')]:
        lab=g[0].add_label(label,xx,16)
        if lab:lab.set_int('fsize',11);lab.set_int('color',op.ocolor('#00c9c9'))
    legend=g[0].label('Legend')
    if legend:legend.remove()
    format_axis(g,0);title_export(g,OUT);graphs.append(g.name)
    project=OUT/'row85_frequency_energy_editable.opju'
    assert op.save(str(project));assert op.open(str(project),readonly=True,asksave=False)
    saved_m={layer.name:layer for page in op.pages() if type(page).__name__=='MBook' for layer in page}
    errors={};coordinate_errors={}
    for name,(_,expected,xy) in matrices.items():
        errors[name]=float(np.max(abs(saved_m[name].to_np2d()-expected)))
        coordinate_errors[name]=float(np.max(abs(np.asarray(saved_m[name].xymap)-np.asarray(xy))))
    assert max(errors.values())==0 and max(coordinate_errors.values())<1e-9
    saved_raw=op.find_book('w','Row85Data')[0]
    raw_error=max(float(np.max(abs(np.asarray(saved_raw.to_list(j+1))-a['signal_pA'][j]))) for j in range(len(s)))
    assert raw_error==0
    # Matrix colormaps are properties of each plot, even when two share ma.
    for name,limits in [('G_Row85_FreqEnergy',meta['absolute_color_limits_pA']),('G_Row85_FreqRelative',meta['relative_color_limits']),('G_Row85_FreqEnergyFull',full_limits)]:
        p=op.find_graph(name)[0].plot_list()[0];levels=p.zlevels['levels'];assert np.allclose([levels[0],levels[-1]],limits,rtol=1e-9,atol=1e-9)
    result=dict(saved_project_reopened=True,native_graph_names=graphs,matrix_max_errors=errors,matrix_coordinate_max_errors=coordinate_errors,raw_line_spectra_max_error_pA=raw_error,plot_color_limits_preserved=True,origin_separate_instance=True,source_untouched=True)
    (OUT/'Origin_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    base_readme=(OUT/'README_中文.md').read_text(encoding='utf-8').split('\n## 可编辑 Origin 工程')[0]
    (OUT/'README_中文.md').write_text(base_readme+'\n## 可编辑 Origin 工程\n\n打开 row85_frequency_energy_editable.opju。G_Row85_FreqEnergy 为主幅度图；G_Row85_FreqRelative 为相对幅度对照；G_Row85_FreqEnergyFull 为完整频率/偏压图；G_Row85_EnergyPosition 为输入线谱；G_Row85_TopoPath 为实空间取样线。三个 FFT 图的竖线都是参考导引线，可编辑；幅度和相对幅度存为原生矩阵，坐标保存为实际 cycles/nm 与 mV。Row85Data 工作簿存原始线谱、FFT 表、频率表、复数系数和像素坐标，SourceInfo 存方法。工程保存后重新打开，全部矩阵与原始线谱最大数值误差为 0；坐标范围与固定色阶通过检查。原生 Origin 热图的绘制方式可在 Origin 中调整。Origin 图不含直接的空间相位判断。\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    try:main()
    finally:
        try:op.exit()
        except Exception:pass
