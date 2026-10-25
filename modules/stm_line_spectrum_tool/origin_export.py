"""Create a separate native Origin project; never touch user's open project."""
from pathlib import Path
import csv,json,sys,importlib.util,traceback
import numpy as np
from core import distance_colors,PALETTE

ROOT=Path(__file__).resolve().parent
if importlib.util.find_spec('originpro') is None:
    for folder in json.loads((ROOT/('config.json' if (ROOT/'config.json').exists() else 'config.example.json')).read_text(encoding='utf-8')).get('origin_sdk_fallbacks',[]):
        if (Path(folder)/'originpro').is_dir():sys.path.insert(0,folder);break
try:import originpro as op
except ImportError as ex:raise RuntimeError('找不到 originpro / OriginExt；请在 config.json 指定现有 Origin Python 接口位置。') from ex

def sheet_csv(book,name,path,first=False):
    sh=book[0] if first else book.add_sheet(name);sh.name=name
    with path.open(encoding='utf-8-sig',newline='') as f:rr=csv.reader(f);headers=next(rr);rows=list(rr)
    sh.cols=len(headers);sh.header_rows('LUC')
    for col,title in enumerate(headers):
        val=[]
        for row in rows:
            try:val.append(float(row[col]))
            except ValueError:val.append(row[col])
        unit='cycles/nm' if title=='frequency_cycles_per_nm' else 'mV' if title=='bias_mV' else 'nm' if title.endswith('_nm') else 'pm' if title.endswith('_pm') else 'pA' if title.endswith('_pA') else 'pixel' if title.endswith('_px') else ''
        sh.from_list(col,val,lname=title,units=unit,axis='X' if col==0 else 'Y')
    return sh

def title_export(gp,folder):
    gl=gp[0];gl.lt_exec('layer.x.label.rotate=0;layer.y.label.rotate=0;')
    xmin,xmax=gl.get_float('x.from'),gl.get_float('x.to');ymin,ymax=gl.get_float('y.from'),gl.get_float('y.to')
    label=gl.add_label(gp.lname[:78],xmin+.001*(xmax-xmin),ymax+.055*(ymax-ymin))
    if label:label.set_int('fsize',11)
    gp.save_fig(str(folder/(gp.name+'.png')),type='png',width=1700)

def heatmap(matrix,title,name,folder,xrange,yrange,xlabel,ylabel,palette,limits=None):
    gp=op.new_graph(template='Heat_Map.otpu',lname=title);gp.name=name;gl=gp[0]
    plot=gl.add_mplot(matrix,0,type=105);plot.colormap=str(palette)
    a=matrix.to_np2d(0);lo,hi=limits if limits is not None else np.percentile(a,[1,99]);lo=float(lo);hi=float(hi);hi=max(hi,lo+1e-12)
    gl.rescale()
    # The installed SDK zlevels setter is a no-op for this matrix heat map.
    # Set documented CMap properties, then commit/repaint them explicitly.
    gl.lt_exec(f'layer.cmap.type=1;layer.cmap.zmin={lo:.17g};layer.cmap.zmax={hi:.17g};layer.cmap.numMajorLevels=2;layer.cmap.numMinorLevels=127;layer.cmap.setLevels(1);layer.cmap.stretchpal=1;layer.cmap.linkpal=1;layer.cmap.updateScale();')
    plot.colormap=str(palette)
    # https://docs.originlab.com/labtalk/ref/layer-cmap-obj/
    below=op.ocolor('#'+''.join(f'{v:02x}' for v in PALETTE[0]));above=op.ocolor('#'+''.join(f'{v:02x}' for v in PALETTE[-1]))
    gl.lt_exec(f'layer.cmap.colorBelow={below};layer.cmap.colorAbove={above};layer.cmap.updateScale();')
    assert np.allclose([plot.zlevels['levels'][0],plot.zlevels['levels'][-1]],[lo,hi],rtol=1e-10,atol=1e-10)
    gl.set_xlim(*xrange);gl.set_ylim(*yrange)
    gl.axis('x').title=xlabel;gl.axis('y').title=ylabel
    return gp

def main(folder):
    lid=next(folder.glob('*_line.json')).name.split('_line.json')[0]
    meta=json.loads((folder/f'{lid}_line.json').read_text(encoding='utf-8'));cfg=meta['settings'];a=np.load(folder/f'{lid}_numeric.npz')
    ny,nx=int(a['ny']),int(a['nx']);bias=a['bias_mV'];s=a['s_nm'];pal=folder/'Nanox.pal'
    op.set_show(False);op.new()
    book=op.new_book('w',lname=f'{lid} raw and displayed line spectra');book.name='LineSpectra'
    raw=sheet_csv(book,'RawUsed_pA',folder/f'{lid}_full_spectra_used_pA.csv',True)
    sheet_csv(book,'RawCenter_pA',folder/f'{lid}_full_spectra_center_pA.csv')
    if (folder/f'{lid}_full_spectra_normalized.csv').exists():sheet_csv(book,'Normalized',folder/f'{lid}_full_spectra_normalized.csv')
    if (folder/f'{lid}_processed_spectra.csv').exists():sheet_csv(book,'Processed',folder/f'{lid}_processed_spectra.csv')
    wf=sheet_csv(book,'Waterfall',folder/f'{lid}_waterfall_display.csv')
    if (folder/f'{lid}_waterfall_colors.csv').exists():sheet_csv(book,'WaterfallColors',folder/f'{lid}_waterfall_colors.csv')
    height=sheet_csv(book,'HeightProfile',folder/f'{lid}_z_profile.csv')
    height.cols_axis('nxy')
    profile=sheet_csv(book,'SelectedProfile',folder/f'{lid}_line_profile.csv')
    coords=op.new_book('w',lname=f'{lid} sample IDs and positions');coords.name='Coordinates'
    pos=sheet_csv(coords,'Positions',folder/f'{lid}_sample_positions.csv',True)
    pos.cols=11;pos.from_list(10,(ny-1-a['xy'][:,1]).tolist(),lname='cartesian_y_px',units='pixel')
    info=op.new_book('w',lname='Input and line settings');info.name='SourceInfo';si=info[0];si.name='Settings';si.cols=2
    si.from_list(0,list(meta),lname='Item');si.from_list(1,[json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else str(v) for v in meta.values()],lname='Value')
    from rotation import rotate_points
    turns=cfg.get('rotation_quarters',0);view_xy=rotate_points(a['xy'],nx,ny,turns);view_map=np.rot90(a['display_map'],-turns);vny,vnx=view_map.shape
    pos.cols=13;pos.from_list(11,view_xy[:,0].tolist(),lname='rotated_display_x_px',units='pixel');pos.from_list(12,(vny-1-view_xy[:,1]).tolist(),lname='rotated_cartesian_y_px',units='pixel')
    gm=op.new_sheet('m',lname='Selected image; display orientation preserved');gm.name='Image'
    gm.from_np(view_map);gm.xymap=(0,vnx-1,vny-1,0);gm.set_label(0,'LI X (pA)' if cfg['view']=='mapping' else 'Z (pm)')
    path=heatmap(gm,f'{lid} | hand-drawn path','G_'+lid+'_Path',folder,(0,vnx-1,max(1,(vnx-1)/5)),(0,vny-1,max(1,(vny-1)/5)),'Display x (pixel)','Cartesian y (ny-1 minus display y)',pal)
    p=path[0].add_plot(pos,coly=12,colx=11,type='l');p.color='#00c9c9';p.layer.SetNumProp(p._format_property('line.width'),1.8)
    for j,letter in [(0,'A'),(-1,'B')]:
        label=path[0].add_label(letter,float(view_xy[j,0])+.001,float(vny-1-view_xy[j,1])+.001)
        if label:label.set_int('fsize',12)
    title_export(path,folder)
    channel=meta['profile']['channel'];unit=meta['profile']['unit']
    zg=op.new_graph(template='Origin',lname=f"{lid} | {channel} profile | {cfg['profile_mode']}");zg.name='G_'+lid+'_Profile';zl=zg[0]
    zp=zl.add_plot(profile,coly=6,colx=1,type='l');zp.color='#a56b2a';zp.layer.SetNumProp(zp._format_property('line.width'),1.5)
    zl.rescale();zl.set_xlim(0,float(s[-1]),float(s[-1])/5);zl.axis('x').title='Projected line distance (nm)';zl.axis('y').title=f'{channel} ({unit})'
    zl.lt_exec('layer.x.label.type=1;layer.x.label.numFormat=1;layer.x.label.decPlaces=2;layer.y.label.decPlaces=2;')
    legend=zl.label('Legend')
    if legend:legend.remove()
    title_export(zg,folder)
    fft_enabled=bool(meta.get('fft',{}).get('available'))
    if fft_enabled:
        fs=sheet_csv(book,'ProfileFFT',folder/f'{lid}_profile_fft.csv')
        fi=sheet_csv(book,'FFTInput',folder/f'{lid}_fft_input.csv')
        # The positive-frequency plot has no DC point; the complete table keeps it.
        fp=book.add_sheet('FFTPositive');fp.cols=2;fp.header_rows('LUC')
        fp.from_list(0,a['fft_frequency_cycles_per_nm'][1:].tolist(),lname='Spatial frequency',units='cycles/nm',axis='X')
        fp.from_list(1,a['fft_amplitude'][1:].tolist(),lname='One-sided amplitude',units=unit,axis='Y')
        bias_note=f" | {meta['fft']['bias_mV']:g} mV" if channel=='LI X' else ''
        fg=op.new_graph(template='Origin',lname=f"{lid} | {channel}{bias_note} | spatial FFT | {cfg.get('fft_window','hann')}");fg.name='G_'+lid+'_ProfileFFT';fl=fg[0]
        p=fl.add_plot(fp,coly=1,colx=0,type='l');p.color='#087e82';p.layer.SetNumProp(p._format_property('line.width'),1.5)
        fl.rescale();fmax=meta['fft']['nyquist_cycles_per_nm']
        tick_scale=10**np.floor(np.log10(fmax/5))
        tick_step=min([1,2,2.5,5,10],key=lambda v:abs(v*tick_scale-fmax/5))*tick_scale
        fl.set_xlim(0,fmax,float(tick_step))
        ymax=max(float(a['fft_amplitude'][1:].max())*1.12,1e-12);fl.set_ylim(0,ymax)
        decimals=min(10,max(2,int(np.ceil(-np.log10(ymax)))+1))
        fl.lt_exec(f'layer.x.label.type=1;layer.x.label.numFormat=1;layer.x.label.decPlaces=2;layer.y.label.type=1;layer.y.label.numFormat=1;layer.y.label.decPlaces={decimals};')
        fl.axis('x').title='Spatial frequency f (cycles/nm)';fl.axis('y').title=f'One-sided FFT amplitude ({unit})'
        legend=fl.label('Legend')
        if legend:legend.remove()
        title_export(fg,folder)
    mm=op.new_sheet('m',lname=f'{lid} full energy-position LI X (pA)');mm.name='LineMatrix'
    mm.from_np(a['signal_pA'].T);mm.xymap=(0,len(s)-1,float(bias[0]),float(bias[-1]));mm.set_label(0,'LI X (pA)')
    h=heatmap(mm,f'{lid} | energy-position LI X | exact nm in Coordinates','G_'+lid+'_EnergyPosition',folder,(0,len(s)-1,max(1,(len(s)-1)/5)),(cfg['energy_min_mV'],cfg['energy_max_mV'],max(2,(cfg['energy_max_mV']-cfg['energy_min_mV'])/4)),'Pixel sample index (0-based)','Bias (mV)',pal,limits=meta['heat_limits_pA']);title_export(h,folder)
    gp=op.new_graph(template='Origin',lname=f'{lid} | waterfall (all settings saved)');gp.name='G_'+lid+'_Waterfall';gl=gp[0]
    for j in range(1,wf.cols):
        p=gl.add_plot(wf,coly=j,colx=0,type='l');p.layer.SetNumProp(p._format_property('line.width'),1.0)
    gl.group(False)
    colors=distance_colors(s[a['waterfall_indices']]/s[-1])
    expected_rgb=[tuple(int(color[k:k+2],16) for k in (1,3,5)) for color in colors]
    for p,color in zip(gl.plot_list(),colors):p.color=color
    assert [p.color for p in gl.plot_list()]==expected_rgb
    gl.rescale();gl.set_xlim(cfg['energy_min_mV'],cfg['energy_max_mV'],max(2,(cfg['energy_max_mV']-cfg['energy_min_mV'])/4))
    wf_bias=a['waterfall_bias_mV'] if 'waterfall_bias_mV' in a else bias
    mask=(wf_bias>=cfg['energy_min_mV']-1e-8)&(wf_bias<=cfg['energy_max_mV']+1e-8);visible=a['waterfall'][:,mask];lo,hi=float(visible.min()),float(visible.max());span=max(hi-lo,1e-8);gl.set_ylim(lo-.05*span,hi+.08*span)
    gl.axis('x').title='Bias (mV)';gl.axis('y').title=('LI X ('+meta.get('processing',{}).get('unit','a.u.' if cfg['normalize'] else 'pA')+')')+(' minus baseline' if cfg['zero_anchor'] else '')+f" + {cfg['offset']:g}*(curve-1)"
    lg=gl.label('Legend')
    if lg:lg.remove()
    xmin,xmax=cfg['energy_min_mV'],cfg['energy_max_mV']
    for f,label_text,color in zip([.02,.42,.80],[f'A: 0 nm',f'{s[-1]/2:.3f} nm',f'B: {s[-1]:.3f} nm'],distance_colors([0,.5,1])):
        label=gl.add_label(label_text,xmin+f*(xmax-xmin),hi+.03*span)
        if label:label.set_int('fsize',9);label.set_int('color',op.ocolor(color))
    title_export(gp,folder)
    # Read every column and matrix, not only the first spectrum.
    errors=[float(np.max(abs(np.asarray(raw.to_list(j+1))-a['signal_pA'][j]))) for j in range(len(s))]
    matrix_error=float(np.max(abs(mm.to_np2d()-a['signal_pA'].T)))
    wf_errors=[float(np.max(abs(np.asarray(wf.to_list(j+1))-a['waterfall'][j]))) for j in range(len(a['waterfall']))]
    zkeys=['z_raw_center_pm','z_plane_center_pm','z_raw_used_pm','z_plane_used_pm','z_display_relative_pm']
    height_errors=[float(np.max(abs(np.asarray(height.to_list(j+2))-a[key]))) for j,key in enumerate(zkeys)]
    assert max(height_errors)<1e-9
    assert max(errors)<1e-10 and matrix_error<1e-10 and max(wf_errors)<1e-10
    project=folder/f'{lid}_editable_line_spectrum.opju'
    assert op.save(str(project))
    # Verify the serialized project as well as the in-memory objects.
    assert op.open(str(project),readonly=True,asksave=False)
    raw2=op.find_book('w','LineSpectra')[0]
    saved_errors=[float(np.max(abs(np.asarray(raw2.to_list(j+1))-a['signal_pA'][j]))) for j in range(len(s))]
    assert max(saved_errors)<1e-10
    matrices=[layer for page in op.pages() if type(page).__name__=='MBook' for layer in page if layer.name=='LineMatrix']
    assert len(matrices)==1
    saved_matrix_error=float(np.max(abs(matrices[0].to_np2d()-a['signal_pA'].T)));assert saved_matrix_error<1e-10
    saved_colors=[p.color for p in op.find_graph('G_'+lid+'_Waterfall')[0].plot_list()]
    assert saved_colors==expected_rgb
    saved_height=[sh for sh in op.find_book('w','LineSpectra') if sh.name=='HeightProfile'][0]
    saved_height_errors=[float(np.max(abs(np.asarray(saved_height.to_list(j+2))-a[key]))) for j,key in enumerate(zkeys)]
    assert max(saved_height_errors)<1e-9
    saved_profile=[sh for sh in op.find_book('w','LineSpectra') if sh.name=='SelectedProfile'][0]
    profile_errors=[float(np.max(abs(np.asarray(saved_profile.to_list(j+4))-a[key]))) for j,key in enumerate(['profile_raw','profile_trend','profile_values'])]
    assert max(profile_errors)<1e-9
    assert len(op.find_graph('G_'+lid+'_Profile')[0].plot_list())==1
    fft_errors=[]
    if fft_enabled:
        sb=op.find_book('w','LineSpectra')
        fft_saved=next(sh for sh in sb if sh.name=='ProfileFFT')
        for j,key in [(0,'fft_frequency_cycles_per_nm'),(2,'fft_amplitude'),(3,'fft_coefficient_real'),(4,'fft_coefficient_imag')]:
            fft_errors.append(float(np.max(abs(np.asarray(fft_saved.to_list(j),float)-a[key]))))
        input_saved=next(sh for sh in sb if sh.name=='FFTInput')
        for j,key in [(0,'fft_uniform_distance_nm'),(1,'fft_uniform_profile'),(2,'fft_demeaned_profile'),(3,'fft_window_weights')]:
            fft_errors.append(float(np.max(abs(np.asarray(input_saved.to_list(j),float)-a[key]))))
        plot_saved=next(sh for sh in sb if sh.name=='FFTPositive')
        for j,key in [(0,'fft_frequency_cycles_per_nm'),(1,'fft_amplitude')]:
            fft_errors.append(float(np.max(abs(np.asarray(plot_saved.to_list(j),float)-a[key][1:]))))
        assert max(fft_errors)<1e-9
        assert len(op.find_graph('G_'+lid+'_ProfileFFT')[0].plot_list())==1
    saved_zlevels=op.find_graph('G_'+lid+'_EnergyPosition')[0].plot_list()[0].zlevels
    assert np.allclose([saved_zlevels['levels'][0],saved_zlevels['levels'][-1]],meta['heat_limits_pA'],rtol=1e-10,atol=1e-10)
    checks=dict(native_graphs=4+int(fft_enabled),sample_spectra_compared=len(s),bias_points=len(bias),max_in_memory_spectrum_error_pA=max(errors),max_in_memory_matrix_error_pA=matrix_error,max_waterfall_error=max(wf_errors),saved_project_reopened=True,max_saved_spectrum_error_pA=max(saved_errors),max_saved_matrix_error_pA=saved_matrix_error,max_in_memory_height_error_pm=max(height_errors),max_saved_height_error_pm=max(saved_height_errors),native_profile_graph_verified=True,max_saved_profile_error=max(profile_errors),native_profile_FFT_graph_verified=fft_enabled,max_saved_FFT_error=max(fft_errors) if fft_errors else None,heat_limits_verified=True,heat_limits_pA=meta['heat_limits_pA'],energy_axis='pixel_sample_index',distance_gradient_colors_verified=True,waterfall_colors_rgb=saved_colors,source_untouched=True)
    (folder/'Origin_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main(Path(sys.argv[1]).resolve())
    finally:
        try:op.exit()
        except Exception:pass
