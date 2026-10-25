"""Editable Origin peak-position and half-depth-width graphs, with readback QA."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from origin_export import op,title_export,sheet_csv
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';TABLE=OUT/'04_tables';DEST=OUT/'05_origin'

def main():
    a=np.load(TABLE/'analysis_arrays.npz');meta=json.loads((TABLE/'methods_and_validation.json').read_text(encoding='utf-8'))
    labels=a['labels'].tolist();x=a['distance_nm'];f=a['frequency_cycles_per_nm'];E=a['bias_mV'];q=float(a['q3']);qb=float(a['qBragg'])
    op.set_show(False);op.new()
    realbook=op.new_book('w',lname='Real-space peak, half-depth gap and control data');realbook.name='RealSpace'
    freqbook=op.new_book('w',lname='Weighted spatial regression amplitude spectra');freqbook.name='Frequency'
    keys=[k[5:] for k in a.files if k.startswith('freq_')]
    real,freq={},{}
    for j,key in enumerate(keys):
        real[key]=sheet_csv(realbook,key,TABLE/f'{key}_real.csv',first=j==0)
        freq[key]=sheet_csv(freqbook,key,TABLE/f'{key}_frequency.csv',first=j==0)
        for sh in [real[key],freq[key]]:
            for k in range(1,12):sh.set_label(k,'pA' if key in ['pos_height','neg_height','g8_pA','outer_background_pA'] else 'pm' if key=='topo_pm' else '' if 'relative' in key else 'meV','U')
    fft=sheet_csv(freqbook,'W50_HannFFT',TABLE/'gap_width_HannFFT.csv')
    control=op.new_book('w',lname='Peak validity, q3 fits, calibration and sensitivity controls');control.name='Controls'
    for j,(name,file) in enumerate([('AllPixels','all_pixel_metrics.csv'),('PeakDetails','peak_measurements.csv'),('Q3Fits','q3_fits.csv'),('FixedShape','fixed_shape_control.csv'),('Bootstrap','conditional_block_bootstrap.csv'),('Covariates','q3_width_covariate_controls.csv'),('NegMask','neg_quad5_validity_mask_frequency.csv'),('PosMask','pos_quad5_validity_mask_frequency.csv')]):sheet_csv(control,name,TABLE/file,first=j==0)
    rawbook=op.new_book('w',lname='Unmodified measured LI Demod 1 X spectra');rawbook.name='RawSpectra'
    for j,label in enumerate(labels):
        sh=rawbook[0] if j==0 else rawbook.add_sheet(label);sh.name=label;sh.cols=100;sh.from_list(0,E.tolist(),lname='Bias',units='mV',axis='X')
        for k in range(99):sh.from_list(k+1,a['raw_signal_pA'][j,k].tolist(),lname=f'{label}_x{k:03d}',units='pA',axis='Y')
    info=op.new_book('w',lname='Sources, fixed extraction rules and limitations');info.name='SourceInfo';sh=info[0];sh.name='Methods';sh.cols=2
    sh.from_list(0,list(meta),lname='Item');sh.from_list(1,[json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else str(v) for v in meta.values()],lname='Value')
    guide=control.add_sheet('FrequencyGuides');guide.cols=3
    guide.from_list(0,[0,1],lname='Unit height',axis='Y')
    guide.from_list(1,[q,q],lname='1/(3a)',units='cycles/nm',axis='X');guide.from_list(2,[qb,qb],lname='1/a',units='cycles/nm',axis='X')
    colors=['#46327e','#3e689d','#21918c','#5cc863','#9dd93b'];graphs=[]
    def graph(sheet,values,indices,name,title,ytitle,frequency=False,scatter=False):
        gp=op.new_graph(template='Origin',lname=title);gp.name=name;assert gp.name==name;gl=gp[0];actual=[]
        for j,col in zip(indices,colors):
            if not np.isfinite(values[j]).any():continue
            p=gl.add_plot(sheet,coly=j+1,colx=0,type=201 if scatter else 'l');p.color=col
            if scatter:p.symbol_size=3
            if not scatter:p.layer.SetNumProp(p._format_property('line.width'),1.2)
            actual.append(j)
        assert actual
        gl.rescale();gl.axis('x').title='Spatial frequency (cycles/nm)' if frequency else 'Distance (nm)';gl.axis('y').title=ytitle
        gl.set_xlim(0,3.5,.5) if frequency else gl.set_xlim(0,float(x[-1]),.5)
        vv=np.concatenate([values[j][(f<=3.5) if frequency else np.ones(len(x),bool)] for j in actual]);vv=vv[np.isfinite(vv)]
        lo,hi=float(vv.min()),float(vv.max());span=max(hi-lo,.05)
        gl.set_ylim(0,hi*1.15 if hi>0 else 1) if frequency else gl.set_ylim(lo-.08*span,hi+.12*span)
        gl.lt_exec('layer.x.label.type=1;layer.x.label.numFormat=1;layer.x.label.decPlaces=2;layer.y.label.type=1;layer.y.label.numFormat=1;layer.y.label.decPlaces=2;')
        legend=gl.label('Legend')
        if legend:legend.remove()
        xmin,xmax=0,3.5 if frequency else float(x[-1]);ymax=gl.get_float('y.to');ymin=gl.get_float('y.from')
        for n,j in enumerate(actual):
            txt=gl.add_label(labels[j],xmax+.025*(xmax-xmin),ymax-(n+1)*.065*(ymax-ymin))
            if txt:txt.set_int('fsize',9);txt.set_int('color',op.ocolor(colors[indices.index(j)]))
        if frequency:
            # Independent worksheet per plot avoids changing already drawn guide heights.
            gs=control.add_sheet('Ref'+str(len(graphs)));gs.cols=3;gs.from_list(0,[0,ymax],lname='Amplitude',axis='Y');gs.from_list(1,[q,q],lname='1/(3a)',axis='X');gs.from_list(2,[qb,qb],lname='1/a',axis='X')
            for col,color in [(1,'#b08000'),(2,'#777777')]:
                p=gl.add_plot(gs,coly=0,colx=col,type='l');p.color=color;p.layer.SetNumProp(p._format_property('line.width'),.8)
            for ff,txt in [(q,'1/(3a)'),(qb,'1/a')]:
                lab=gl.add_label(txt,ff+.03,ymax*.94)
                if lab:lab.set_int('fsize',9)
        title_export(gp,DEST);graphs.append(dict(name=name,title=title,kind='frequency' if frequency else 'real',sheet=sheet.name,source_columns=[j+1 for j in actual],plots=len(actual)+(2 if frequency else 0)))
        print('Created',name,flush=True)
    se=list(range(6,11));dark=list(range(1,6))
    graph(real['neg_quad5'],a['neg_quad5'],se,'G_Se_Neg_Position','Se paths | negative peak positions | accepted 5-point fits','Peak energy (meV)',scatter=True)
    graph(freq['neg_quad5'],a['freq_neg_quad5'],se,'G_Se_Neg_Frequency','Se paths | negative-peak spatial amplitude spectrum','Peak-position amplitude (meV)',frequency=True)
    graph(real['neg_discrete'],a['neg_discrete'],se,'G_Se_Neg_Discrete','Se paths | negative peak | discrete measured maxima','Peak energy (meV)',scatter=True)
    graph(real['neg_quad3'],a['neg_quad3'],se,'G_Se_Neg_Quad3','Se paths | negative peak | 3-point quadratic vertices','Peak energy (meV)',scatter=True)
    graph(real['pos_quad5'],a['pos_quad5'],dark,'G_Dark_Pos_Position','Dark reference paths | sparse positive peaks | accepted only','Peak energy (meV)',scatter=True)
    graph(real['neg_quad5'],a['neg_quad5'],dark,'G_Dark_Neg_Position','Dark reference paths | sparse negative peaks | accepted only','Peak energy (meV)',scatter=True)
    for inds,tag,lab in [(dark,'Dark','Dark reference paths'),(se,'Se','Se paths')]:
        for key,short,ytitle in [('gap_width','W50','Half-depth full width W50 (meV)'),('gap_half','Delta50','Half-depth half width Delta50 (meV)'),('gap_center','Center','Half-depth center (meV)')]:
            graph(real[key],a[key],inds,f'G_{tag}_{short}_Real',lab+' | '+short+' along measured line',ytitle)
            graph(freq[key],a['freq_'+key],inds,f'G_{tag}_{short}_Freq',lab+' | '+short+' spatial amplitude spectrum',short+' amplitude (meV)',frequency=True)
        graph(fft,a['gap_width_HannFFT'],inds,f'G_{tag}_W50_FFT',lab+' | W50 | conventional Hann FFT check','W50 FFT amplitude (meV)',frequency=True)
        for key,short in [('edge_pos','EdgePlus'),('edge_neg','EdgeMinus')]:graph(real[key],a[key],inds,f'G_{tag}_{short}_Real',lab+' | '+short+' half-depth crossing','Crossing energy (meV)')
    graph(real['neg_height'],a['neg_height'],se,'G_Se_Neg_Height','Se paths | measured negative peak heights','LI X peak height (pA)',scatter=True)
    graph(real['neg_height_relative'],a['neg_height_relative'],se,'G_Se_Neg_RelHeight','Se paths | negative peak / reference background','Relative peak height',scatter=True)
    index=control.add_sheet('GraphIndex');index.cols=2;index.from_list(0,[r['name'] for r in graphs],lname='Graph');index.from_list(1,[r['title'] for r in graphs],lname='Description')
    project=DEST/'peak_positions_halfheight_gap_editable.opju';assert op.save(str(project));assert op.open(str(project),readonly=True,asksave=False)
    def numeric(seq):
        return np.array([np.nan if v is None or v=='' else float(v) for v in seq],float)
    errors={};masks={}
    for bookname,prefix in [('RealSpace',''),('Frequency','freq_')]:
        book=op.find_book('w',bookname)
        for key in keys:
            sh=next(sh for sh in book if sh.name==key)
            for j in range(11):
                yy=numeric(sh.to_list(j+1));expected=a[prefix+key][j]
                # Origin may omit trailing missing cells; restore to the known coordinate axis.
                if len(yy)<len(expected):yy=np.pad(yy,(0,len(expected)-len(yy)),constant_values=np.nan)
                assert len(yy)==len(expected)
                assert np.array_equal(np.isfinite(yy),np.isfinite(expected)),(bookname,key,j,'missing mask')
                good=np.isfinite(expected)
                if good.any():errors[bookname+'/'+key+'/'+str(j)]=float(np.max(abs(yy[good]-expected[good])))
            masks[bookname+'/'+key]=True
    assert max(errors.values())==0
    raw2=op.find_book('w','RawSpectra');rawerr=0.
    for j,label in enumerate(labels):
        sh=next(sh for sh in raw2 if sh.name==label)
        for k in range(99):rawerr=max(rawerr,float(np.max(abs(numeric(sh.to_list(k+1))-a['raw_signal_pA'][j,k]))))
    assert rawerr==0
    for rec in graphs:assert len(op.find_graph(rec['name'])[0].plot_list())==rec['plots']
    assert hashlib.sha256(Path(meta['source']).read_bytes()).hexdigest()==meta['source_sha256']
    result=dict(saved_project_reopened=True,graphs=len(graphs),native_graph_names=[r['name'] for r in graphs],numeric_max_error=0,raw_spectra_max_error=rawerr,missing_masks_preserved=True,raw_spectra_count=1089,source_hash_unchanged=True,independent_origin_instance=True)
    (DEST/'Origin_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(DEST/'graph_index.json').write_text(json.dumps(graphs,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    finally:
        try:op.exit()
        except Exception:pass
