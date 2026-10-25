from pathlib import Path
import json,sys,csv
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from origin_export import op,sheet_csv,title_export
OUT=HERE/'results_v3_lines';DEST=OUT/'origin';DEST.mkdir(exist_ok=True)
a=np.load(OUT/'line_analysis.npz');items=json.loads((OUT/'plot_index.json').read_text());colors=['#46327e','#3e689d','#21918c','#5cc863','#b59c14']
def main():
    op.set_show(False);op.new();real=op.new_book('w',lname='Real-space curves and available-row mean');real.name='RealSpace';freq=op.new_book('w',lname='Missing-aware frequency spectra and means');freq.name='Frequency';control=op.new_book('w',lname='Coverage, recovery source and method controls');control.name='Controls';graphs=[];sheets=[]
    for j,item in enumerate(items):
        group,key=item['group'],item['key'];indices=slice(0,5) if group=='Se' else slice(5,10)
        for domain,book in [('Line',real),('Freq',freq)]:
            stem=f'{group}_{key}_{domain}';file=OUT/(stem+'.csv');sh=sheet_csv(book,stem,file,first=j==0);sheets.append((book.name,sh.name,file))
            isfreq=domain=='Freq';vals=a['freq_'+key][indices] if isfreq else a['real_'+key][indices];eligible=int(np.isfinite(vals).any(axis=1).sum())
            title=f'{group} | {key} | '+('WLS spectrum' if isfreq else 'real space')
            if isfreq and eligible<5:title+=f' | only {eligible}/5 lines eligible'
            gp=op.new_graph(template='Origin',lname=title);gp.name='G_'+stem;gl=gp[0]
            for k in range(5):
                if np.isfinite(vals[k]).any():p=gl.add_plot(sh,coly=k+1,colx=0,type='l');p.color=colors[k]
            if eligible:
                p=gl.add_plot(sh,coly=6,colx=0,type='l');p.color='#000000';p.layer.SetNumProp(p._format_property('line.width'),2)
                if isfreq:
                    p=gl.add_plot(sh,coly=7,colx=0,type='l');p.color='#777777';p.layer.SetNumProp(p._format_property('line.width'),1.5)
            gl.rescale();gl.set_xlim(0,3.5 if isfreq else float(a['x'][-1]),.5)
            gl.axis('x').title='Spatial frequency (cycles/nm)' if isfreq else 'Distance (nm)';gl.axis('y').title=('Amplitude' if isfreq else key)+' ('+item['unit']+')'
            v=vals[:,a['f']<=3.5] if isfreq else vals;v=v[np.isfinite(v)]
            if len(v):
                lo,hi=float(v.min()),float(v.max());pad=max((hi-lo)*.1,.1);gl.set_ylim(0,hi*1.15) if isfreq else gl.set_ylim(lo-pad,hi+pad)
            else:gl.set_ylim(0,1)
            leg=gl.label('Legend')
            if leg:leg.remove()
            xmax=3.5 if isfreq else float(a['x'][-1]);ymin=gl.get_float('y.from');ymax=gl.get_float('y.to')
            for k,row in enumerate(item['rows']):
                lab=gl.add_label(f'y={row}, n={item["coverage"][k]}',xmax*1.01,ymax-(k+1)*.065*(ymax-ymin));lab.set_int('fsize',8);lab.set_int('color',op.ocolor(colors[k]))
            lab=gl.add_label('Black: mean' if not isfreq else f'Black: mean ({eligible})',xmax*1.01,ymax-.44*(ymax-ymin));lab.set_int('fsize',8)
            if isfreq:
                lab=gl.add_label('Gray: coherent',xmax*1.01,ymax-.50*(ymax-ymin));lab.set_int('fsize',8)
                if eligible:
                    guide=control.add_sheet('Ref'+str(j));guide.cols=4;guide.from_list(0,[0,ymax],axis='Y')
                    for col,ff,label,color in [(1,float(a['q3']),'q3','#bf3300'),(2,2*float(a['q3']),'2q3','#b58a00'),(3,float(a['qb']),'qB','#999999')]:
                        guide.from_list(col,[ff,ff],axis='X');p=gl.add_plot(guide,coly=0,colx=col,type='l');p.color=color;lab=gl.add_label(label,ff+.02,ymax*.95);lab.set_int('fsize',9)
                else:lab=gl.add_label('Insufficient valid samples\nNo spectrum or five-line mean',.3,.5);lab.set_int('fsize',12)
            title_export(gp,DEST);graphs.append({'name':gp.name,'title':title,'sheet':sh.name,'book':book.name,'unit':item['unit'],'eligible_lines':eligible});print(gp.name,flush=True)
    for j,(short,file) in enumerate([('Coverage','coverage.csv'),('Recovery','recovery_counts_before_pair_check.csv'),('Q3','q3_with_Bragg_controls.csv'),('Summary','summary.csv'),('Harmonics','q3_and_2q3_joint_fit.csv')]):sheet_csv(control,short,OUT/file,first=j==0)
    for feat in ['in_neg','in_pos','out_neg','out_pos']:
        sheet_csv(control,'Codes_'+feat,OUT/('source_codes_'+feat+'.csv'));sheet_csv(control,'Mask_'+feat,OUT/('recovery_mask_frequency_'+feat+'.csv'))
    for key in ['E_in_pos','h_in_pos']:
        file=OUT/f'Fe_{key}_TrueFFT.csv';sh=sheet_csv(freq,'Fe_'+key+'_TrueFFT',file);sheets.append((freq.name,sh.name,file));gp=op.new_graph(template='Origin',lname='Fe '+key+' | true FFT | all 5 lines complete');gp.name='G_Fe_'+key+'_FFT';gl=gp[0]
        for j in range(6):
            p=gl.add_plot(sh,coly=j+1,colx=0,type='l');p.color=colors[j] if j<5 else '#000000'
            if j==5:p.layer.SetNumProp(p._format_property('line.width'),2)
        gl.rescale();gl.set_xlim(0,3.5,.5);gl.axis('x').title='Spatial frequency (cycles/nm)';gl.axis('y').title='FFT amplitude ('+('pA' if key.startswith('h') else 'meV')+')'
        legend=gl.label('Legend')
        if legend:legend.remove()
        for j,label in enumerate(['y=73','y=79','y=85','y=91','y=97','Mean of 5']):
            lab=gl.add_label(label,3.55,gl.get_float('y.to')*(.92-j*.07));lab.set_int('fsize',8);lab.set_int('color',op.ocolor(colors[j] if j<5 else '#000000'))
        title_export(gp,DEST);graphs.append({'name':gp.name,'title':gp.lname,'sheet':sh.name,'book':freq.name,'eligible_lines':5});print(gp.name,flush=True)
    index=control.add_sheet('GraphIndex');index.cols=2;index.from_list(0,[r['name'] for r in graphs]);index.from_list(1,[r['title'] for r in graphs])
    path=DEST/'dark_Se_Fe_derivative_lines_frequency_v3.opju';assert op.save(str(path));assert op.open(str(path),readonly=True,asksave=False)
    maxerr=0
    for bn,sn,file in sheets:
        sh=next(t for t in op.find_book('w',bn) if t.name==sn)
        data=list(csv.reader(file.open(encoding='utf-8-sig')))[1:]
        for c in range(len(data[0])):
            want=np.array([float(r[c]) if r[c] else np.nan for r in data]);got=np.array([float(v) if v not in (None,'') else np.nan for v in sh.to_list(c)]);mask=np.isfinite(want);assert np.array_equal(np.isfinite(got),mask),(bn,sn,c)
            if mask.any():maxerr=max(maxerr,float(np.max(abs(got[mask]-want[mask]))))
    assert maxerr==0
    for rec in graphs:assert op.find_graph(rec['name'])
    (DEST/'validation.json').write_text(json.dumps({'saved_reopened':True,'graphs':len(graphs),'numeric_error':maxerr,'missing_masks_preserved':True,'data_sheets_validated':len(sheets)}),encoding='utf-8');(DEST/'graph_index.json').write_text(json.dumps(graphs,indent=2),encoding='utf-8');print(path)
if __name__=='__main__':
    try:main()
    finally:op.exit()
