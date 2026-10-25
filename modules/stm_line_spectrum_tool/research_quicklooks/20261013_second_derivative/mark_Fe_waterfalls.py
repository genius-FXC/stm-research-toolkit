from pathlib import Path
import sys,json,csv
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from core import distance_colors
OUT=HERE/'results_v4_Fe_waterfall_markers';OUT.mkdir(exist_ok=True)
a=np.load(HERE/'results_v3_lines/line_analysis.npz');b=np.load(HERE/'results_v2/analysis_arrays.npz')
features=['in_neg','in_pos','out_neg','out_pos'];E=a['E'];offset=.464

def write(file,header,rows):
    with (OUT/file).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

def prepare():
    audit=[];coverage=[];graphs=[]
    for j in range(5,10):
        y=int(a['rows'][j]);raw=a['raw'][j];display=raw+offset*np.arange(99)[:,None];marks=[];ncurve=np.zeros(99,int)
        for x in range(99):
            for feat in features:
                e=float(a['real_E_'+feat][j,x]);stable=float(b['stable_E_'+feat][y-69,x]);good=np.isfinite(e) and np.isfinite(stable) and abs(e-stable)<1e-8
                h=float(np.interp(e,E,raw[x])) if np.isfinite(e) else np.nan
                if good:marks.append([e,h+offset*x,x,float(a['x'][x]),h,feat]);ncurve[x]+=1
                audit.append([y,x,float(a['x'][x]),feat,e,h,int(good),int(a['codes_'+feat][j,x]),int(b['quality_'+feat][y-69,x])])
        write(f'Fe_y{y}_Waterfall.csv',['bias_mV']+[f'y{y}_x{x:02d}_display_pA' for x in range(99)],np.column_stack([E,display.T]))
        write(f'Fe_y{y}_Raw.csv',['bias_mV']+[f'y{y}_x{x:02d}_raw_pA' for x in range(99)],np.column_stack([E,raw.T]))
        for lo,hi,suffix in [(0,98,'All'),(0,32,'P1'),(33,65,'P2'),(66,98,'P3')]:
            subset=[r for r in marks if lo<=r[2]<=hi];stem=f'Fe_y{y}_{suffix}'
            write(stem+'_Markers.csv',['energy_mV','display_pA','x_px','distance_nm','raw_h_pA','feature'],subset)
            graphs.append(dict(name='G_'+stem,title=f'Fe y={y} | x={lo}-{hi} | red: stable derivative features',y=y,j=j,lo=lo,hi=hi,markers=len(subset),stem=stem))
        coverage.append([y,*[sum(r[5]==feat for r in marks) for feat in features],len(marks),int((ncurve>0).sum()),int((ncurve==4).sum())])
    write('feature_audit.csv',['y_px','x_px','distance_nm','feature','candidate_energy_mV','raw_h_pA','red_marker','source_code_v3','quality_v2'],audit)
    write('coverage.csv',['y_px',*features,'red_dots','curves_with_red','curves_with_all_four'],coverage)
    (OUT/'graph_index.json').write_text(json.dumps(graphs,indent=2),encoding='utf-8')
    print(coverage)
    return graphs

def export():
    from origin_export import op,sheet_csv,title_export
    graphs=json.loads((OUT/'graph_index.json').read_text());dest=OUT/'origin';dest.mkdir(exist_ok=True)
    op.set_show(False);op.new();wb=op.new_book('w',lname='Waterfall raw spectra plus 0.464 pA per pixel');wb.name='Waterfalls';mb=op.new_book('w',lname='Reliable derivative feature red dots');mb.name='Markers';rb=op.new_book('w',lname='Original measured LI X pA');rb.name='RawSpectra';checks=[]
    try:
        sheets={}
        for k,y in enumerate([73,79,85,91,97]):
            sheets[y]=sheet_csv(wb,f'FeY{y}',OUT/f'Fe_y{y}_Waterfall.csv',first=k==0)
            rs=sheet_csv(rb,f'FeY{y}',OUT/f'Fe_y{y}_Raw.csv',first=k==0)
            checks.extend([(wb.name,sheets[y].name,OUT/f'Fe_y{y}_Waterfall.csv',100),(rb.name,rs.name,OUT/f'Fe_y{y}_Raw.csv',100)])
        for k,item in enumerate(graphs):
            y=item['y'];lo=item['lo'];hi=item['hi'];sh=sheets[y];mf=OUT/(item['stem']+'_Markers.csv');ms=sheet_csv(mb,item['stem'],mf,first=k==0);checks.append((mb.name,ms.name,mf,5))
            gp=op.new_graph(template='Origin',lname=item['title']);gp.name=item['name'];gl=gp[0]
            for x in range(lo,hi+1):
                p=gl.add_plot(sh,coly=x+1,colx=0,type='l');p.color=distance_colors([x/98])[0];p.layer.SetNumProp(p._format_property('line.width'),.8)
            p=gl.add_plot(ms,coly=1,colx=0,type='s');p.color='#ff0000';p.symbol_kind=2;p.symbol_size=2.5 if item['stem'].endswith('All') else 3.5
            gl.rescale();gl.set_xlim(-30,30,10)
            visible=a['raw'][item['j'],lo:hi+1][:,(E>=-30)&(E<=30)]+offset*np.arange(lo,hi+1)[:,None];low=float(visible.min());high=float(visible.max());pad=(high-low)*.04;gl.set_ylim(low-pad,high+pad)
            gl.axis('x').title='Bias (mV)';gl.axis('y').title='LI X (pA) + 0.464*x_pixel'
            legend=gl.label('Legend')
            if legend:legend.remove()
            title_export(gp,dest);print(gp.name,flush=True)
        ctrl=op.new_book('w',lname='Per-pixel feature quality and coordinates');ctrl.name='FeatureAudit';sheet_csv(ctrl,'Audit',OUT/'feature_audit.csv',first=True);sheet_csv(ctrl,'Coverage',OUT/'coverage.csv')
        project=dest/'FeFe_waterfalls_reliable_derivative_markers.opju';assert op.save(str(project));assert op.open(str(project),readonly=True,asksave=False)
        err=0
        for bn,sn,file,ncol in checks:
            s=next(t for t in op.find_book('w',bn) if t.name==sn);r=list(csv.reader(file.open(encoding='utf-8-sig')))[1:]
            for col in range(ncol):
                want=np.array([float(t[col]) for t in r]);got=np.array(s.to_list(col),float);assert want.shape==got.shape
                err=max(err,float(np.max(abs(want-got))))
        assert err<1e-10
        for item in graphs:
            gp=op.find_graph(item['name']);assert len(gp[0].plot_list())==item['hi']-item['lo']+2
        (dest/'validation.json').write_text(json.dumps(dict(saved_reopened=True,numeric_error=err,graphs=len(graphs),spectra=495,offset_pA=offset,reliable_filter='V3 energy equals V2 stable energy',all_curves_displayed=True),indent=2),encoding='utf-8')
    finally:op.exit()
if __name__=='__main__':
    if '--origin' in sys.argv:export()
    else:prepare()
