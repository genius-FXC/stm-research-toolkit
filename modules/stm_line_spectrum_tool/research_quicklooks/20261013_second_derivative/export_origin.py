from pathlib import Path
import sys,json,shutil
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from origin_export import op,heatmap,title_export,sheet_csv
OUT=HERE/'results';DEST=OUT/'origin';DEST.mkdir(exist_ok=True)
a=np.load(OUT/'analysis_arrays.npz');meta=json.loads((OUT/'methods_and_validation.json').read_text(encoding='utf-8'))
palette=HERE.parent/'20261011_Se_rows070_076_082_088_094/results/Nanox.pal';shutil.copy2(palette,DEST/'Nanox.pal')
features=['in_neg','in_pos','out_neg','out_pos'];keys=['E_'+t for t in features]+['h_'+t for t in features]+['delta_in','delta_out']
def main():
    op.set_show(False);op.new();book=op.new_book('w',lname='Every dark pixel: energy, local height, separation and quality');book.name='PixelData'
    sheet_csv(book,'Metrics',OUT/'all_dark_pixel_metrics.csv',True)
    info=op.new_book('w',lname='Definitions and graph index');info.name='ReadMe';sh=info[0];sh.cols=2;sh.from_list(0,list(meta),lname='Item');sh.from_list(1,[json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else str(v) for v in meta.values()],lname='Definition')
    graphs=[]
    for key in keys:
        mat=op.new_sheet('m',lname=key);mat.name='M_'+key;mat.from_np(np.flipud(a[key]));mat.xymap=(0,98,0,29)
        name='G_'+('D_'+key[2:] if key.startswith('E_') else key)
        unit='pA' if key.startswith('h_') else 'meV' if key.startswith('delta') else 'mV'
        title=key+' | '+unit+' | SG5 accepted; missing = unidentified'
        gp=heatmap(mat,title,name,DEST,(0,98,20),(0,29,5),'Display x (pixel)','98 - display y (pixel)',DEST/'Nanox.pal',limits=meta['display_limits'][key]);gp.name=name
        gp[0].lt_exec(f'layer.height=30;layer.top=30;layer.cmap.colorMiss={op.ocolor("#d0d0d0")};layer.cmap.updateScale();')
        for labname in ['ZTitle','ColorScaleTitle']:
            lab=gp[0].label(labname)
            if lab:lab.text=unit
        title_export(gp,DEST);graphs.append({'graph':name,'metric':key,'matrix':mat.name,'unit':unit,'pixels':int(np.isfinite(a[key]).sum())});print(name,flush=True)
    idx=info.add_sheet('GraphIndex');idx.cols=3;idx.from_list(0,[r['graph'] for r in graphs],lname='Graph');idx.from_list(1,[r['metric'] for r in graphs],lname='Array key');idx.from_list(2,[r['unit'] for r in graphs],lname='Unit')
    path=DEST/'dark_second_derivative_maps.opju';assert op.save(str(path));assert op.open(str(path),readonly=True,asksave=False)
    err=0.;masks=True
    saved_m={layer.name:layer for page in op.pages() if type(page).__name__=='MBook' for layer in page}
    for rec in graphs:
        mat=saved_m[rec['matrix']];got=mat.to_np2d(0);expected=np.flipud(a[rec['metric']]);good=np.isfinite(expected)
        # Origin stores missing numeric cells as its native missing value.
        missing=~np.isfinite(got)|(abs(got)>1e100);assert np.array_equal(missing,~good),rec
        err=max(err,float(np.max(abs(got[good]-expected[good]))));assert op.find_graph(rec['graph'])
    assert err==0
    (DEST/'validation.json').write_text(json.dumps({'reopened':True,'graphs':10,'matrix_error':err,'missing_masks_preserved':True,'orientation':'matrix rows flipped, cartesian y=98-display_y'}),encoding='utf-8')
    (DEST/'graph_index.json').write_text(json.dumps(graphs,indent=2),encoding='utf-8');print(path)
if __name__=='__main__':
    try:main()
    finally:op.exit()
