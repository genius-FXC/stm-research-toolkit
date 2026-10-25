"""Independent comparisons against user-provided source functions and exports."""
from pathlib import Path
import ast,json,hashlib
import numpy as np
from core import ROOT,GridDataset,LineSettings,extract_line
from spectrum_processing import process_spectra
from plotting import save_numeric_bundle,export_images

REFERENCE=Path(__import__('os').environ['STM_REFERENCE_SCRIPT'])
def reference_functions():
    # Compile only the reviewed pure numerical functions. Do not import/run
    # os.listdir, DAT reading, plotting, or the reference's main program.
    tree=ast.parse(REFERENCE.read_text(encoding='utf-8'));names={'nm01','crop_data','gaussian_kernel','gaussian_smooth','adjacent_averaging'}
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    env={'np':np};exec(compile(ast.Module(body=selected,type_ignores=[]),str(REFERENCE),'exec'),env)
    return env

def oracle(x,y,curr,cfg,fn):
    xs=[];ys=[]
    for original in y:
        bx=x.copy();g=original.copy()
        if cfg.smoothing_method=='gaussian':bx,g=fn['gaussian_smooth'](bx,g,cfg.smoothing_order,cfg.smoothing_sigma)
        elif cfg.smoothing_method=='adjacent':bx,g=fn['adjacent_averaging'](bx,g,cfg.smoothing_order)
        bx=(bx+cfg.x_offset_mV)*cfg.x_factor
        bx,g=fn['crop_data'](bx,g,cfg.spectrum_min_mV,cfg.spectrum_max_mV)
        i=len(ys)
        if cfg.normalization_mode=='minmax':g=fn['nm01'](g)
        elif cfg.normalization_mode=='none':g=g*cfg.y_factor
        elif cfg.normalization_mode=='first':g=g/g[0]
        elif cfg.normalization_mode=='current':g=cfg.y_factor*g/(curr[i]/curr[cfg.current_reference_sample])
        xs.append(bx);ys.append(g)
    return xs[0],np.array(ys)

def main():
    fn=reference_functions();rng=np.random.default_rng(94231);errors=[];cases=0
    for descending in [False,True]:
        x=np.linspace(-35,35,51);x=x[::-1] if descending else x
        y=4+rng.random((7,len(x)))*5+.1*x[None,:];current=np.linspace(-50,-120,7)
        for smooth in ['none','adjacent','gaussian']:
            for mode in ['none','minmax','first','current']:
                cfg=LineSettings([0,0],[6,0],normalization_mode=mode,smoothing_method=smooth,smoothing_order=3,smoothing_sigma=1.35,x_offset_mV=.5,x_factor=1.2,y_factor=2.4,spectrum_min_mV=-25,spectrum_max_mV=23,current_reference_sample=2)
                before=y.copy();a=process_spectra(x,y,cfg,current,x[0]);xx,yy=oracle(x,y,current,cfg,fn)
                assert np.array_equal(a['bias_mV'],xx)
                e=float(np.max(abs(a['values']-yy)));assert e<1e-12;e2=float(np.max(abs(y-before)));assert e2==0
                errors.append(e);cases+=1
    # Reject singular normalizations without silently fabricating values.
    invalid=[('minmax',np.ones((2,51)),None),('first',np.zeros((2,51)),None),('current',np.ones((2,51)),None),('current',np.ones((2,51)),np.zeros(2))]
    for mode,y,curr in invalid:
        try:process_spectra(x,y,LineSettings([0,0],[1,0],normalization_mode=mode),curr)
        except ValueError:pass
        else:raise AssertionError('Singular normalization was accepted')
    path=Path(json.loads((ROOT/'config.json').read_text())['default_file']);before=hashlib.sha256(path.read_bytes()).hexdigest();d=GridDataset(path)
    cfg=LineSettings([0,85],[98,85],normalization_mode='minmax',smoothing_method='gaussian',smoothing_order=3,smoothing_sigma=1.0,spectrum_min_mV=-30,spectrum_max_mV=30,zero_anchor=False,curves=20)
    r=extract_line(d,cfg,'TEST');raw=r['signal_pA'].copy()
    x=d.bias[d.acquisition_bias_order];y=raw[:,d.acquisition_bias_order];cur=d.current_first_pA[r['xy'][:,1],r['xy'][:,0]]
    xx,yy=oracle(x,y,cur,cfg,fn);assert np.array_equal(xx,r['waterfall_bias_mV']);e=float(np.max(abs(yy-r['processed_spectra'])));assert e<1e-12
    assert np.array_equal(raw,d.g[r['xy'][:,1],r['xy'][:,0]])
    assert np.allclose(r['processed_spectra'].min(axis=1),0) and np.allclose(r['processed_spectra'].max(axis=1),1)
    folder=ROOT/'validation/processing_export_20261014';save_numeric_bundle(d,r,folder);export_images(d,r,folder,('png',))
    for filename,expected in [('TEST_processed_spectra.csv',r['processed_spectra']),('TEST_waterfall_display.csv',r['waterfall']),('TEST_full_spectra_used_pA.csv',raw)]:
        a=np.loadtxt(folder/filename,delimiter=',',skiprows=1,encoding='utf-8-sig');assert np.array_equal(a[:,1:].T,expected)
    arr=np.load(folder/'TEST_numeric.npz');assert np.array_equal(arr['waterfall_bias_mV'],xx)
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(reference_file=REFERENCE.name,reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),synthetic_combinations=cases,reference_max_error=max(errors),ascending_and_descending_sweeps=True,real_line_reference_max_error=e,original_raw_spectra_exact=True,current_channel=d.current_channel,current_first_bias_mV=d.current_first_bias_mV,raw_energy_points=len(d.bias),processed_energy_points=len(xx),all_four_reference_normalizations=True,sequential_adjacent_and_gaussian_matching=True,singular_denominators_rejected=True,csv_npz_export_matches_preview=True,source_unchanged=True)
    (ROOT/'validation/processing_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
