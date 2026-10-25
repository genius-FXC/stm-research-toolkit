"""Scientific checks for direct, discrete measured-pixel sampling."""
import json,hashlib
from pathlib import Path
import numpy as np
from core import GridDataset,LineSettings,extract_line,ROOT,pixel_path

def run():
    d=GridDataset.__new__(GridDataset);d.nx=9;d.ny=7;d.bias=np.array([-80.,-40.,0.,40.,80.]);yy,xx=np.indices((7,9))
    d.g=3+xx[:,:,None]*.7+yy[:,:,None]*1.3+np.arange(5)[None,None,:]*2
    d.z=5000+.8*xx+2*yy;d.topo=.1*xx-.3*yy
    theta=.3;d.transform=np.array([[12,20],[.1*np.cos(theta),.1*np.sin(theta)],[.2*np.sin(theta),-.2*np.cos(theta)]])
    cfg=LineSettings.from_dict(dict(start=[1.2,1.4],end=[7.1,5.3],step_px=.05,width_px=2,curves=4))
    r=extract_line(d,cfg);xy=r['xy']
    assert r['config']['start']==[1,1] and r['config']['end']==[7,5]
    assert r['config']['step_px']==1 and r['config']['width_px']==0
    assert np.array_equal(r['signal_pA'],d.g[xy[:,1],xy[:,0]])
    assert np.array_equal(r['profile_raw'],d.z[xy[:,1],xy[:,0]])
    assert np.array_equal(r['profile_raw'],r['profile_values'])
    assert abs(r['length_nm']-np.hypot(6*.1,4*.2))<1e-12
    rev=extract_line(d,LineSettings(cfg.end,cfg.start,curves=4))
    assert np.array_equal(rev['xy'],xy[::-1]) and np.array_equal(rev['signal_pA'],r['signal_pA'][::-1])
    assert np.allclose(rev['s_nm'],r['length_nm']-r['s_nm'][::-1])
    for x in range(9):
        for y in range(7):
            for b in ([0,0],[8,0],[0,6],[8,6],[4,3]):
                path=pixel_path([x,y],b)
                assert np.array_equal(path,pixel_path(b,[x,y])[::-1])
                assert len(path)==max(abs(x-b[0]),abs(y-b[1]))+1
                assert len(np.unique(path,axis=0))==len(path)
                if len(path)>1:assert np.all(np.max(abs(np.diff(path,axis=0)),axis=1)==1)
    for mode in ('subtract_average','linear_fit'):
        q=extract_line(d,LineSettings([1,2],[7,2],profile_mode=mode))
        assert np.allclose(q['signal_pA'],d.g[q['xy'][:,1],q['xy'][:,0]])
        assert abs(q['profile_values'].mean())<1e-9
        if mode=='linear_fit':assert np.max(abs(q['profile_values']))<1e-9
    mapping=extract_line(d,LineSettings([1,2],[7,2],view='mapping',map_bias_mV=39,profile_mode='raw'))
    assert mapping['profile_bias_mV']==40 and mapping['profile_unit']=='pA'
    assert np.array_equal(mapping['profile_values'],mapping['signal_pA'][:,3])
    a=extract_line(d,LineSettings([1,2],[7,2],curves=2000,heat_min_percentile=20,heat_max_percentile=70))
    b=extract_line(d,LineSettings([1,2],[7,2],curves=2))
    assert np.array_equal(a['signal_pA'],b['signal_pA']) and len(a['waterfall'])==a['n_samples']
    assert b['waterfall_indices'].tolist()==[0,b['n_samples']-1]
    assert np.allclose(a['heat_limits_pA'],np.percentile(a['signal_pA'][:,a['energy_mask']],[20,70]))
    for bad in (dict(start=[-1,0],end=[5,5]),dict(start=[1,1],end=[1.4,1.4]),dict(start=[1,1],end=[5,5],heat_min_percentile=90,heat_max_percentile=10)):
        try:extract_line(d,LineSettings.from_dict(bad))
        except ValueError:pass
        else:raise AssertionError('Invalid parameters accepted')
    import os
    if not os.environ.get('STM_DATA_FILE'):
        print('Synthetic sampling checks passed; dataset-specific checks skipped (STM_DATA_FILE unset).')
        return
    path=Path(os.environ['STM_DATA_FILE']);before=hashlib.sha256(path.read_bytes()).hexdigest();real=GridDataset(path)
    rr=extract_line(real,LineSettings([12.3,17.4],[85.6,73.2],view='mapping',map_bias_mV=-10,profile_mode='linear_fit'))
    from nanonis_reader import read_grid
    raw=read_grid(path);order=np.argsort(raw['bias']);errors=[];z_errors=[]
    for i,(x,y) in enumerate(rr['xy']):
        truth=raw['lix'][real.ny-1-y,x,order].astype(float)*1e12
        errors.append(float(np.max(abs(truth-rr['signal_pA'][i]))))
        z_errors.append(abs(float(raw['z'][real.ny-1-y,x])*1e12-rr['z_raw_center_pm'][i]))
    assert max(errors)==0 and max(z_errors)==0
    assert np.max(abs(np.column_stack([np.ones(rr['n_samples']),rr['s_nm']]).T@rr['profile_values']))<1e-8
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(all_octants_reverse_exact=True,integer_endpoints_and_legacy_migration=True,unique_8_connected_pixels=True,all_measured_pixel_spectra_exact=True,profile_raw_average_linear_fit=True,mapping_profile_selected_bias_exact=True,contrast_does_not_change_signal=True,waterfall_display_subset_only=True,rotated_lab_projected_distance_exact=True,outside_same_pixel_bad_contrast_rejected=True,real_grid_shape=list(real.g.shape),real_pixels_checked=rr['n_samples'],real_pixel_spectrum_max_error_pA=max(errors),real_Z_max_error_pm=max(z_errors),source_unchanged=True)
    (ROOT/'validation').mkdir(exist_ok=True);(ROOT/'validation/discrete_numerical_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
