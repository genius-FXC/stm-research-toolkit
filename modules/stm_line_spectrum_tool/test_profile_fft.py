"""Analytic amplitude/frequency tests and real selected-channel integration."""
from pathlib import Path
import json,hashlib,numpy as np
from profile_fft import profile_fft
from core import GridDataset,LineSettings,extract_line,ROOT

def run():
    n=128;ds=.08;j=np.arange(n);s=ds*j;amplitude=3.4;k=11
    y=8000+amplitude*np.sin(2*np.pi*k*j/n+.2)
    plain=profile_fft(s,y,'none');hann=profile_fft(s,y,'hann')
    assert plain['peak']['index']==k and hann['peak']['index']==k
    assert abs(plain['peak']['frequency_cycles_per_nm']-k/(n*ds))<1e-13
    assert abs(plain['peak']['period_nm']-n*ds/k)<1e-13
    assert abs(plain['peak']['amplitude']-amplitude)<1e-10
    assert abs(hann['peak']['amplitude']-amplitude)<.001
    assert plain['sample_count']==n and len(plain['frequency_cycles_per_nm'])==n//2+1
    assert abs(plain['frequency_spacing_cycles_per_nm']-1/(n*ds))<1e-13
    assert plain['zero_padding']==0 and not plain['resampled']
    for window in ['none','hann']:
        f=profile_fft(s,y,window);reverse=profile_fft(s,y[::-1],window)
        assert np.allclose(f['amplitude'],reverse['amplitude'],rtol=1e-8,atol=1e-10)
        const=profile_fft(s,np.full(n,-12700.),window)
        assert np.max(const['amplitude'])==0 and const['peak'] is None
    even=profile_fft(s,2.6*(-1.)**j,'none')
    assert abs(even['amplitude'][-1]-2.6)<1e-12
    odd_n=129;odd_j=np.arange(odd_n);odd_k=64
    odd=profile_fft(ds*odd_j,2.6*np.sin(2*np.pi*odd_k*odd_j/odd_n),'none')
    assert abs(odd['amplitude'][-1]-2.6)<1e-11
    assert not profile_fft([0,.08],[2,3])['available']
    # Alternating digital-pixel step lengths: interpolation affects FFT only.
    nonuniform=np.r_[0,np.cumsum(np.tile([.07,.095],64))]
    profile=4*nonuniform+2;original=profile.copy()
    f=profile_fft(nonuniform,profile,'none')
    assert f['resampled'] and np.array_equal(profile,original)
    assert np.allclose(f['uniform_profile'],4*np.array(f['uniform_distance_nm'])+2)
    for bad in [([0,1,1,2],[1,2,3,4]),([0,1,2,3],[1,2,np.nan,4])]:
        try:profile_fft(*bad)
        except ValueError:pass
        else:raise AssertionError('Invalid profile accepted')

    if not __import__('os').environ.get('STM_DATA_FILE'):
        print('Synthetic profile FFT checks passed; dataset-specific checks skipped (STM_DATA_FILE unset).')
        return
    path=Path(__import__('os').environ['STM_DATA_FILE']);before=hashlib.sha256(path.read_bytes()).hexdigest()
    data=GridDataset(path)
    base=dict(start=[125,53],end=[57,65],normalize=False,energy_min_mV=-30,energy_max_mV=30,profile_mode='linear_fit')
    z=extract_line(data,LineSettings(**base,view='topo'))
    m=extract_line(data,LineSettings(**base,view='mapping',map_bias_mV=-15))
    changed=extract_line(data,LineSettings(**base,view='mapping',map_bias_mV=15,fft_window='none',curves=4))
    assert z['fft']['channel']=='Z' and z['fft']['unit']=='pm' and z['fft']['bias_mV'] is None
    assert m['fft']['channel']=='LI X' and m['fft']['unit']=='pA' and m['fft']['bias_mV']==-15
    assert not np.allclose(m['fft']['amplitude'],changed['fft']['amplitude'])
    assert m['fft']['resampled']
    for r in [z,m,changed]:
        again=profile_fft(r['s_nm'],r['profile_values'],r['config']['fft_window'])
        assert np.array_equal(r['fft']['amplitude'],again['amplitude'])
        assert np.array_equal(r['signal_pA'],data.g[r['xy'][:,1],r['xy'][:,0]])
    other=extract_line(data,LineSettings(**base,view='mapping',map_bias_mV=-15,curves=2,offset=999,heat_min_percentile=25,heat_max_percentile=70))
    assert np.array_equal(m['fft']['amplitude'],other['fft']['amplitude'])
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(known_sinusoid_frequency_and_period_exact=True,unwindowed_amplitude_error=abs(plain['peak']['amplitude']-amplitude),hann_gain_corrected_amplitude_error=abs(hann['peak']['amplitude']-amplitude),even_nyquist_not_doubled=True,odd_highest_bin_doubled=True,constant_profile_no_false_peak=True,short_line_unsupported_without_blocking_profile=True,reverse_amplitude_invariant=True,nonuniform_resampling_only_in_fft=True,selected_height_and_LI_X_channels_correct=True,current_bias_updates_FFT=True,waterfall_and_contrast_settings_do_not_affect_FFT=True,original_pixel_spectra_unchanged=True,source_unchanged=True,real_line_samples=m['n_samples'],real_line_fft_resampled=m['fft']['resampled'])
    (ROOT/'validation/profile_fft_numerical_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
