"""Real-grid and malformed-header checks for optional spectroscopy settings."""
import hashlib,json,re,tempfile
from pathlib import Path
import numpy as np
from nanonis_reader import read_grid
from core import GridDataset,LineSettings,extract_line,ROOT

def expect_rejection(path,reason):
    try:read_grid(path)
    except ValueError as ex:
        assert reason in str(ex),(reason,str(ex))
        return str(ex)
    raise AssertionError('Malformed sweep accepted')

def run():
    target=Path(__import__('os').environ['STM_DATA_FILE'])
    original=target.read_bytes();before=hashlib.sha256(original).hexdigest()
    grid=read_grid(target);data=GridDataset(target)
    assert (grid['nx'],grid['ny'],grid['npts'])==(256,128,13)
    assert grid['bias_endpoint_sources']==['record fixed parameter Sweep Start','record fixed parameter Sweep End']
    assert np.array_equal(data.bias,np.arange(-30,31,5))
    stop=original.index(b':HEADER_END:')+len(b':HEADER_END:')
    offset=stop
    for term in [b'\r\r\n',b'\r\n',b'\n']:
        if original[offset:offset+len(term)]==term:offset+=len(term);break
    # Independently address the second recorded channel in the binary payload.
    records=np.frombuffer(original,dtype='>f4',offset=offset).reshape(128,256,52)
    expected_lix=records[::-1,:,26:39][:,:,::-1].astype(float)*1e12
    expected_z=records[::-1,:,4].astype(float)*1e12
    assert np.array_equal(data.g,expected_lix) and np.array_equal(data.z,expected_z)
    settings=LineSettings([20,80],[220,80],norm_lo_mV=22.5,norm_hi_mV=30,energy_min_mV=-30,energy_max_mV=30)
    line=extract_line(data,settings)
    assert line['signal_pA'].shape==(201,13)
    assert np.array_equal(line['signal_pA'],expected_lix[80,20:221])
    for mode in ['raw','subtract_average','linear_fit']:
        r=extract_line(data,LineSettings([20,80],[220,80],normalize=False,energy_min_mV=-30,energy_max_mV=30,profile_mode=mode))
        assert np.array_equal(r['profile_raw'],expected_z[80,20:221])
    r=extract_line(data,LineSettings([20,80],[220,80],normalize=False,energy_min_mV=-30,energy_max_mV=30,view='mapping',map_bias_mV=10))
    assert np.array_equal(r['profile_raw'],expected_lix[80,20:221,8])
    failures={}
    with tempfile.TemporaryDirectory(prefix='reader_compat_',dir=ROOT/'validation') as tmp:
        tmp=Path(tmp)
        partial=tmp/'one_header_endpoint.3ds'
        partial.write_bytes(original[:stop-len(b':HEADER_END:')]+b'Bias Spectroscopy>Sweep Start (V)=0.03\r\r\n'+original[stop-len(b':HEADER_END:'):])
        assert np.array_equal(read_grid(partial)['bias'],grid['bias'])
        bad=tmp/'mismatched_header.3ds'
        bad.write_bytes(original[:stop-len(b':HEADER_END:')]+b'Bias Spectroscopy>Sweep Start (V)=0.04\r\r\n'+original[stop-len(b':HEADER_END:'):])
        failures['header_mismatch']=expect_rejection(bad,'Binary and header')
        bad=tmp/'varying_endpoints.3ds';modified=bytearray(original)
        modified[offset+52*4+4:offset+52*4+8]=np.array([-.025],dtype='>f4').tobytes()
        bad.write_bytes(modified);failures['varying_endpoints']=expect_rejection(bad,'endpoints vary')
        bad=tmp/'MLS.3ds';bad.write_bytes(original.replace(b'Filetype=Linear',b'Filetype=MLS',1))
        failures['MLS']=expect_rejection(bad,'linear bias sweep')
        bad=tmp/'truncated.3ds';bad.write_bytes(original[:-4])
        failures['truncated']=expect_rejection(bad,'payload')
    old_path=Path(json.loads((ROOT/'config.json').read_text(encoding='utf-8'))['default_file'])
    old=read_grid(old_path)
    assert np.allclose(old['bias'],np.linspace(.08,-.08,81)) or np.allclose(old['bias'],np.linspace(-.08,.08,81))
    assert all(s.startswith('Bias Spectroscopy>') for s in old['bias_endpoint_sources'])
    assert before==hashlib.sha256(target.read_bytes()).hexdigest()
    report={'source_file':str(target),'source_unchanged':True,'source_sha256':before,'grid_shape':[128,256,13],'bias_mV':data.bias.tolist(),'bias_endpoint_sources':grid['bias_endpoint_sources'],'all_pixels_all_13_LI_X_bias_values_exact':True,'all_Z_pixels_exact':True,'LI_X_max_error_pA':0,'Z_max_error_pm':0,'line_preview_pixels':201,'default_adapted_normalization_works':True,'three_profile_modes_and_mapping_profile_checked':True,'one_missing_header_endpoint_supported':True,'malformed_sweeps_rejected':failures,'original_99x99_81bias_grid_regression_passed':True}
    (ROOT/'validation/reader_compatibility_pattern0929003.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
