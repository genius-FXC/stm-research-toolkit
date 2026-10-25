from pathlib import Path
import sys,json,uuid,hashlib,ast,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'adapters'));import batch_runner as runner
import numpy as np
out=ROOT/'tests'/('run_'+uuid.uuid4().hex[:8]);out.mkdir();source=out/'synthetic.3ds'
n=64;k=5;w=12.8e-9;coords=(np.arange(n)+.5)*w/n-w/2;x,y=np.meshgrid(coords,coords)
z=(np.cos(2*np.pi*1.25*x*1e9)+.5*np.sin(2*np.pi*.625*y*1e9))*1e-12
records=np.zeros((n,n,5+2*k),dtype='>f4');records[:,:,0]=-.02;records[:,:,1]=.02;records[:,:,2]=x;records[:,:,3]=y;records[:,:,4]=z
records[:,:,5:10]=np.linspace(-.02,.02,k);records[:,:,10:]=z[:,:,None]*(1+np.arange(k))
fields={'Grid dim':f'{n} x {n}','Points':str(k),'# Parameters (4 byte)':'5','Channels':'Bias (V);LI Demod 1 X (A)','Fixed parameters':'Sweep Start;Sweep End','Experiment parameters':'X (m);Y (m);Z (m)','Filetype':'Linear','Sweep Signal':'Bias (V)','Experiment size (bytes)':str(k*2*4),'Grid settings':f'0;0;{w};{w};0','Start time':'15.10.2026 12:00:00','Bias>Bias (V)':'0.05'}
source.write_bytes(('\n'.join(f'{a}="{b}"' for a,b in fields.items())+'\n:HEADER_END:\n').encode()+records.tobytes());before=runner.sha(source)
r1=runner.run({'tool_id':'3ds-sxm','input':str(source),'output':str(out/'sxm')});sxms=list((out/'sxm/synthetic').glob('*.sxm'));assert len(sxms)==k+1
from _sxm_reader import read_sxm
assert np.array_equal(read_sxm(out/'sxm/synthetic/0000_Topo_Z.sxm').channels[0].data,z.astype(np.float32).astype(float))
r2=runner.run({'tool_id':'fft-stp','input':str(out/'sxm/synthetic'),'output':str(out/'fft')})
r3=runner.run({'tool_id':'batch-ifft','input':str(out/'sxm/synthetic/0000_Topo_Z.sxm'),'output':str(out/'ifft'),'centers':[[1.25,0]],'radius':.2,'formats':['stp','npz','csv']})
assert runner.sha(source)==before
recon=np.load(out/'ifft/0000_Topo_Z_iFFT.npz')['reconstruction'];assert np.all(np.isfinite(recon))
from bragg_ifft_trials import synthetic_check
check=synthetic_check();assert check['error']<1e-12
try:runner.prepare({'tool_id':'fft-stp','input':str(out/'sxm/synthetic'),'output':str(out/'fft')});raise AssertionError('overwrite allowed')
except ValueError:pass
from agent_bridge import SessionStore
s=SessionStore(out/'sessions');obj=s.create('kondo');obj['messages'].append({'role':'user','content':'test'});s.save(obj);assert s.load(out/'sessions'/(obj['id']+'.json'))==obj
report={'passed':True,'output':str(out),'3ds_maps':len(sxms),'fft_files':len(r2['files']),'ifft_outputs':r3['files'][0]['files'],'source_unchanged':True,'phase_check':check,'session_roundtrip':True,'overwrite_rejected':True}
(ROOT/'tests/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
