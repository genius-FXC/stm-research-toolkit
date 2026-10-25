from pathlib import Path
import csv,json,sys,hashlib
import numpy as np
from scipy.signal import savgol_filter,savgol_coeffs,find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]));from plotting import NANOX
OUT=HERE/'results_v2';OUT.mkdir(exist_ok=True)
old=np.load(HERE/'results/analysis_arrays.npz');E=old['E'];g=old['g'];features=['in_neg','in_pos','out_neg','out_pos'];shape=g.shape[:2]
der={w:savgol_filter(g,w,3,deriv=2,delta=2,axis=-1) for w in [5,7,9]}
noise=old['sigma_proxy_pA'];maps={};stats={};reason={}
records=list(csv.DictReader((HERE/'results/candidate_details.csv').open(encoding='utf-8-sig')))
for feat in features:
    rr=[r for r in records if r['feature']==feat]
    reason[feat]={'no_candidate':2970-len(rr),'below_prominence':sum(float(r['prominence'])<float(r['threshold']) for r in rr),'competing':sum(r['competing']=='True' for r in rr),'boundary':sum(r['search_boundary']=='True' for r in rr),'old_accepted':int(np.isfinite(old['E_'+feat]).sum())}
def extract(D,split):
    ii,props=find_peaks(-D,prominence=0,wlen=9);res={}
    for feat in features:
        inner=feat.startswith('in');sign=-1 if feat.endswith('neg') else 1
        candidates=[(int(i),float(p)) for i,p in zip(ii,props['prominences']) if D[i]<0 and E[i]*sign>0 and 4<=abs(E[i])<=24 and ((abs(E[i])<=split) if inner else (abs(E[i])>split))]
        if not candidates:continue
        candidates.sort(key=lambda t:t[1],reverse=True);i,p=candidates[0];b=np.polyfit(E[i-1:i+2]-E[i],D[i-1:i+2],2);en=E[i]+np.clip(-b[1]/(2*b[0]),-1,1)
        res[feat]=(float(en),p,len(candidates)>1 and candidates[1][1]>=.7*p,abs(E[i]) in [4,24],float(np.polyval(b,en-E[i])))
    return res
for w in [5,7,9]:
    for feat in features:
        for key in ['E','h','prom','ambiguous','outer_boundary','D'] :maps[f'{key}{w}_{feat}']=np.full(shape,np.nan)
    for y in range(30):
        for x in range(99):
            for feat,(en,p,amb,edge,Dval) in extract(der[w][y,x],14).items():
                for key,value in [('E',en),('h',np.interp(en,E,g[y,x])),('prom',p),('ambiguous',int(amb)),('outer_boundary',int(edge)),('D',Dval)]:maps[f'{key}{w}_{feat}'][y,x]=value
# Split-window sensitivity, evaluated without reclassifying a missing branch as present.
for split in [12,16]:
    for feat in features:maps[f'E_split{split}_{feat}']=np.full(shape,np.nan)
    for y in range(30):
        for x in range(99):
            for feat,c in extract(der[7][y,x],split).items():maps[f'E_split{split}_{feat}'][y,x]=c[0]
for feat in features:
    e=maps['E7_'+feat];e5=maps['E5_'+feat];maps['E_'+feat]=e;maps['h_'+feat]=maps['h7_'+feat]
    p=maps['prom7_'+feat];local=np.max(np.maximum(-der[7][...,(abs(E)>=4)&(abs(E)<=24)],0),axis=-1)
    threshold=np.maximum.reduce([np.full(shape,.003),2*noise*np.linalg.norm(savgol_coeffs(7,3,deriv=2,delta=2)),.05*local])
    splitstable=np.isfinite(maps['E_split12_'+feat])&np.isfinite(maps['E_split16_'+feat])&(abs(e-maps['E_split12_'+feat])<=2)&(abs(e-maps['E_split16_'+feat])<=2)
    smoothstable=np.isfinite(e5)&(abs(e-e5)<=2)
    # Retain every actual minimum in candidate maps; quality is separate.
    confident=np.isfinite(e)&(p>=threshold)&(maps['ambiguous7_'+feat]==0)&(maps['outer_boundary7_'+feat]==0)&smoothstable&splitstable
    quality=np.where(np.isfinite(e),1,0);quality[confident]=2
    maps['quality_'+feat]=quality;maps['threshold_'+feat]=threshold;maps['SG7_minus_SG5_'+feat]=e-e5;maps['split_stable_'+feat]=splitstable
for side in ['neg','pos']:
    sep=abs(maps['E_in_'+side]-maps['E_out_'+side]);bad=np.isfinite(sep)&(sep<4)
    for feat in ['in_'+side,'out_'+side]:maps['quality_'+feat][bad]=1
for feat in features:
    stable=maps['quality_'+feat]==2;maps['stable_E_'+feat]=np.where(stable,maps['E_'+feat],np.nan);maps['stable_h_'+feat]=np.where(stable,maps['h_'+feat],np.nan)
    stats[feat]={'SG5_candidates':int(np.isfinite(maps['E5_'+feat]).sum()),'SG7_candidates':int(np.isfinite(maps['E_'+feat]).sum()),'SG9_candidates':int(np.isfinite(maps['E9_'+feat]).sum()),'stable':int(stable.sum())}
for part in ['in','out']:
    maps['delta_'+part]=maps['E_'+part+'_pos']-maps['E_'+part+'_neg'];maps['stable_delta_'+part]=maps['stable_E_'+part+'_pos']-maps['stable_E_'+part+'_neg']
    stats['delta_'+part]={'candidate':int(np.isfinite(maps['delta_'+part]).sum()),'stable':int(np.isfinite(maps['stable_delta_'+part]).sum())}
np.savez_compressed(OUT/'analysis_arrays.npz',E=E,g=g,D5=der[5],D7=der[7],D9=der[9],**maps)
with (OUT/'all_dark_pixel_metrics.csv').open('w',newline='',encoding='utf-8-sig') as h:
    wr=csv.writer(h);keys=list(maps);wr.writerow(['pixel_x','display_y',*keys])
    for y in range(30):
        for x in range(99):wr.writerow([x,y+69,*[float(maps[k][y,x]) if np.isfinite(maps[k][y,x]) else '' for k in keys]])
limits={};cmap=NANOX.copy();cmap.set_bad('#d0d0d0')
for mode in ['candidate','stable']:
    for kind in ['E','h','delta']:
        keys=[kind+'_'+t for t in (['in','out'] if kind=='delta' else features)]
        fig,axes=plt.subplots(2,1 if kind=='delta' else 2,figsize=(12,6),layout='constrained')
        for ax,key in zip(axes.flat,keys):
            dat=maps[('stable_' if mode=='stable' else '')+key];base=maps[key];v=base[np.isfinite(base)];lo,hi=np.percentile(v,[2,98]) if len(v) else [0,1];limits[key]=[float(lo),float(hi)]
            im=ax.imshow(dat,origin='upper',extent=(-.5,98.5,98.5,68.5),cmap=cmap,vmin=lo,vmax=max(hi,lo+.01),interpolation='nearest');ax.set(title=key+f' | {np.isfinite(dat).sum()}/2970',xlabel='Display x (pixel)',ylabel='Display y (pixel)');fig.colorbar(im,ax=ax,label='pA' if kind=='h' else 'meV' if kind=='delta' else 'mV',shrink=.8)
        fig.suptitle('SG7 '+mode+' | gray = no minimum' if mode=='candidate' else 'SG7 stable subset | gray = not retained by robustness checks')
        fig.savefig(OUT/f'{mode}_{kind}_maps.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(4,2,figsize=(12,13),layout='constrained')
for j,(y,x) in enumerate([(70,50),(79,20),(85,50),(94,50)]):
    s=g[y-69,x];ax,ad=axes[j];ax.plot(E,s,'.-',color='black',label='raw',ms=3)
    for w,col in [(5,'#4586bd'),(7,'#d66c16'),(9,'#2a9857')]:ax.plot(E,savgol_filter(s,w,3),color=col,label=f'SG{w}');ad.plot(E,der[w][y-69,x],color=col,label=f'SG{w}')
    for feat,col in zip(features,['#16a1bc','#e16d12','#8046a3','#df2d75']):
        en=maps['E_'+feat][y-69,x]
        if np.isfinite(en):ax.plot(en,maps['h_'+feat][y-69,x],'o',color=col);ad.axvline(en,color=col,ls=':',label=feat)
    central=s[abs(E)<=30];ax.set_ylim(float(central.min())-.5,float(central.max())+1)
    for t in (ax,ad):t.set_xlim(-30,30);t.set_xlabel('Bias (mV)');t.legend(fontsize=7)
    ax.set(title=f'x={x}, y={y}',ylabel='LI X (pA)');ad.set_ylabel('D (pA/mV^2)')
fig.savefig(OUT/'smoothing_and_extraction_examples.png',dpi=150)
meta=json.loads((HERE/'results/methods_and_validation.json').read_text(encoding='utf-8'));meta.update(version='v2',main='SG7 cubic derivative; span12mV',control='SG5 and SG9 cubic derivative; span8 and16mV',map_interpretation='ALL actual SG7 negative minima candidates, including weak/ambiguous/boundary; not accepted-gap maps',quality='0 no minimum,1 candidate only,2 passes prominence + ambiguity + SG5 agreement2mV + split12/16 agreement2mV + inner/outer separation4mV',old_rejection_reasons=reason,stats=stats,display_limits=limits)
meta['threshold']='SG7 propagated proxy; same numerical rule as v1';meta['inner_window']='4<=|V|<=14; boundary is flagged through split sensitivity, not deleted from candidate maps';meta['outer_window']='14<|V|<=24; |V|24 retained as candidate only';meta['candidate_choice']='largest prominence in window; all actual minima retained as candidates even if ambiguous or weak; stability filters separate'
assert hashlib.sha256(Path(meta['source']).read_bytes()).hexdigest()==meta['source_sha256']
for feat in features:
    for y,x in zip(*np.where(np.isfinite(maps['E_'+feat]))):assert maps['h_'+feat][y,x]==np.interp(maps['E_'+feat][y,x],E,g[y,x])
meta['validation']={'source_unchanged':True,'h_matches_raw_interpolation':True}
(OUT/'methods_and_validation.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'old_rejections_overlapping':reason,'v2':stats},indent=2))
# Reuse verified native heatmap export with explicit candidate titles and separate project.
code=(HERE/'export_origin.py').read_text(encoding='utf-8').replace("OUT=HERE/'results'","OUT=HERE/'results_v2'").replace('SG5 accepted; missing = unidentified','SG7 CANDIDATE; not all reliable').replace('dark_second_derivative_maps.opju','dark_second_derivative_SG7_candidates_v2.opju')
(HERE/'export_origin_v2.py').write_text(code,encoding='utf-8')
