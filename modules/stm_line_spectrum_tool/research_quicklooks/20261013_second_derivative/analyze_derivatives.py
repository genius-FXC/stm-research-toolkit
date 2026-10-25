from pathlib import Path
import json,csv,sys,hashlib
import numpy as np
from scipy.signal import savgol_filter,savgol_coeffs,find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]));from plotting import NANOX
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
a=np.load(HERE/'source_arrays.npz');E=a['E'];g=a['g'][69:99];ny,nx,ne=g.shape;assert (ny,nx,ne)==(30,99,81);assert np.allclose(np.diff(E),2)
maskfile=Path(__import__('os').environ['STM_REGION_MASKS'])
mask=np.load(maskfile)['R1_dark'];assert mask[69:].all() and mask.sum()==2970
D5=savgol_filter(g,5,3,deriv=2,delta=2,axis=-1);D7=savgol_filter(g,7,3,deriv=2,delta=2,axis=-1)
diff=np.diff(g,n=2,axis=-1);sel=(abs(E[1:-1])>=40)&(abs(E[1:-1])<=68);v=diff[...,sel]
sigma=np.maximum(1.4826*np.median(abs(v-np.median(v,axis=-1)[...,None]),axis=-1)/np.sqrt(6),.001)
sigD=sigma*np.linalg.norm(savgol_coeffs(5,3,deriv=2,delta=2))
features=['in_neg','in_pos','out_neg','out_pos'];candidate_records=[]
def candidates(D):
    idx,props=find_peaks(-D,prominence=0,wlen=9)
    return [(int(i),float(p)) for i,p in zip(idx,props['prominences']) if 4<=abs(E[i])<=24 and D[i]<0]
def selected(D,feature):
    inner=feature.startswith('in');sgn=-1 if feature.endswith('neg') else 1
    cs=[(i,p) for i,p in candidates(D) if E[i]*sgn>0 and ((4<=abs(E[i])<=14) if inner else (14<abs(E[i])<=24))]
    if not cs:return None
    cs.sort(key=lambda c:c[1],reverse=True);i,p=cs[0]
    coeff=np.polyfit(E[i-1:i+2]-E[i],D[i-1:i+2],2)
    shift=-coeff[1]/(2*coeff[0]) if coeff[0]>0 else 0
    energy=float(E[i]+np.clip(shift,-1,1));curv=float(np.polyval(coeff,energy-E[i]))
    competing=len(cs)>1 and cs[1][1]>=.7*p
    # A search-edge extremum is retained for review but not accepted as a branch.
    boundary=abs(E[i]) in ([4,14] if inner else [24])
    return dict(energy=energy,discrete=float(E[i]),D=curv,prominence=p,competing=competing,boundary=boundary)
maps={};stats={}
for feat in features:
    res={k:np.full((ny,nx),np.nan) for k in ['E','h','D','E_discrete','E_candidate','h_candidate','E_SG7','shift_SG7','prominence','threshold','quality']}
    for y in range(ny):
        for x in range(nx):
            s=g[y,x];D=D5[y,x];c=selected(D,feat);b=selected(D7[y,x],feat)
            threshold=max(.003,2*sigD[y,x],.05*np.max(np.maximum(-D[(abs(E)>=4)&(abs(E)<=24)],0)))
            res['quality'][y,x]=0;res['threshold'][y,x]=threshold
            if b:res['E_SG7'][y,x]=b['energy']
            if c is None:continue
            ec=c['energy'];h=float(np.interp(ec,E,s));res['E_candidate'][y,x]=ec;res['h_candidate'][y,x]=h;res['prominence'][y,x]=c['prominence'];res['E_discrete'][y,x]=c['discrete'];res['D'][y,x]=c['D']
            shift=b['energy']-ec if b else np.nan;res['shift_SG7'][y,x]=shift
            qualified=c['prominence']>=threshold and not c['competing'] and not c['boundary']
            quality=1 if not qualified else 3 if np.isfinite(shift) and abs(shift)<=2 and not b['competing'] and not b['boundary'] else 2
            res['quality'][y,x]=quality
            if qualified:res['E'][y,x]=ec;res['h'][y,x]=h
            candidate_records.append([x,y+69,feat,ec,c['discrete'],h,c['D'],c['prominence'],threshold,c['competing'],c['boundary'],shift,quality])
    for k,v in res.items():maps[k+'_'+feat]=v
    stats[feat]={str(k):int((res['quality']==k).sum()) for k in range(4)}
for side in ['neg','pos']:
    sep=abs(maps['E_in_'+side]-maps['E_out_'+side]);bad=np.isfinite(sep)&(sep<4)
    for feat in ['in_'+side,'out_'+side]:
        maps['E_'+feat][bad]=np.nan;maps['h_'+feat][bad]=np.nan;maps['quality_'+feat][bad]=1
for part in ['in','out']:
    maps['delta_'+part]=maps['E_'+part+'_pos']-maps['E_'+part+'_neg']
    maps['delta_candidate_'+part]=maps['E_candidate_'+part+'_pos']-maps['E_candidate_'+part+'_neg']
    maps['stable_delta_'+part]=np.where((maps['quality_'+part+'_pos']==3)&(maps['quality_'+part+'_neg']==3),maps['delta_'+part],np.nan)
for feat in features:stats[feat]={'candidate':int(np.isfinite(maps['E_candidate_'+feat]).sum()),'accepted':int(np.isfinite(maps['E_'+feat]).sum()),'stable_SG7':int((maps['quality_'+feat]==3).sum()),'median_energy_mV':float(np.nanmedian(maps['E_'+feat]))}
for part in ['in','out']:stats['delta_'+part]={'accepted':int(np.isfinite(maps['delta_'+part]).sum()),'median_meV':float(np.nanmedian(maps['delta_'+part]))}
def csvwrite(name,header,data):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as h:
        w=csv.writer(h);w.writerow(header)
        for row in data:w.writerow(['' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for v in row])
csvwrite('candidate_details.csv',['pixel_x','display_y','feature','E_mV','E_discrete_mV','h_raw_pA','D_pA_per_mV2','prominence','threshold','competing','search_boundary','SG7_minus_SG5_mV','initial_quality'],candidate_records)
keys=list(maps)
csvwrite('all_dark_pixel_metrics.csv',['pixel_x','display_y',*keys],[[x,y+69,*[maps[k][y,x] for k in keys]] for y in range(ny) for x in range(nx)])
np.savez_compressed(OUT/'analysis_arrays.npz',E=E,g=g,D5=D5,D7=D7,sigma_proxy_pA=sigma,z_pm=a['z'][69:],**maps)
titles={'in_neg':'inner negative','in_pos':'inner positive','out_neg':'outer negative','out_pos':'outer positive'}
cmap=NANOX.copy();cmap.set_bad('#d0d0d0')
limits={}
def draw(ax,key,title,unit):
    arr=maps[key];valid=arr[np.isfinite(arr)];lo,hi=np.percentile(valid,[2,98]) if len(valid) else (0,1)
    if hi<=lo:hi=lo+1
    limits[key]=[float(lo),float(hi)];im=ax.imshow(arr,origin='upper',extent=(-.5,98.5,98.5,68.5),cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest',aspect='equal');ax.set_title(title+f' | {len(valid)}/2970',fontsize=10);ax.set(xlabel='Display x (pixel)',ylabel='Display y (pixel)');plt.colorbar(im,ax=ax,label=unit,shrink=.85)
for prefix,label,unit in [('E','Feature energy','mV'),('h','g at local feature','pA')]:
    fig,axes=plt.subplots(2,2,figsize=(13,6),layout='constrained')
    for ax,feat in zip(axes.flat,features):draw(ax,prefix+'_'+feat,label+': '+titles[feat],unit)
    fig.suptitle('Dark region | SG5 accepted candidates | gray = not identified reliably')
    fig.savefig(OUT/(prefix+'_maps.png'),dpi=180);fig.savefig(OUT/(prefix+'_maps.pdf'))
fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained')
for ax,part in zip(axes,['in','out']):draw(ax,'delta_'+part,'Full separation: '+part+' (E_pos - E_neg)','meV')
fig.savefig(OUT/'delta_maps.png',dpi=180);fig.savefig(OUT/'delta_maps.pdf')
fig,axes=plt.subplots(2,2,figsize=(13,6),layout='constrained')
for ax,feat in zip(axes.flat,features):
    im=ax.imshow(maps['quality_'+feat],origin='upper',extent=(-.5,98.5,98.5,68.5),vmin=0,vmax=3,cmap='viridis',interpolation='nearest');ax.set_title(feat);plt.colorbar(im,ax=ax,ticks=[0,1,2,3])
fig.suptitle('0 absent | 1 weak/ambiguous/boundary/unresolved | 2 accepted SG5 | 3 SG7 matched within 2 mV')
fig.savefig(OUT/'quality_maps.png',dpi=180)
fig,axes=plt.subplots(2,2,figsize=(13,6),layout='constrained')
for ax,feat in zip(axes.flat,features):draw(ax,'E_candidate_'+feat,'UNFILTERED candidate: '+feat,'mV')
fig.suptitle('Exploratory candidates only: may include noise and branch ambiguity')
fig.savefig(OUT/'candidate_energy_maps.png',dpi=180)
# Representative pixel spectra and derivative with all retained features.
scores=np.isfinite(maps['delta_out']).astype(int)+sum((maps['quality_'+t]==3).astype(int) for t in features)
coords=[tuple(np.unravel_index(np.argmax(scores),scores.shape)),(1,50),(16,50),(25,50)]
fig,axes=plt.subplots(4,2,figsize=(12,13),layout='constrained')
for j,(y,x) in enumerate(coords):
    left,right=axes[j];left.plot(E,g[y,x],'.-',ms=3,color='black');right.plot(E,D5[y,x],'.-',ms=3,label='SG5');right.plot(E,D7[y,x],label='SG7',alpha=.7)
    for feat,col in zip(features,['#168aad','#ff8a00','#542788','#c51b7d']):
        en=maps['E_'+feat][y,x]
        if np.isfinite(en):
            left.plot(en,maps['h_'+feat][y,x],'o',color=col);left.axvline(en,color=col,alpha=.4);right.axvline(en,color=col,label=feat)
    for ax in (left,right):ax.set_xlim(-30,30);ax.set_xlabel('Bias (mV)')
    central=g[y,x,(E>=-30)&(E<=30)];left.set_ylim(float(central.min())-.5,float(central.max())+1)
    left.set(title=f'x={x}, display y={y+69}',ylabel='Raw LI X (pA)');right.set_ylabel('D = d2g/dV2 (pA/mV^2)');right.legend(fontsize=7)
fig.savefig(OUT/'extraction_examples.png',dpi=150)
# Algebraic derivative and height-sampling verification.
assert np.allclose(savgol_filter(E**2,5,3,deriv=2,delta=2)[2:-2],2)
for feat in features:
    for y,x in zip(*np.where(np.isfinite(maps['E_'+feat]))):assert maps['h_'+feat][y,x]==np.interp(maps['E_'+feat][y,x],E,g[y,x])
source=Path(__import__('os').environ['STM_DATA_FILE'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==str(a['sha256'])
metadata=dict(source=str(source),source_sha256=str(a['sha256']),region='R1_dark display y=69..98, x=0..98',pixels=2970,derivative='second derivative of LI X vs bias, not second derivative of current',main='Savitzky-Golay window 5, order 3, deriv 2, delta 2 mV; window span 8 mV',control='SG7 order3, span12mV',inner_window='4<=abs(V)<=14 mV; reject extrema at discrete |V|=4 or14 as boundary',outer_window='14<abs(V)<=24 mV; reject extrema at |V|=24',extremum='negative local minimum of D; equivalent to peak in -D',height='linear interpolation of each ORIGINAL pixel spectrum at its own subpixel-energy D extremum',delta='full positive-negative separation, NOT half gap, NOT W50',quality='0 no minimum; 1 insufficient/ambiguous/boundary/unresolved; 2 accepted SG5; 3 also within2mV of SG7 nonboundary unambiguous candidate (SG7 significance not required)',threshold='max(0.003 pA/mV2,2*propagated noise proxy,5% local largest negative curvature); heuristic not calibrated significance',candidate_choice='largest prominence in each fixed window; secondary >=70% prominence marks ambiguity; inner-outer separation <4mV rejects both',vertex='3 derivative samples quadratic vertex, shift clipped to +/-1mV; does not improve experimental resolution',noise='MAD second difference over40..68mV divided sqrt6, floor .001pA; proxy may contain physical curvature; propagated by SG coefficient norm',stats=stats,display_limits=limits,validation={'source_unchanged':True,'quadratic_derivative_test':True,'h_matches_raw_interpolation':True})
(OUT/'methods_and_validation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(stats,indent=2))
