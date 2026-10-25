from pathlib import Path
import sys,json,csv,hashlib,warnings
import numpy as np
from scipy.signal import savgol_filter,savgol_coeffs,find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
OUT=HERE/'results_v3_lines';OUT.mkdir(exist_ok=True)
a=np.load(HERE/'results_v2/analysis_arrays.npz');noise=np.load(HERE/'results/analysis_arrays.npz')['sigma_proxy_pA']
ref=np.load(HERE.parent/'20261011_Se_rows070_076_082_088_094/results/Se_rows_frequency_energy.npz')
x=ref['distance_nm'];f=ref['frequency_cycles_per_nm'][1:];q3=float(ref['q_third_cycles_per_nm']);qb=float(ref['q_bragg_cycles_per_nm']);E=a['E']
rows=np.array([70,76,82,88,94,73,79,85,91,97]);g=a['g'][rows-69];features=['in_neg','in_pos','out_neg','out_pos']
labels=[('Se' if j<5 else 'Fe')+f'_y{r}' for j,r in enumerate(rows)]
d74=savgol_filter(g,7,4,deriv=2,delta=2,axis=-1);d3=savgol_filter(g,3,2,deriv=2,delta=2,axis=-1)
def pick(D,feat):
    ii,p=find_peaks(-D,prominence=0,wlen=9);sign=-1 if feat.endswith('neg') else 1;inner=feat.startswith('in')
    cs=[(int(i),float(pr)) for i,pr in zip(ii,p['prominences']) if D[i]<0 and E[i]*sign>0 and 4<=abs(E[i])<=24 and ((abs(E[i])<=14) if inner else abs(E[i])>14)]
    if not cs:return None
    cs.sort(key=lambda t:t[1],reverse=True);i,pr=cs[0];b=np.polyfit(E[i-1:i+2]-E[i],D[i-1:i+2],2);en=float(E[i]+np.clip(-b[1]/(2*b[0]),-1,1))
    return en,pr,len(cs)>1 and cs[1][1]>=.7*pr,abs(E[i]) in [4,14,24]
metrics={};base={};uniform5={};codes={};recovery=[]
for feat in features:
    en=a['E_'+feat][rows-69].copy();orig=en.copy();e5=a['E5_'+feat][rows-69].copy();code=np.where(np.isfinite(en),1,0)
    missing=~np.isfinite(en)&np.isfinite(e5);en[missing]=e5[missing];code[missing]=2
    for j,k in zip(*np.where(~np.isfinite(en))):
        c=pick(d74[j,k],feat);r=pick(d3[j,k],feat)
        if c is None or r is None:continue
        sigma=float(noise[rows[j]-69,k])*np.linalg.norm(savgol_coeffs(7,4,deriv=2,delta=2))
        if c[1]>=max(.003,3*sigma) and abs(c[0]-r[0])<=2 and not c[2] and not c[3]:en[j,k]=c[0];code[j,k]=3
    metrics['E_'+feat]=en;base['E_'+feat]=orig;uniform5['E_'+feat]=e5;codes[feat]=code
    for j in range(10):recovery.append([labels[j],feat,int(np.isfinite(orig[j]).sum()),int((code[j]==2).sum()),int((code[j]==3).sum()),int(np.isfinite(en[j]).sum())])
# A fallback must not erase an existing v2 feature. Reject only the added
# member if it cannot be resolved from the retained SG7 feature.
pair_flags={}
for side in ['neg','pos']:
    inn,out=metrics['E_in_'+side],metrics['E_out_'+side];bad=np.isfinite(inn)&np.isfinite(out)&(abs(inn-out)<4)
    pair_flags[side]=bad.copy()
    ci,co=codes['in_'+side],codes['out_'+side]
    dropin=bad&(ci>1)&(co==1);dropout=bad&(co>1)
    inn[dropin]=np.nan;ci[dropin]=0;out[dropout]=np.nan;co[dropout]=0
for data in [metrics,base,uniform5]:
    for feat in features:
        en=data['E_'+feat];h=np.full_like(en,np.nan)
        for j,k in zip(*np.where(np.isfinite(en))):h[j,k]=np.interp(en[j,k],E,g[j,k])
        data['h_'+feat]=h
    for part in ['in','out']:data['delta_'+part]=data['E_'+part+'_pos']-data['E_'+part+'_neg']
keys=['E_'+t for t in features]+['h_'+t for t in features]+['delta_in','delta_out']
def csvsave(file,header,data):
    with (OUT/file).open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.writer(h);w.writerow(header)
        for rr in data:w.writerow(['' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for v in rr])
csvsave('recovery_counts_before_pair_check.csv',['line','feature','SG7','SG5_added','SG7p4_added','total'],recovery)
def wls(y,freq,bragg=False,minn=40):
    ok=np.isfinite(y);idx=np.flatnonzero(ok)
    if len(idx)<minn or np.ptp(x[ok])<.8*np.ptp(x):return np.nan+1j*np.nan
    xx=x[ok];yy=y[ok];X=[np.ones(len(xx)),xx-xx.mean(),np.cos(2*np.pi*freq*xx),np.sin(2*np.pi*freq*xx)]
    if bragg:
        for fq in [qb,2*qb]:X += [np.cos(2*np.pi*fq*xx),np.sin(2*np.pi*fq*xx)]
    X=np.array(X).T;weights=np.sqrt(np.hanning(99)[ok]);Xw=X*weights[:,None]
    if np.linalg.cond(Xw)>1e6:return np.nan+1j*np.nan
    b=np.linalg.lstsq(Xw,yy*weights,rcond=None)[0];return b[2]-1j*b[3]
def spectrum(y):return np.array([wls(y,ff) for ff in f])
def mean(arr,axis=0):
    with warnings.catch_warnings():warnings.simplefilter('ignore');return np.nanmean(arr,axis=axis)
spectra={};basefreq={};freq5={};fft={};summary=[];coverage=[];qtable=[];plotinfo=[];arrout={}
colors=plt.cm.viridis(np.linspace(.1,.85,5));window=np.hanning(99)
for key in keys:
    spectra[key]=np.array([spectrum(y) for y in metrics[key]]);basefreq[key]=np.array([spectrum(y) for y in base[key]]);freq5[key]=np.array([spectrum(y) for y in uniform5[key]])
    # Genuine FFT available only when every measured pixel has a feature.
    fft[key]=np.full((10,len(f)),np.nan)
    for j,y in enumerate(metrics[key]):
        if np.isfinite(y).all():
            b=np.polyfit(x,y,1);v=y-np.polyval(b,x);v-=v.mean();fft[key][j]=2*abs(np.fft.rfft(v*window)[1:])/window.sum()
        coverage.append([labels[j],key,int(np.isfinite(y).sum()),int(np.isfinite(base[key][j]).sum()),int(np.isfinite(uniform5[key][j]).sum()),'FFT+WLS' if np.isfinite(y).all() else 'WLS' if np.isfinite(spectra[key][j]).any() else 'insufficient'])
        for method,data in [('Recovered',metrics),('SG7',base),('SG5',uniform5)]:
            cc=wls(data[key][j],q3,True);qtable.append([labels[j],key,method,int(np.isfinite(data[key][j]).sum()),abs(cc),np.angle(cc),cc.real,cc.imag])
    arrout['real_'+key]=metrics[key];arrout['SG7_'+key]=base[key];arrout['SG5_'+key]=uniform5[key];arrout['freq_'+key]=abs(spectra[key]);arrout['complex_'+key]=spectra[key];arrout['fft_'+key]=fft[key]
    for group,inds in [('Se',slice(0,5)),('Fe',slice(5,10))]:
        vals=metrics[key][inds];sp=abs(spectra[key][inds]);m=mean(vals);n=np.isfinite(vals).sum(axis=0);av=mean(sp);coh=abs(mean(spectra[key][inds]));bav=mean(abs(basefreq[key][inds]));a5=mean(abs(freq5[key][inds]))
        nn=np.isfinite(sp).sum(axis=0)
        csvsave(f'{group}_{key}_Line.csv',['distance_nm',*labels[inds],'MeanAvailable','N_rows'],np.column_stack([x,vals.T,m,n]))
        csvsave(f'{group}_{key}_Freq.csv',['frequency_cycles_per_nm',*labels[inds],'MeanAmplitude','CoherentAmplitude','SG7_mean','SG5_mean','N_rows'],np.column_stack([f,sp.T,av,coh,bav,a5,nn]))
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');unit='pA' if key.startswith('h') else 'meV';a0,a1,a2,a3=axes.flat
        for j,label in enumerate(labels[inds]):
            a0.plot(x,vals[j],color=colors[j],lw=.8,label=f'{label} ({np.isfinite(vals[j]).sum()}/99)');a1.plot(f,sp[j],color=colors[j],lw=.8)
        a0.plot(x,m,color='black',lw=1.8,label='Mean of available rows');a0.set(xlabel='Distance (nm)',ylabel=key+' ('+unit+')');a0.legend(fontsize=7)
        a1.plot(f,av,color='black',lw=1.8,label='Mean amplitude');a1.plot(f,coh,color='black',ls='--',label='Magnitude of mean coefficient');a1.legend(fontsize=7)
        a2.plot(f,av,label='Recovered');a2.plot(f,bav,label='SG7 only');a2.plot(f,a5,label='SG5 only');a2.legend(fontsize=8)
        for ax in [a1,a2]:ax.axvline(q3,color='red',ls=':');ax.axvline(qb,color='gray',ls=':');ax.set(xlim=(0,3.5),xlabel='Spatial frequency (cycles/nm)',ylabel='WLS amplitude ('+unit+')',ylim=(0,None))
        a3.step(x,n,where='mid',label='Rows contributing to spatial mean');a3.set(xlabel='Distance (nm)',ylabel='Number of rows',ylim=(-.2,5.5));a3.legend(fontsize=8)
        fig.suptitle(f'{group} paths | {key} | fixed rows, no spatial fill | red = 1/(3a) reference')
        fig.savefig(OUT/f'{group}_{key}_overview.png',dpi=150);plt.close(fig)
        sel=(f>=.5)&(f<=3.5);peak=float(f[sel][np.nanargmax(av[sel])]) if np.isfinite(av[sel]).any() else np.nan
        coeff=np.array([wls(y,q3,True) for y in vals]);bcoeff=np.array([wls(y,q3,True) for y in base[key][inds]]);c5=np.array([wls(y,q3,True) for y in uniform5[key][inds]])
        summary.append([group,key,int(np.isfinite(vals).sum()),int((np.isfinite(vals).sum(axis=1)==99).sum()),int(np.isfinite(coeff).sum()),peak,float(mean(abs(coeff))),float(abs(mean(coeff))),float(mean(abs(bcoeff))),float(mean(abs(c5)))])
        plotinfo.append(dict(group=group,key=key,unit=unit,rows=rows[inds].tolist(),coverage=np.isfinite(vals).sum(axis=1).tolist()))
# Recoverability-mask spectrum helps identify selection structure, not subtract it.
for feat in features:
    csvsave('source_codes_'+feat+'.csv',['distance_nm',*labels],np.column_stack([x,codes[feat].T]))
    csvsave('recovery_mask_frequency_'+feat+'.csv',['frequency_cycles_per_nm',*labels],np.column_stack([f,np.array([abs(spectrum((c>1).astype(float))) for c in codes[feat]]).T]))
csvsave('coverage.csv',['line','metric','recovered_valid','SG7_valid','SG5_valid','frequency_method'],coverage)
csvsave('q3_with_Bragg_controls.csv',['line','metric','method','valid','q3_amplitude','phase_rad','real','imag'],qtable)
csvsave('summary.csv',['group','metric','valid_out_of495','complete_lines','q3_valid_lines','dominant_native_f_0p5_to3p5','q3_mean_amplitude_with_Bragg','q3_coherent_amplitude_with_Bragg','q3_SG7_mean','q3_SG5_mean'],summary)
np.savez_compressed(OUT/'line_analysis.npz',x=x,f=f,rows=rows,labels=labels,q3=q3,qb=qb,E=E,raw=g,**arrout,**{'codes_'+k:v for k,v in codes.items()},**{'pair_unresolved_'+k:v for k,v in pair_flags.items()})
(OUT/'plot_index.json').write_text(json.dumps(plotinfo,indent=2),encoding='utf-8')
# Validate linear algebra with the actual masks: known q3 plus Bragg, not a detection test.
testerr=[]
for key in keys:
    for y in metrics[key]:
        yy=2*np.cos(2*np.pi*q3*x+.4)+3*np.sin(2*np.pi*qb*x)+.1*x
        yy[~np.isfinite(y)]=np.nan;c=wls(yy,q3,True)
        if np.isfinite(c):testerr.append(abs(abs(c)-2))
assert max(testerr)<1e-10
for feat in features:
    for j,k in zip(*np.where(np.isfinite(metrics['E_'+feat]))):assert metrics['h_'+feat][j,k]==np.interp(metrics['E_'+feat][j,k],E,g[j,k])
meta=json.loads((HERE/'results_v2/methods_and_validation.json').read_text(encoding='utf-8'));assert hashlib.sha256(Path(meta['source']).read_bytes()).hexdigest()==meta['source_sha256']
(OUT/'validation.json').write_text(json.dumps({'source_unchanged':True,'h_checked':True,'known_q3_recovery_max_error':max(testerr),'n_mask_tests':len(testerr),'spatial_interpolation':False,'frequency_min_points':40,'frequency_min_span_fraction':.8,'source_codes':{'0':'missing/unresolved','1':'original v2 SG7 candidate','2':'SG5 actual minimum fallback','3':'SG7 order4 minimum plus 3point derivative match <=2mV, prominence>=3noiseproxy, no boundary or competition'}},indent=2),encoding='utf-8')
print('RECOVERY:',json.dumps(recovery));print('SUMMARY:',json.dumps(summary));print(OUT)
