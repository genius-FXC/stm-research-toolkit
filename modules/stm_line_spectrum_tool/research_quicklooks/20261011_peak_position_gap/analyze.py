"""Peak-position and half-depth gap analysis; all sampling from measured pixels."""
from pathlib import Path
import csv,json,sys,hashlib
import numpy as np
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from plotting import plt,NANOX
from core import GridDataset
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
for folder in ['01_peak_positions','02_halfheight_gap','03_controls','04_tables','05_origin']:(OUT/folder).mkdir(parents=True,exist_ok=True)
BASE=ROOT/'research_quicklooks/20261011_Se_rows070_076_082_088_094/results'
a=np.load(BASE/'Se_rows_frequency_energy.npz');oldmeta=json.loads((BASE/'methods_and_validation.json').read_text(encoding='utf-8'))
E=a['bias_mV'];x=a['distance_nm'];f=a['frequency_cycles_per_nm'][1:];q=float(a['q_third_cycles_per_nm']);qb=float(a['q_bragg_cycles_per_nm'])
rows=np.r_[a['reference_pixel_rows'],a['pixel_rows']].astype(int)
labels=[f'Ref{r:03d}' for r in rows[:6]]+[f'SeR{r:02d}' for r in a['Se_atom_rows']]
groups=['Transition']+['RefDark']*5+['SePath']*5
raw=np.concatenate([a['reference_raw_signal_pA'],a['raw_signal_pA']]);assert raw.shape==(11,99,81)
d=GridDataset(Path(oldmeta['source_file']));assert d.hash==oldmeta['source_sha256'];assert np.array_equal(raw,d.g[rows]);assert np.array_equal(E,d.bias)

def writecsv(path,headers,values):
    with path.open('w',encoding='utf-8-sig',newline='') as out:
        w=csv.writer(out);w.writerow(headers)
        for row in values:w.writerow(['' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for v in row])

def noise_proxy(g):
    z=np.r_[np.diff(g[(E>=40)&(E<=70)],n=2),np.diff(g[(E>=-70)&(E<=-40)],n=2)]
    return max(.001,float(1.4826*np.median(abs(z-np.median(z)))/np.sqrt(6)))

def quadratic(g,i,n):
    sl=slice(i-n//2,i+n//2+1);ee=E[sl]-E[i];yy=g[sl]
    if len(yy)!=n:return (np.nan,)*4+('window_edge',)
    D=np.column_stack([ee**2,ee,np.ones(n)]);coef=np.linalg.lstsq(D,yy,rcond=None)[0];aa,bb,cc=coef
    if aa>=-1e-10:return (np.nan,)*4+('not_concave',)
    delta=-bb/(2*aa);rms=float(np.sqrt(np.mean((yy-D@coef)**2)))
    if abs(delta)>2:return (np.nan,np.nan,rms,np.nan,'vertex_outside_one_step')
    se=np.nan
    if n>3:
        cov=np.linalg.inv(D.T@D)*np.sum((yy-D@coef)**2)/(n-3)
        grad=np.array([bb/(2*aa**2),-1/(2*aa),0.]);se=float(np.sqrt(max(0,grad@cov@grad)))
    return float(E[i]+delta),float(cc-bb*bb/(4*aa)),rms,se,'ok'

def peak(g,sign,window=None,prom_factor=1.):
    lo,hi=window or ((6,18) if sign=='pos' else (-24,-8))
    sig=noise_proxy(g);dynamic=np.mean(g[(abs(E)>=24)&(abs(E)<=30)])-min(g[abs(E)<=4]);thr=prom_factor*max(.05,3*sig,.01*dynamic)
    inds,props=find_peaks(g,prominence=thr,wlen=13)
    valid=np.flatnonzero((E[inds]>=lo)&(E[inds]<=hi))
    out=dict(discrete=np.nan,height=np.nan,prominence=np.nan,quad3=np.nan,height3=np.nan,quad5=np.nan,height5=np.nan,se5=np.nan,rms5=np.nan,delta35=np.nan,noise_proxy=sig,threshold=thr,candidates=len(valid),candidate_energy=np.nan,flag='no_clear_local_peak',quad_flag='no_peak')
    if not len(valid):return out
    k=valid[np.argmax(g[inds[valid]])];i=inds[k];prom=float(props['prominences'][k]);out['candidate_energy']=float(E[i])
    if E[i]<=lo or E[i]>=hi:out.update(flag='search_boundary',quad_flag='search_boundary');return out
    out.update(discrete=float(E[i]),height=float(g[i]),prominence=prom,flag='ok')
    ambiguous=any(abs(E[inds[v]]-E[i])>=4 and props['prominences'][v]>=.5*prom for v in valid if v!=k)
    if ambiguous:out['flag']='competing_peaks';out['quad_flag']='competing_peaks';return out
    e3,h3,_,_,flag3=quadratic(g,i,3);e5,h5,rms,se,flag5=quadratic(g,i,5)
    out.update(quad3=e3,height3=h3,rms5=rms,se5=se,delta35=abs(e5-e3) if np.isfinite(e5+e3) else np.nan)
    if flag5!='ok':out['quad_flag']=flag5
    elif not lo<e5<hi:out['quad_flag']='vertex_outside_search'
    elif rms>max(2*sig,.35*prom):out['quad_flag']='large_residual'
    elif not np.isfinite(e3) or abs(e5-e3)>1:out['quad_flag']='window_unstable'
    elif not np.isfinite(se) or se>1.5:out['quad_flag']='vertex_uncertain'
    else:out.update(quad5=e5,height5=h5,quad_flag='ok')
    return out

def crossing(e,r,i,level,step):
    k=i
    while 0<=k+step<len(e):
        j=k+step
        if r[k]<level<=r[j] and r[j]>r[k]:return float(e[k]+(level-r[k])/(r[j]-r[k])*(e[j]-e[k])),k,j
        k=j
    return np.nan,-1,-1

def halfgap(g,window=(24,30),fraction=.5,side_separate=False):
    lo,hi=window;neg=(E>=-hi)&(E<=-lo);pos=(E>=lo)&(E<=hi)
    bn,bp=float(np.mean(g[neg])),float(np.mean(g[pos]));en,ep=float(np.mean(E[neg])),float(np.mean(E[pos]));b=bn+(bp-bn)*(E-en)/(ep-en)
    if np.any(b[abs(E)<=20]<=0):return (np.nan,)*7
    r=g/b;inside=np.flatnonzero(abs(E)<=4);i=inside[np.argmin(r[inside])];minimum=float(r[i]);level=minimum+fraction*(1-minimum)
    if not 0<=minimum<.8:return (np.nan,)*7
    if side_separate:
        # Alternative: separate constant reference levels in raw LI X.
        rawmin=g[i];ln=rawmin+fraction*(bn-rawmin);lp=rawmin+fraction*(bp-rawmin)
        left,_,_=crossing(E,g,i,ln,-1);right,_,_=crossing(E,g,i,lp,1)
    else:
        left,_,_=crossing(E,r,i,level,-1);right,_,_=crossing(E,r,i,level,1)
    if not (-20<=left<E[i]<right<=20):return (np.nan,)*7
    return left,right,right-left,(right-left)/2,(left+right)/2,minimum,level

def spectrum(y):
    good=np.isfinite(y);n=int(good.sum());result=np.full(len(f),np.nan)
    if n<40 or np.ptp(x[good])<.8*x[-1]:return result
    xx=x[good];v=y[good];w=np.hanning(len(x))[good];D0=np.column_stack([np.ones(n),xx-xx.mean()]);sw=np.sqrt(w)
    for k,ff in enumerate(f):
        D=np.column_stack([D0,np.cos(2*np.pi*ff*xx),np.sin(2*np.pi*ff*xx)]);DW=D*sw[:,None]
        if np.linalg.cond(DW)>1e7:continue
        beta=np.linalg.lstsq(DW,v*sw,rcond=None)[0];result[k]=np.hypot(beta[-2],beta[-1])
    return result

def qfit(y,bragg=False):
    good=np.isfinite(y);n=int(good.sum())
    if n<40 or np.ptp(x[good])<.8*x[-1]:return np.nan,np.nan,np.full(99,np.nan)
    xx=x[good];D=np.column_stack([np.ones(n),xx-x.mean(),np.cos(2*np.pi*q*xx),np.sin(2*np.pi*q*xx)])
    if bragg:D=np.column_stack([D,np.cos(2*np.pi*qb*xx),np.sin(2*np.pi*qb*xx),np.cos(4*np.pi*qb*xx),np.sin(4*np.pi*qb*xx)])
    w=np.sqrt(np.hanning(99)[good]);DW=D*w[:,None]
    if np.linalg.cond(DW)>1e6:return np.nan,np.nan,np.full(99,np.nan)
    b=np.linalg.lstsq(DW,y[good]*w,rcond=None)[0];amp=float(np.hypot(b[2],b[3]));phase=float(np.arctan2(-b[3],b[2]));curve=b[0]+b[1]*(x-x.mean())+b[2]*np.cos(2*np.pi*q*x)+b[3]*np.sin(2*np.pi*q*x)
    return amp,phase,curve

metrics={};peak_records=[];gap_records=[];normalized=np.zeros_like(raw);backgrounds=np.zeros_like(raw)
keys=['discrete','height','prominence','quad3','height3','quad5','height5','se5','rms5','delta35']
for sign in ['pos','neg']:
    for key in keys:metrics[f'{sign}_{key}']=np.full((11,99),np.nan)
for key in ['edge_neg','edge_pos','gap_width','gap_half','gap_center','gap_min','gap_level','width_w20_30','width_w30_40','width_fraction40','width_fraction60','width_side_reference','peak_separation','neg_core_quad5','neg_strict_quad5','pos_height_relative','neg_height_relative']:
    metrics[key]=np.full((11,99),np.nan)
for j,(label,row,group) in enumerate(zip(labels,rows,groups)):
    for k,g in enumerate(raw[j]):
        for sign in ['pos','neg']:
            pp=peak(g,sign)
            for key in keys:metrics[f'{sign}_{key}'][j,k]=pp[key]
            peak_records.append([label,int(row),group,k,float(x[k]),sign,*[pp[key] for key in pp]])
        gap=halfgap(g)
        for key,v in zip(['edge_neg','edge_pos','gap_width','gap_half','gap_center','gap_min','gap_level'],gap):metrics[key][j,k]=v
        for key,win,frac,separate in [('width_w20_30',(20,30),.5,False),('width_w30_40',(30,40),.5,False),('width_fraction40',(24,30),.4,False),('width_fraction60',(24,30),.6,False),('width_side_reference',(24,30),.5,True)]:metrics[key][j,k]=halfgap(g,win,frac,separate)[2]
        metrics['peak_separation'][j,k]=metrics['pos_quad5'][j,k]-metrics['neg_quad5'][j,k]
        bn=g[(E>=-30)&(E<=-24)].mean();bp=g[(E>=24)&(E<=30)].mean();backgrounds[j,k]=bn+(bp-bn)*(E+27)/54;normalized[j,k]=g/backgrounds[j,k]
        metrics['neg_core_quad5'][j,k]=peak(g,'neg',(-22,-10))['quad5'];metrics['neg_strict_quad5'][j,k]=peak(g,'neg',prom_factor=1.5)['quad5']
        for sign in ['pos','neg']:
            en=metrics[f'{sign}_discrete'][j,k]
            if np.isfinite(en):metrics[f'{sign}_height_relative'][j,k]=metrics[f'{sign}_height'][j,k]/np.interp(en,E,backgrounds[j,k])
    print('extracted',label,{key:int(np.isfinite(metrics[key][j]).sum()) for key in ['pos_discrete','pos_quad5','neg_discrete','neg_quad5','gap_width']},flush=True)

writecsv(OUT/'04_tables/peak_measurements.csv',['line','display_row','group','pixel_x','distance_nm','sign',*list(pp)],peak_records)
allkeys=list(metrics)
writecsv(OUT/'04_tables/all_pixel_metrics.csv',['line','display_row','group','pixel_x','distance_nm',*allkeys],[[label,int(row),group,k,x[k],*[metrics[key][j,k] for key in allkeys]] for j,(label,row,group) in enumerate(zip(labels,rows,groups)) for k in range(99)])
metrics['topo_pm']=d.topo[rows].copy();metrics['g8_pA']=raw[:,:,np.argmin(abs(E-8))].copy();metrics['outer_background_pA']=backgrounds[:,:,40].copy()
spectral_keys=['pos_discrete','pos_quad3','pos_quad5','neg_discrete','neg_quad3','neg_quad5','neg_core_quad5','neg_strict_quad5','pos_height','neg_height','pos_height_relative','neg_height_relative','gap_width','gap_half','gap_center','edge_pos','edge_neg','width_w20_30','width_w30_40','width_fraction40','width_fraction60','width_side_reference','peak_separation','topo_pm','g8_pA','outer_background_pA']
spectra={key:np.array([spectrum(y) for y in metrics[key]]) for key in spectral_keys};qrecords=[]
qcurves={}
for key in spectral_keys:
    qcurves[key]=np.full((11,99),np.nan)
    for j,label in enumerate(labels):
        aq,pq,curve=qfit(metrics[key][j]);ab,pb,_=qfit(metrics[key][j],True);qcurves[key][j]=curve
        qrecords.append([label,int(rows[j]),groups[j],key,int(np.isfinite(metrics[key][j]).sum()),aq,pq,ab,pb])
    writecsv(OUT/f'04_tables/{key}_real.csv',['distance_nm',*labels],np.column_stack([x,metrics[key].T]))
    writecsv(OUT/f'04_tables/{key}_frequency.csv',['frequency_cycles_per_nm',*labels],np.column_stack([f,spectra[key].T]))
writecsv(OUT/'04_tables/q3_fits.csv',['line','display_row','group','metric','valid_pixels','amplitude_q3','phase_rad_display_x0','amplitude_q3_with_Bragg_harmonics','phase_with_Bragg_rad'],qrecords)

# Deterministic fixed-template control: spectral shape is shared along each row.
# Amplitude, constant background and slope vary with measured fitted coefficients.
null_raw=np.zeros_like(raw);null_width=np.zeros((11,99));null_neg=np.full((11,99),np.nan);null_pos=np.full((11,99),np.nan);coeffs=np.zeros((11,99,3))
fitmask=((abs(E)>=24)&(abs(E)<=40))|(abs(E)<=2)
for j in range(11):
    template=raw[j].mean(axis=0);D=np.column_stack([template,np.ones(81),E]);coef=np.linalg.lstsq(D[fitmask],raw[j][:,fitmask].T,rcond=None)[0]
    assert np.all(coef[0]>0);coeffs[j]=coef.T;null_raw[j]=(D@coef).T
    for k,g in enumerate(null_raw[j]):
        null_width[j,k]=halfgap(g)[2];null_neg[j,k]=peak(g,'neg')['quad5'];null_pos[j,k]=peak(g,'pos')['quad5']
null_spec=np.array([spectrum(y) for y in null_width])
null_rows=[]
for j,label in enumerate(labels):
    null_rows.append([label,qfit(metrics['gap_width'][j])[0],qfit(null_width[j])[0],qfit(metrics['neg_quad5'][j])[0],qfit(null_neg[j])[0]])
writecsv(OUT/'04_tables/fixed_shape_control.csv',['line','observed_width_q3_meV','fixed_shape_width_q3_meV','observed_negative_peak_q3_meV','fixed_shape_negative_peak_q3_meV'],null_rows)
writecsv(OUT/'04_tables/fixed_shape_gap_width_real.csv',['distance_nm',*labels],np.column_stack([x,null_width.T]))
writecsv(OUT/'04_tables/fixed_shape_gap_width_frequency.csv',['frequency_cycles_per_nm',*labels],np.column_stack([f,null_spec.T]))

# Conditional residual block-bootstrap sensitivity; does not make a hypothesis-test p value.
rng=np.random.default_rng(20261011);boot=[]
for key in ['neg_quad5','gap_width']:
    for j,label in enumerate(labels):
        y=metrics[key][j];ok=np.isfinite(y)
        if ok.sum()<70:continue
        xx=x[ok];D=np.column_stack([np.ones(ok.sum()),xx-x.mean(),np.cos(2*np.pi*q*xx),np.sin(2*np.pi*q*xx),np.cos(2*np.pi*qb*xx),np.sin(2*np.pi*qb*xx),np.cos(4*np.pi*qb*xx),np.sin(4*np.pi*qb*xx)])
        b=np.linalg.lstsq(D,y[ok],rcond=None)[0];prediction=D@b;res=y[ok]-prediction
        # Blocks are consecutive valid samples; gaps mean this is only a sensitivity analysis.
        for size in [3,6,9]:
            amps=[]
            for trial in range(200):
                start=rng.integers(0,len(res),size=int(np.ceil(len(res)/size)));rr=np.concatenate([res[(v+np.arange(size))%len(res)] for v in start])[:len(res)];bb=np.linalg.lstsq(D,prediction+rr,rcond=None)[0];amps.append(np.hypot(bb[2],bb[3]))
            boot.append([label,key,size,float(np.hypot(b[2],b[3])),*np.percentile(amps,[2.5,50,97.5]).tolist()])
writecsv(OUT/'04_tables/conditional_block_bootstrap.csv',['line','metric','consecutive_valid_sample_block','unweighted_q3_amplitude','conditional_p025','conditional_median','conditional_p975'],boot)

# Fully sampled width permits an ordinary spatial FFT as an independent display.
design=np.column_stack([np.ones(99),x]);residual=metrics['gap_width']-np.array([design@np.linalg.lstsq(design,y,rcond=None)[0] for y in metrics['gap_width']]);width_fft=abs(np.fft.rfft(residual*np.hanning(99),axis=1))[:,1:]*2/np.hanning(99).sum()
writecsv(OUT/'04_tables/gap_width_HannFFT.csv',['frequency_cycles_per_nm',*labels],np.column_stack([f,width_fft.T]))
control_records=[]
for j,label in enumerate(labels):
    base=np.column_stack([np.ones(99),x-x.mean(),np.cos(2*np.pi*q*x),np.sin(2*np.pi*q*x),np.cos(2*np.pi*qb*x),np.sin(2*np.pi*qb*x),np.cos(4*np.pi*qb*x),np.sin(4*np.pi*qb*x)])
    amplitudes=[]
    for extra in [[],['topo_pm'],['topo_pm','outer_background_pA']]:
        D=np.column_stack([base,*[(metrics[key][j]-metrics[key][j].mean())/metrics[key][j].std() for key in extra]])
        sw=np.sqrt(np.hanning(99));b=np.linalg.lstsq(D*sw[:,None],metrics['gap_width'][j]*sw,rcond=None)[0];amplitudes.append(float(np.hypot(b[2],b[3])))
    control_records.append([label,*amplitudes,float(np.corrcoef(metrics['gap_width'][j],metrics['topo_pm'][j])[0,1])])
writecsv(OUT/'04_tables/q3_width_covariate_controls.csv',['line','q3_with_Bragg','q3_with_Bragg_Z','q3_with_Bragg_Z_background','Pearson_width_Z'],control_records)

def save(fig,folder,name):
    for ext in ['png','pdf']:fig.savefig(OUT/folder/f'{name}.{ext}',dpi=160)
    plt.close(fig)

colors=plt.get_cmap('viridis')(np.linspace(.1,.85,5))
for j,(label,row) in enumerate(zip(labels,rows)):
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for t,sign in enumerate(['pos','neg']):
        ax=axs[t,0]
        for suffix,sty,color,lab in [('discrete','s','#aaa','离散极大值'),('quad3','o','#1c83b5','3 点二次拟合'),('quad5','o','#ce593c','5 点拟合（通过质控）')]:ax.plot(x,metrics[f'{sign}_{suffix}'][j],sty,ms=3,color=color,label=lab)
        ax.set(xlabel='距离（nm）',ylabel='峰位（meV）',title=('正峰' if sign=='pos' else '负峰')+f'：5 点合格 {np.isfinite(metrics[sign+"_quad5"][j]).sum()}/99');ax.legend(fontsize=8)
        ax=axs[t,1]
        plotted=False
        for suffix,color,lab in [('discrete','#aaa','离散'),('quad3','#1c83b5','3 点'),('quad5','#ce593c','5 点')]:
            yy=spectra[f'{sign}_{suffix}'][j]
            if np.isfinite(yy).any():ax.plot(f,yy,'.-',ms=3,color=color,label=lab);plotted=True
        ax.axvline(q,color='#c49000',ls='--',label='1/(3a)');ax.axvline(qb,color='#777',ls=':',label='1/a');ax.set(xlim=(0,3.5),ylim=(0,None),xlabel='空间频率（cycles/nm）',ylabel='正弦回归幅度（meV）')
        if not plotted:ax.text(.5,.5,'有效峰位不足 40 点\n不补值计算频谱',transform=ax.transAxes,ha='center')
        ax.legend(fontsize=8)
    fig.suptitle(f'{label} · 像素行 {row} · 峰位实空间 / 频率空间',fontsize=12);save(fig,'01_peak_positions',f'{label}_peak_positions')
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    axs[0,0].plot(x,metrics['edge_pos'][j],label='正侧半深度交点');axs[0,0].plot(x,metrics['edge_neg'][j],label='负侧半深度交点');axs[0,0].set(ylabel='交点能量（meV）',xlabel='距离（nm）');axs[0,0].legend(fontsize=8)
    axs[0,1].plot(x,metrics['gap_width'][j],color='#008c99',label='半深度全宽 W50');axs[0,1].plot(x,qcurves['gap_width'][j],color='#b07800',ls='--',label='趋势 + 固定 1/(3a)');axs[0,1].set(xlabel='距离（nm）',ylabel='W50（meV）');axs[0,1].legend(fontsize=8)
    for key,lab in [('gap_width','背景 24–30 mV'),('width_w20_30','背景 20–30 mV'),('width_w30_40','背景 30–40 mV')]:axs[1,0].plot(f,spectra[key][j],'.-',ms=3,label=lab)
    axs[1,0].axvline(q,color='#b07800',ls='--');axs[1,0].axvline(qb,color='#777',ls=':');axs[1,0].set(xlim=(0,3.5),ylim=(0,None),xlabel='空间频率（cycles/nm）',ylabel='W50 调制幅度（meV）');axs[1,0].legend(fontsize=8)
    axs[1,1].plot(x,metrics['gap_center'][j],label='半深度中心');axs[1,1].set(xlabel='距离（nm）',ylabel='中心能量（meV）');axs[1,1].legend(fontsize=8)
    fig.suptitle(f'{label} · 行 {row} · 半深度宽度与周期（Δ50 = W50/2）',fontsize=12);save(fig,'02_halfheight_gap',f'{label}_halfheight_gap')

fig,axs=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
for ax,key,title in zip(axs.flat,['pos_discrete','pos_quad5','neg_discrete','neg_quad5'],['正峰：离散','正峰：5 点合格','负峰：离散','负峰：5 点合格']):
    v=metrics[key];im=ax.imshow(np.ma.masked_invalid(v),aspect='auto',extent=(-.5,98.5,10.5,-.5),cmap=NANOX,interpolation='nearest');ax.set_yticks(range(11),[f'{lab} / {row}' for lab,row in zip(labels,rows)],fontsize=8);ax.set(title=title,xlabel='实测像素 x');fig.colorbar(im,ax=ax,label='峰位（meV）');ax.set_facecolor('#ddd')
fig.suptitle('峰位覆盖：灰色为缺失 / 不合格；正负峰采用各自色阶');save(fig,'01_peak_positions','peak_coverage_overview')

fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for j,c in zip(range(6,11),colors):
    axs[0].plot(x,metrics['neg_quad5'][j],'o',ms=3,color=c,label=labels[j]);axs[1].plot(f,spectra['neg_quad5'][j],'.-',ms=3,color=c,label=labels[j])
axs[0].set(xlabel='距离（nm）',ylabel='负峰拟合峰位（meV）',title='过 Se 五线 · 5 点合格峰位');axs[0].legend(fontsize=8)
axs[1].axvline(q,color='#b07800',ls='--',label='1/(3a)');axs[1].axvline(qb,color='#888',ls=':',label='1/a');axs[1].set(xlim=(0,3.5),ylim=(0,None),xlabel='空间频率（cycles/nm）',ylabel='正弦回归幅度（meV）',title='包含缺失掩码的加权回归幅度谱');axs[1].legend(fontsize=8);save(fig,'01_peak_positions','Se_negative_peak_real_frequency')

fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
for rr,inds,name in [(0,range(1,6),'此前 dark 五线'),(1,range(6,11),'过 Se 五线')]:
    for j,c in zip(inds,colors):
        axs[rr,0].plot(x,metrics['gap_width'][j],color=c,lw=.8,label=labels[j]);axs[rr,1].plot(f,spectra['gap_width'][j],color=c,lw=.9,label=labels[j]);axs[rr,2].plot(x,metrics['gap_center'][j],color=c,lw=.8,label=labels[j])
    axs[rr,0].set(xlabel='距离（nm）',ylabel='W50（meV）',title=name+' · 半深度全宽');axs[rr,0].legend(fontsize=7,ncol=2)
    axs[rr,1].axvline(q,color='#b07800',ls='--');axs[rr,1].axvline(qb,color='#777',ls=':');axs[rr,1].set(xlim=(0,3.5),ylim=(0,None),xlabel='空间频率（cycles/nm）',ylabel='W50 调制幅度（meV）',title='频率空间：虚线为 1/(3a)')
    axs[rr,2].set(xlabel='距离（nm）',ylabel='中心能量（meV）',title='半深度中心位置')
save(fig,'02_halfheight_gap','gap_width_real_frequency_overview')

fig,axs=plt.subplots(1,2,figsize=(11.5,4.5),layout='constrained')
for inds,c,lab in [(range(1,6),'#168da1','此前 dark 五线'),(range(6,11),'#ba8725','过 Se 五线')]:
    for ax,values,title in [(axs[0],spectra['gap_width'],'逐行回归幅度的平均'),(axs[1],width_fft,'逐行 Hann FFT 幅度的平均')]:
        vals=values[list(inds)];mean=vals.mean(axis=0);sd=vals.std(axis=0,ddof=1);ax.plot(f,mean,'.-',color=c,label=lab);ax.fill_between(f,np.maximum(0,mean-sd),mean+sd,color=c,alpha=.14);ax.set(xlim=(0,3.5),ylim=(0,None),xlabel='空间频率（cycles/nm）',ylabel='W50 幅度（meV）',title=title);ax.axvline(q,color='#b07800',ls='--');ax.axvline(qb,color='#777',ls=':');ax.legend(fontsize=8)
for ax,values in zip(axs,[spectra['gap_width'],width_fft]):
    ymax=max(float(np.max(values[list(inds)].mean(axis=0)[f<=3.5]+values[list(inds)].std(axis=0,ddof=1)[f<=3.5])) for inds in [range(1,6),range(6,11)])
    ax.set_ylim(0,ymax*1.08)
fig.suptitle('阴影为行间 SD，不是测量误差；两种频率分析交叉检查');save(fig,'02_halfheight_gap','gap_group_mean_frequency')

fig,axs=plt.subplots(1,2,figsize=(11.5,4.5),layout='constrained')
for ax,key,title in [(axs[0],'neg_core_quad5','搜索窗缩为 -22..-10 mV'),(axs[1],'neg_strict_quad5','峰突出度门槛提高 50%')]:
    for j,c in zip(range(6,11),colors):ax.plot(f,spectra[key][j],'.-',ms=3,color=c,label=labels[j])
    ax.set(xlim=(0,3.5),ylim=(0,None),xlabel='空间频率（cycles/nm）',ylabel='负峰位回归幅度（meV）',title=title);ax.axvline(q,color='#b07800',ls='--');ax.axvline(qb,color='#777',ls=':');ax.legend(fontsize=8)
save(fig,'03_controls','negative_peak_selection_sensitivity')

# QC examples selected by quality flags and fixed x positions, not periodic phase.
examples=[(3,20),(3,50),(8,20),(8,50)]
fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for ax,(j,k) in zip(axs.flat,examples):
    ax.plot(E,normalized[j,k],'ko-',ms=3,lw=.9,label='原始谱 / 背景');ax.axhline(1,color='#888',ls=':',label='背景 = 1');ax.axhline(metrics['gap_level'][j,k],color='#008c99',ls='--',label='半深度阈值')
    for key in ['edge_neg','edge_pos']:ax.axvline(metrics[key][j,k],color='#008c99',ls='--')
    for sign,c in [('pos','#b84633'),('neg','#286ab7')]:
        for suffix,marker in [('discrete','s'),('quad5','*')]:
            en=metrics[f'{sign}_{suffix}'][j,k]
            if np.isfinite(en):ax.plot(en,np.interp(en,E,normalized[j,k]),marker,color=c,ms=8,label=f'{sign} {suffix}')
    ax.set(xlim=(-32,32),ylim=(-.1,1.6),xlabel='偏压（mV）',ylabel='背景归一化 LI X',title=f'{labels[j]} / x={k}; W50={metrics["gap_width"][j,k]:.2f} meV');ax.legend(fontsize=7,ncol=2)
save(fig,'03_controls','extraction_examples')

fig,axs=plt.subplots(1,2,figsize=(11.5,4.4),layout='constrained')
for ax,inds,name in zip(axs,[range(1,6),range(6,11)],['此前 dark 五线','过 Se 五线']):
    observed=[qfit(metrics['gap_width'][j])[0] for j in inds];null=[qfit(null_width[j])[0] for j in inds];xx=np.arange(5)
    ax.bar(xx-.18,observed,.36,label='实测 W50');ax.bar(xx+.18,null,.36,label='固定谱形背景对照');ax.set_xticks(xx,[labels[j] for j in inds],fontsize=8);ax.set(ylabel='1/(3a) 拟合幅度（meV）',title=name);ax.legend(fontsize=8)
save(fig,'03_controls','fixed_shape_q3_control')

mask_spectra={}
for key in ['pos_quad5','neg_quad5']:
    mask=np.isfinite(metrics[key]).astype(float);mask_spectra[key]=np.abs(np.fft.rfft((mask-mask.mean(axis=1,keepdims=True))*np.hanning(99),axis=1))[:,1:]*2/np.hanning(99).sum()
    writecsv(OUT/f'04_tables/{key}_validity_mask_frequency.csv',['frequency_cycles_per_nm',*labels],np.column_stack([f,mask_spectra[key].T]))

# Numeric checks on synthetic spectra and definitions, independent of source data.
gaussian=lambda shift: .1+np.exp(-(E-(10+shift))**2/(2*3**2))
ep=peak(gaussian(.6),'pos');assert abs(ep['quad3']-10.6)<.15
test=1-.9*np.exp(-E**2/(2*6**2));hg=halfgap(test);assert abs(hg[2]-2*np.sqrt(2*np.log(2))*6)<.2
maxscale=max(abs(halfgap(g*3.7)[2]-halfgap(g)[2]) for g in raw.reshape(-1,81));assert maxscale<1e-10
assert np.array_equal(raw,d.g[rows]);assert hashlib.sha256(Path(oldmeta['source_file']).read_bytes()).hexdigest()==oldmeta['source_sha256']

np.savez_compressed(OUT/'04_tables/analysis_arrays.npz',bias_mV=E,distance_nm=x,frequency_cycles_per_nm=f,rows=rows,labels=np.array(labels),groups=np.array(groups),raw_signal_pA=raw,normalized_signal=normalized,backgrounds=backgrounds,null_raw_signal=null_raw,null_gap_width=null_width,null_coefficients=coeffs,null_gap_frequency=null_spec,gap_width_HannFFT=width_fft,q3=q,qBragg=qb,**metrics,**{f'freq_{k}':v for k,v in spectra.items()},**{f'qcurve_{k}':v for k,v in qcurves.items()})
summary=dict(source=oldmeta['source_file'],source_sha256=oldmeta['source_sha256'],input_arrays=str(BASE/'Se_rows_frequency_energy.npz'),pixel_rows=rows.tolist(),labels=labels,groups=groups,
             extraction_windows_mV={'positive':[6,18],'negative':[-24,-8]},peak_prominence_rule='max(0.05 pA, 3 * outer second-difference noise proxy, 1% outer-reference minus central-minimum)',
             quadratic_rules='concave; vertex within 2 mV of measured peak; RMS <= max(2*noise proxy, .35*prominence); |vertex5-vertex3| <=1 mV; conditional vertex SE<=1.5 mV; reject comparable competing peaks',
             halfheight_definition='background line connects mean g at -30..-24 and +24..+30 mV; r=g/background; rmin in -4..4 mV; threshold=rmin+.5*(1-rmin); first outward crossings; linear energy interpolation; W50=Eright-Eleft, Delta50=W50/2',
             spectrum_method='Hann-weighted sine/cosine regression with intercept and linear trend, native full-line frequencies; missing positions excluded, never filled; minimum 40 valid samples and >=80% line span; not a zero-filled FFT',
             q3=q,qBragg=qb,frequency_step=float(f[0]),line_length_nm=float(x[-1]),candidate_periods_in_line=float(x[-1]*q),
             instrument=dict(bias_step_mV=2,lockin_header_amplitude_V=.0016,lockin_amplitude_peak_or_RMS='not specified by header',temperature1_K=4.24373,temperature2_K=4.8362,sweeps=1),
             counts={label:{key:int(np.isfinite(metrics[key][j]).sum()) for key in ['pos_discrete','pos_quad3','pos_quad5','neg_discrete','neg_quad3','neg_quad5','gap_width','peak_separation']} for j,label in enumerate(labels)},
             validation=dict(source_spectra_match_frozen_arrays=True,source_hash_unchanged=True,synthetic_quadratic_peak_error_meV=abs(ep['quad3']-10.6),synthetic_halfwidth_error_meV=abs(hg[2]-2*np.sqrt(2*np.log(2))*6),multiplicative_gain_width_max_error_meV=maxscale),
             caveats=['Original spectra do not have two resolved peaks everywhere; no manufactured peaks.', 'Half-depth width is an operational spectral suppression width, not a model fit to order-parameter gap.', 'Missing-peak mask may carry lattice structure.', 'Conditional fit SE is not total experimental uncertainty.', 'Block bootstrap is sensitivity, not a detection p value.', 'Fixed-template control is illustrative, not a calibrated normal-state model.', 'Only ~3.6 candidate periods fit in each line; no PDW claim.'])
(OUT/'04_tables/methods_and_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False),flush=True)
