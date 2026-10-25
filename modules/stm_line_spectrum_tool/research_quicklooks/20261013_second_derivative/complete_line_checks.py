from pathlib import Path
import numpy as np,csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'results_v3_lines';a=np.load(OUT/'line_analysis.npz');x=a['x'];f=a['f'];q=float(a['q3']);qb=float(a['qb']);w=np.hanning(99)
records=[];summ={}
for key in ['E_in_pos','h_in_pos']:
    vals=a['real_'+key][5:];assert np.isfinite(vals).all()
    fft=a['fft_'+key][5:];meanfft=fft.mean(axis=0)
    with (OUT/f'Fe_{key}_TrueFFT.csv').open('w',encoding='utf-8-sig',newline='') as h:
        wr=csv.writer(h);wr.writerow(['frequency_cycles_per_nm',*a['labels'][5:].tolist(),'MeanAmplitude']);wr.writerows(np.column_stack([f,fft.T,meanfft]))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for j,v in enumerate(vals):
        X=np.column_stack([np.ones(99),x-x.mean(),np.cos(2*np.pi*q*x),np.sin(2*np.pi*q*x),np.cos(4*np.pi*q*x),np.sin(4*np.pi*q*x),np.cos(2*np.pi*qb*x),np.sin(2*np.pi*qb*x)])
        b=np.linalg.lstsq(X*np.sqrt(w)[:,None],v*np.sqrt(w),rcond=None)[0]
        records.append([str(a['labels'][j+5]),key,float(np.hypot(*b[2:4])),float(np.arctan2(-b[3],b[2])),float(np.hypot(*b[4:6])),float(np.hypot(*b[6:8]))])
        axes[0].plot(f,fft[j],lw=.7)
    axes[0].plot(f,meanfft,'k',lw=2,label='Mean of 5 FFT amplitudes')
    for meth in ['SG7','SG5']:
        vv=a[meth+'_'+key][5:];fs=[]
        for y in vv:
            assert np.isfinite(y).all();r=y-np.polyval(np.polyfit(x,y,1),x);r-=r.mean();fs.append(2*abs(np.fft.rfft(r*w)[1:])/w.sum())
        axes[1].plot(f,np.mean(fs,axis=0),label=meth)
    for ax in axes:
        for fq,c,lab in [(q,'red','q3'),(2*q,'orange','2q3'),(qb,'gray','qB')]:ax.axvline(fq,color=c,ls=':',label=lab)
        ax.set(xlim=(0,3.5),ylim=(0,None),xlabel='Spatial frequency (cycles/nm)',ylabel='FFT amplitude ('+('pA' if key.startswith('h') else 'meV')+')');ax.legend(fontsize=8)
    fig.suptitle('Fe '+key+' | all 5 lines complete (99/99), no recovered pixels')
    fig.savefig(OUT/f'Fe_{key}_TrueFFT.png',dpi=170);plt.close(fig)
    peak=f[(f>=.5)&(f<=3.5)][np.argmax(meanfft[(f>=.5)&(f<=3.5)])]
    summ[key]={'dominant_fft_frequency':float(peak),'mean_amplitude_at_1cycle':float(meanfft[np.argmin(abs(f-1))]),'mean_amplitude_at_1p75cycles':float(meanfft[np.argmin(abs(f-1.75))])}
with (OUT/'q3_and_2q3_joint_fit.csv').open('w',encoding='utf-8-sig',newline='') as h:
    wr=csv.writer(h);wr.writerow(['line','metric','q3_amplitude','q3_phase','2q3_amplitude','Bragg_amplitude']);wr.writerows(records)
(OUT/'complete_line_fft_check.json').write_text(json.dumps(summ,indent=2),encoding='utf-8');print(json.dumps(summ));print(records)
