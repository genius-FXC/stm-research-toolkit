"""Read-only preliminary check of the exact L006 export shown by the user.

No q optimization, peak fitting, statistical significance or PDW identification.
"""
from pathlib import Path
import csv, json
import numpy as np
from scipy.signal import detrend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'exports/x_Bias_Spectroscopy-grid-p0313-001/L006_20261010_164024_e8fe'
a=np.load(SOURCE/'L006_numeric.npz')
s=a['s_nm']; bias=a['bias_mV']; signal=a['signal_pA']
assert signal.shape==(98,81) and np.all(a['xy'][:,1]==85)
assert np.allclose(np.diff(s),np.mean(np.diff(s)),rtol=1e-6)
w=np.hanning(len(s)); f=np.fft.rfftfreq(len(s),float(np.mean(np.diff(s))))
processed=detrend(signal,axis=0,type='linear')
amp=np.abs(np.fft.rfft(processed*w[:,None],axis=0))/w.sum()
amp[1:-1]*=2 # even N: Nyquist is not doubled
quality=Path(__import__('os').environ['STM_QUALITY_REPORT'])
cal=json.loads(quality.read_text(encoding='utf-8'))
spacing=cal['Se_column_spacing_nm']; f3=1/(3*spacing); fb=1/spacing
jlo=np.flatnonzero(f<f3)[-1]; jhi=jlo+1
with (OUT/'row85_fixed_frequency_amplitudes.csv').open('w',encoding='utf-8-sig',newline='') as file:
    writer=csv.writer(file);writer.writerow(['bias_mV','mean_LI_X_pA',f'FFT_A_at_{f[jlo]:.9f}_cycles_per_nm_pA',f'FFT_A_at_{f[jhi]:.9f}_cycles_per_nm_pA'])
    writer.writerows(zip(bias,signal.mean(axis=0),amp[jlo],amp[jhi]))
meta=dict(source_export=str(SOURCE),source_sha256=json.loads((SOURCE/'L006_line.json').read_text(encoding='utf-8'))['source_sha256'],display_pixel_endpoints=[[0,85],[97,85]],samples=len(s),line_length_nm=float(s[-1]),frequency_spacing_cycles_per_nm=float(f[1]),calibrated_Se_column_spacing_nm=spacing,conditional_three_times_column_period_nm=3*spacing,conditional_three_times_column_frequency_cycles_per_nm=f3,calibration_source=str(quality),fft_method='Spatial linear detrend; symmetric Hann; one-sided coherent-gain-corrected amplitude; native N=98; no zero padding; all measured biases',limitations=['3a here means three calibrated horizontal Se-column spacings; alignment with the claimed Fe next-nearest-neighbor lattice must be checked independently.','This field spans only approximately 3.55 candidate periods.','Finite-window broadening prevents interpreting adjacent native bins as different ordered states.','No gap estimation, q fitting, uncertainty estimate, noise significance, or PDW identification.','Raw LI X pA is not absolutely calibrated dI/dV.'])
(OUT/'quicklook_methods.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
fig,axs=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
for v,c in [(-8,'#3978a5'),(4,'#d67828'),(8,'#b63532'),(20,'#658158')]:
    k=int(np.argmin(abs(bias-v))); axs[0].plot(f[1:],amp[1:,k],label=f'{v:+d} mV',color=c,lw=1.5)
axs[0].axvline(f3,color='black',ls='--',lw=1,label='候选 1/(3a) ≈ 0.906')
axs[0].axvline(fb,color='#777777',ls=':',lw=1,label='原子周期 1/a ≈ 2.717')
axs[0].set(xlim=(0,3.5),ylim=(0,None),xlabel='空间频率 f (cycles/nm)',ylabel='FFT 幅度 (LI X, pA)',title='第 85 行：低频与原子周期成分')
axs[0].legend(fontsize=8);axs[0].grid(alpha=.13)
for j,c in [(jlo,'#3978a5'),(jhi,'#b63532')]:
    axs[1].plot(bias,amp[j],'.-',ms=3,color=c,label=f'f={f[j]:.3f} cycles/nm')
axs[1].axvspan(0,10,color='#d9b56e',alpha=.18,label='用户关注：0–10 mV')
axs[1].set(xlim=(-30,30),ylim=(0,None),xlabel='偏压 (mV)',ylabel='固定频点 FFT 幅度 (pA)',title='1/(3a) 两侧频点的能量变化')
axs[1].legend(fontsize=8);axs[1].grid(alpha=.13)
fig.suptitle('初步检查：98 个实测像素，空间去线性背景 + Hann 窗；不作 PDW 认定',fontsize=11)
fig.savefig(OUT/'row85_candidate_period_quicklook.png',dpi=180)
fig.savefig(OUT/'row85_candidate_period_quicklook.pdf')
plt.close(fig)
print(json.dumps({k:meta[k] for k in ['samples','line_length_nm','frequency_spacing_cycles_per_nm','calibrated_Se_column_spacing_nm','conditional_three_times_column_period_nm','conditional_three_times_column_frequency_cycles_per_nm']},ensure_ascii=False))
