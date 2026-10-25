"""One spatial FFT per measured bias, using the exact displayed L006 row 85.

Read-only source validation; native frequency bins, no energy interpolation.
"""
from pathlib import Path
import sys, json, csv, shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from core import GridDataset, bin_edges
from profile_fft import profile_fft
from plotting import NANOX
import matplotlib.pyplot as plt

SOURCE=ROOT/'exports/x_Bias_Spectroscopy-grid-p0313-001/L006_20261010_164024_e8fe'
OUT=Path(__file__).resolve().parent/'frequency_energy'
OUT.mkdir(exist_ok=True)
a=np.load(SOURCE/'L006_numeric.npz')
source_meta=json.loads((SOURCE/'L006_line.json').read_text(encoding='utf-8'))
data=GridDataset(source_meta['source_file'])
s=a['s_nm']; E=a['bias_mV']; signal=a['signal_pA']; xy=a['xy'].astype(int)
assert signal.shape==(98,81) and np.array_equal(xy,np.column_stack([np.arange(98),np.full(98,85)]))
assert data.hash==source_meta['source_sha256']
raw_error=float(np.max(abs(signal-data.g[xy[:,1],xy[:,0],:])))
assert raw_error==0 and np.array_equal(E,data.bias)
assert np.allclose(np.diff(s),np.mean(np.diff(s)),rtol=1e-6)
design=np.column_stack([np.ones(len(s)),s])
trend=design@np.linalg.lstsq(design,signal,rcond=None)[0]
processed=signal-trend
ffts=[profile_fft(s,processed[:,k],'hann') for k in range(len(E))]
assert all(not f['resampled'] and f['zero_padding']==0 for f in ffts)
f=np.asarray(ffts[0]['frequency_cycles_per_nm'])
amp=np.array([v['amplitude'] for v in ffts])
coeff=np.array([np.asarray(v['coefficient_real'])+1j*np.asarray(v['coefficient_imag']) for v in ffts])
mean=signal.mean(axis=0)
assert np.all(mean>0)
relative=amp/mean[:,None]
quality=Path(__import__('os').environ['STM_QUALITY_REPORT'])
cal=json.loads(quality.read_text(encoding='utf-8'))
spacing=cal['Se_column_spacing_nm']; fb=1/spacing; f3=fb/3
visible_e=(E>=-30)&(E<=30); visible_f=(f>0)&(f<=3.5)
limits=dict(absolute=[0,float(np.percentile(amp[np.ix_(visible_e,visible_f)],99))],relative=[0,float(np.percentile(relative[np.ix_(visible_e,visible_f)],99))])
np.savez_compressed(OUT/'row85_frequency_energy.npz',bias_mV=E,frequency_cycles_per_nm=f,fft_amplitude_pA=amp,fft_relative_amplitude=relative,fft_raw_coefficient_real=coeff.real,fft_raw_coefficient_imag=coeff.imag,signal_pA=signal,linear_background_pA=trend,processed_signal_pA=processed,mean_signal_pA=mean,distance_nm=s,display_xy_px=xy,topo_plane_subtracted_pm=data.topo,source_z_pm=data.z,fft_window_weights=np.hanning(len(s)),q_bragg_cycles_per_nm=fb,q_third_cycles_per_nm=f3)

def table(name,headers,values):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.writer(file);writer.writerow(headers);writer.writerows(values)

table('row85_FFT_amplitude_pA.csv',['bias_mV',*[f'f_{v:.12g}_cycles_per_nm' for v in f]],np.column_stack([E,amp]))
table('row85_FFT_relative_amplitude.csv',['bias_mV',*[f'f_{v:.12g}_cycles_per_nm' for v in f]],np.column_stack([E,relative]))
table('row85_frequency_axis.csv',['frequency_index_zero_based','frequency_cycles_per_nm','period_nm','frequency_over_horizontal_Bragg'],[[i,v,'' if v==0 else 1/v,v/fb] for i,v in enumerate(f)])
table('row85_FFT_complex_coefficients.csv',['bias_mV','frequency_cycles_per_nm','raw_DFT_real_pA','raw_DFT_imag_pA','amplitude_pA','phase_rad'],[[e,freq,c.real,c.imag,av,float(np.angle(c))] for e,cs,amps in zip(E,coeff,amp) for freq,c,av in zip(f,cs,amps)])
shutil.copyfile(SOURCE/'L006_sample_positions.csv',OUT/'row85_sample_positions.csv')
shutil.copyfile(SOURCE/'L006_full_spectra_used_pA.csv',OUT/'row85_raw_pixel_spectra_pA.csv')
shutil.copyfile(SOURCE/'Nanox.pal',OUT/'Nanox.pal')
meta=dict(source_file=str(data.path),source_export=str(SOURCE),source_sha256=data.hash,line_id_in_user_figure='L006',display_pixel_endpoints=[[0,85],[97,85]],source_row_zero_based=13,display_coordinate_origin='top left; x right; y down; source row=98-display_y',region='R1_dark; 1x1 assigned by user; no full-region averaging',sample_count=len(s),line_length_nm=float(s[-1]),physical_step_nm=ffts[0]['mean_step_nm'],frequency_spacing_cycles_per_nm=ffts[0]['frequency_spacing_cycles_per_nm'],frequency_count_with_DC=len(f),bias_count=len(E),bias_step_mV=float(E[1]-E[0]),bias_range_mV=[float(E[0]),float(E[-1])],fft_axis='spatial distance x, separately for each measured bias; NOT transform along energy',fft_processing='Least-squares constant+slope along physical distance, then remove residual mean, symmetric Hann, native-length rFFT, one-sided amplitude corrected by sum(window)',fft_amplitude_formula='abs(DFT)/sum(window), double positive frequencies except even-N Nyquist',fft_no_interpolation=True,fft_zero_padding=0,DC_displayed=False,energy_smoothing=False,raw_LI_X_unit='pA; not absolutely calibrated dI/dV',relative_display='same FFT amplitude divided by positive raw line-mean LI X at that bias; supplementary contrast view; near-zero mean can amplify noise',absolute_color_limits_pA=limits['absolute'],relative_color_limits=limits['relative'],color_limits_policy='one fixed linear scale across all energies per panel; P99 of visible positive-frequency bins in -30..30 mV and f<=3.5; clipping only affects color display',reference_horizontal_atomic_spacing_nm=spacing,reference_Bragg_frequency_cycles_per_nm=fb,candidate_third_frequency_cycles_per_nm=f3,candidate_third_period_nm=3*spacing,reference_meaning='a is the calibrated horizontal Se column spacing; correspondence to Fe next-nearest-neighbor direction requires lattice registration',reference_lines='1/(3a), 2/(3a), 1/a; guides are hypotheses, not detected/fitted peaks',validation=dict(source_hash_matches_export=True,raw_pixel_spectra_max_error_pA=raw_error,all_biases_match_source=True,direct_uniform_sampling=True),limitations=['Line contains about 3.55 candidate 3a periods. Native df is about 0.253 cycles/nm.','The guides do not establish exact 3a commensurability, non-dispersion, or PDW.','0..10 mV contains only six measured biases and is not an established gap interval.','Magnitude images contain no phase information; raw complex coefficients are retained for later work.'])
(OUT/'row85_methods_and_validation.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')

def decorate_freq(ax,span,full=False):
    ax.set(xlim=(0,float(f[-1]) if full else 3.5),ylim=span,xlabel='空间频率 f（cycles/nm）',ylabel='偏压 V（mV）')
    for val,label in [(f3,'1/(3a)'),(2*f3,'2/(3a)'),(fb,'1/a')]:
        ax.axvline(val,color='#00c9c9',ls='--',lw=.9,alpha=.95)
        ax.text(val+.03,span[1]-3,label,color='#00c9c9',fontsize=8,ha='left',va='top',bbox=dict(fc='black',ec='none',alpha=.6,pad=1.5))
    ax.axhline(0,color='#ffffff',lw=.6,alpha=.6)
    ax.axhline(10,color='#ffffff',ls=':',lw=.7,alpha=.75)

def freq_plot(ax,values,mode='absolute',span=(-30,30),full=False):
    scale=limits[mode]
    if full: scale=[0,float(np.percentile(values[:,1:],99))]
    im=ax.pcolormesh(bin_edges(f[1:]),bin_edges(E),values[:,1:],cmap=NANOX,shading='flat',vmin=scale[0],vmax=scale[1],rasterized=True)
    decorate_freq(ax,span,full);return im

fig,axes=plt.subplots(1,3,figsize=(14.1,5.1),layout='constrained',gridspec_kw={'width_ratios':[.83,1,1.12]})
lo,hi=np.percentile(data.topo,[1,99])
axes[0].imshow(data.topo,cmap=NANOX,vmin=lo,vmax=hi,origin='upper',extent=(-.5,98.5,98.5,-.5),interpolation='nearest')
axes[0].plot(xy[:,0],xy[:,1],color='#00e4df',lw=1.4)
axes[0].scatter([0,97],[85,85],s=25,c='#00e4df',edgecolors='white',lw=.8,zorder=5)
axes[0].axhline(67.14,c='#cccccc',ls='--',lw=.8)
axes[0].text(5,76,'dark / 1×1',color='white',fontsize=9)
for xx,label in [(0,'A'),(97,'B')]:axes[0].annotate(label,(xx,85),xytext=(3,-11),textcoords='offset points',color='#00e4df',fontweight='bold')
axes[0].set(xlabel='显示 x（像素）',ylabel='显示 y（向下，像素）',title='实空间：第 85 行的单像素线')
im=axes[1].pcolormesh(bin_edges(s),bin_edges(E),signal.T,cmap=NANOX,shading='flat',vmin=0,vmax=float(np.percentile(signal[:,visible_e],99)),rasterized=True)
axes[1].set(xlim=(0,s[-1]),ylim=(-30,30),xlabel='沿 A→B 距离（nm）',ylabel='偏压 V（mV）',title='输入：能量—位置线谱')
fig.colorbar(im,ax=axes[1],label='LI X（pA）',fraction=.045,pad=.035)
im=freq_plot(axes[2],amp);axes[2].set_title('输出：频率—能量图')
fig.colorbar(im,ax=axes[2],label='FFT 幅度（pA）',fraction=.045,pad=.035)
fig.suptitle('L006：A(0,85) → B(97,85) · 98 像素 · 3.919 nm · 各偏压沿空间方向 FFT',fontsize=11)
fig.savefig(OUT/'row85_frequency_energy_overview.png',dpi=180)
fig.savefig(OUT/'row85_frequency_energy_overview.pdf');plt.close(fig)

for name,values,mode,span,full in [('absolute',amp,'absolute',(-30,30),False),('relative',relative,'relative',(-30,30),False),('full',amp,'absolute',(-80,80),True)]:
    fig,ax=plt.subplots(figsize=(7.1,5.9),layout='constrained')
    im=freq_plot(ax,values,mode,span,full)
    ax.set_title('第 85 行 · '+('相对幅度：FFT / 同偏压沿线均值' if mode=='relative' else '单像素线的空间 FFT—偏压图')+'\n去线性背景 + Hann · 原生频点，不补零',fontsize=10)
    fig.colorbar(im,ax=ax,label='相对幅度（无量纲）' if mode=='relative' else 'FFT 幅度（LI X, pA）')
    for fmt in ['png','pdf','svg']:fig.savefig(OUT/f'row85_frequency_energy_{name}.{fmt}',dpi=190)
    plt.close(fig)

readme=f'''# 第 85 行频率—能量图

本次对应用户截图中的 L006，第 85 行。实空间取样为 A(0,85) 到 B(97,85) 的 98 个实测像素，不是整块 dark 区域，也没有沿横向多行平均。显示坐标左上为 (0,0)，原始文件对应 source_row=13、source_col=0..97。线长 {s[-1]:.9f} nm，所有点位于用户指定的 dark/1×1 区域。

## 图怎么读

- row85_frequency_energy_overview.png：左边显示线段位置，中间为输入 g(x,V)，右边为各偏压沿这条线做空间 FFT 后的 A(f,V)。
- row85_frequency_energy_absolute.png：主要幅度图，横轴空间频率 cycles/nm，纵轴偏压 mV，颜色为 LI X 的一维 FFT 幅度 pA。
- row85_frequency_energy_relative.png：补充对照，同一 FFT 幅度除以同偏压原始沿线 LI X 均值。低信号处噪声也会被放大，不能单独用它认定周期。
- row85_frequency_energy_full.png：显示完整原生正频率及 -80..80 mV。

每个偏压有一条 98 点的空间曲线。分别拟合并减去 constant+slope，再减残余均值，乘对称 Hann 窗，做原生 N=98 的 rFFT。正频率双倍幅度，偶数 N 的 Nyquist 不双倍，用 sum(window) 修正幅度。没有沿能量做 FFT、能量平滑、空间插值、补零或沿邻近原子行平均。矩阵方向为 [偏压, 频率]。

频率分辨率 {f[1]:.9f} cycles/nm。a 暂指此前标注拟合得到的横向 Se 列间距 {spacing:.9f} nm。1/(3a)={f3:.9f} cycles/nm，3a={3*spacing:.9f} nm。青色竖线 1/(3a)、2/(3a)、1/a 是参考位置，不是检出的拟合峰；与 Fe 次近邻方向的对应需要原子晶格配准。白色横线是 0 和 10 mV，其中 10 mV 用点线，仅标记关注窗口，不代表能隙边缘。

主图所有偏压共用固定线性色阶，不逐偏压归一化。上限为可见 -30..30 mV、f≤3.5 的原生正频率幅度 P99，仅颜色截断，完整数值不截断。使用 Nanox 配色。相对幅度图也使用其全窗口固定色阶。

## 数据与验证

- row85_FFT_amplitude_pA.csv：81 个偏压 × 50 个原生频率（包含 DC，图中不显示 DC）。首列偏压，后续列频率名；精确频率和周期见 row85_frequency_axis.csv。
- row85_FFT_relative_amplitude.csv：相对幅度矩阵。
- row85_FFT_complex_coefficients.csv：保留原始复数 DFT 的实部、虚部及相位，DC 相位不作解释。
- row85_raw_pixel_spectra_pA.csv、row85_sample_positions.csv：原始线谱与对应逐像素坐标。
- row85_frequency_energy.npz：完整原始及处理数组。
- row85_methods_and_validation.json：来源、坐标、计算与色阶说明。

98×81 原始线谱全部逐点对照 3DS，最大误差 {raw_error:g} pA；偏压轴和原始文件哈希匹配。LI X 从 A 换算 pA，没有绝对 dI/dV 标定。原始文件保持不变。

## 解释边界

这一视野只包含约 {s[-1]/(3*spacing):.2f} 个候选三倍周期。Hann 窗进一步拓宽峰形，不能据此确认严格 3a、非色散、统计显著性或 PDW。0..10 mV 只有六个实测偏压点，并未在本次估计超导能隙。幅度热图不含空间相位，后续应比较正负偏压、相邻同类原子线和更长视野。整个 dark 区域的二维 FFT 与本单线 FFT 属于不同空间取样，不能直接混称。
'''
(OUT/'README_中文.md').write_text(readme,encoding='utf-8')
print(json.dumps(dict(folder=str(OUT),shape=amp.shape,raw_source_error_pA=raw_error,frequency_resolution=float(f[1]),color_limits=limits),ensure_ascii=False))
