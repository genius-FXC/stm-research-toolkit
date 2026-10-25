"""Spatial FFT at every measured bias for six explicitly selected pixel rows.

Primary average = arithmetic mean of one-sided amplitudes, not a coherent FFT.
All source data remain read-only. Each row uses x=0..98 (99 measured pixels).
"""
from pathlib import Path
import sys, json, csv, shutil, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core import GridDataset, bin_edges
from profile_fft import profile_fft
from plotting import NANOX
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
ROWS=np.array([67,73,79,85,91,97],dtype=int)
SOURCE=Path(__import__('os').environ['STM_DATA_FILE'])
REGIONS=Path(__import__('os').environ['STM_REGION_DEFINITIONS'])
QUALITY=REGIONS.parent.parent/'00_quality_and_methods/data_quality_and_coordinate_check.json'
d=GridDataset(SOURCE);assert (d.nx,d.ny,len(d.bias))==(99,99,81)
E=d.bias;signals=d.g[ROWS,:,:].copy();step=float(np.linalg.norm(d.transform[1]));s=np.arange(d.nx)*step
coords=np.array([np.column_stack([np.arange(d.nx),np.full(d.nx,r)]) for r in ROWS])
lab=np.array([d.lab(xy) for xy in coords]);direction=d.transform[1]/step
distances=(lab-lab[:,0:1,:])@direction
assert np.allclose(distances,s[None,:],rtol=1e-8,atol=1e-9)

# Independent direct payload check, including row reversal, channel and bias order.
blob=SOURCE.read_bytes();marker=b':HEADER_END:';pos=blob.index(marker)+len(marker)
for ending in (b'\r\r\n',b'\r\n',b'\n'):
    if blob[pos:pos+len(ending)]==ending:pos+=len(ending);break
else:raise ValueError('Unsupported source terminator')
h=d.header;np0=int(h['# Parameters (4 byte)']);pts=int(h['Points']);channels=h['Channels'].split(';')
params=h['Fixed parameters'].split(';')+h['Experiment parameters'].split(';')
records=np.frombuffer(blob,dtype='>f4',offset=pos).reshape(d.ny,d.nx,np0+len(channels)*pts)
order=np.argsort(np.linspace(float(records[0,0,params.index('Sweep Start')]),float(records[0,0,params.index('Sweep End')]),pts))
ci=channels.index('LI Demod 1 X (A)');direct=records[d.ny-1-ROWS,:,np0+ci*pts:np0+(ci+1)*pts].astype(np.float64)[:,:,order]*1e12
raw_error=float(np.max(abs(direct-signals)));assert raw_error==0
assert hashlib.sha256(blob).hexdigest()==d.hash

design=np.column_stack([np.ones(d.nx),s]);trends=np.array([design@np.linalg.lstsq(design,g,rcond=None)[0] for g in signals]);processed=signals-trends
all_fft=[[profile_fft(s,g[:,k],'hann') for k in range(len(E))] for g in processed]
f=np.asarray(all_fft[0][0]['frequency_cycles_per_nm'])
assert all(not z['resampled'] and z['zero_padding']==0 and np.array_equal(z['frequency_cycles_per_nm'],f) for row in all_fft for z in row)
amp=np.array([[z['amplitude'] for z in row] for row in all_fft]) # [row, bias, frequency]
coeff=np.array([[np.asarray(z['coefficient_real'])+1j*np.asarray(z['coefficient_imag']) for z in row] for row in all_fft])
means=signals.mean(axis=1);assert np.all(means>0)
relative=amp/means[:,:,None]
mean6=amp.mean(axis=0);dark5=amp[1:].mean(axis=0);sd6=amp.std(axis=0,ddof=1)
meanrel6=relative.mean(axis=0);meanpower6=np.mean(amp**2,axis=0)
average_error=float(np.max(abs(mean6-sum(amp)/6)))
dark_average_error=float(np.max(abs(dark5-sum(amp[1:])/5)))
assert average_error<1e-12 and dark_average_error<1e-12
reverse_error=max(float(np.max(abs(np.asarray(profile_fft(s,g[::-1,k],'hann')['amplitude'])-amp[j,k]))) for j,g in enumerate(processed) for k in [0,40,45,80]);assert reverse_error<1e-10

regions=json.loads(REGIONS.read_text(encoding='utf-8'));cal=json.loads(QUALITY.read_text(encoding='utf-8'))
fb=1/cal['Se_column_spacing_nm'];f3=fb/3
visible_e=(E>=-30)&(E<=30);visible_f=(f>0)&(f<=3.5)
limits=[0.,float(np.percentile(amp[:,visible_e,:][:,:,visible_f],99))]
relative_limits=[0.,float(np.percentile(relative[:,visible_e,:][:,:,visible_f],99))]
full_limits=[0.,float(np.percentile(amp[:,:,1:],99))]
series={**{f'R{r:03d}':amp[j] for j,r in enumerate(ROWS)},'Mean6':mean6,'Dark5':dark5}
titles={**{f'R{r:03d}':f'第 {r} 行'+('（过渡区）' if r==67 else '') for r in ROWS},'Mean6':'六行 FFT 幅度平均','Dark5':'五行平均（73–97，排除 67）'}
np.savez_compressed(OUT/'six_rows_frequency_energy.npz',pixel_rows=ROWS,bias_mV=E,frequency_cycles_per_nm=f,distance_nm=s,display_xy_px=coords,recorded_lab_xy_nm=lab,raw_signal_pA=signals,linear_background_pA=trends,processed_signal_pA=processed,fft_amplitude_pA=amp,fft_complex_real=coeff.real,fft_complex_imag=coeff.imag,fft_relative_amplitude=relative,line_mean_pA=means,mean6_amplitude_pA=mean6,dark5_amplitude_pA=dark5,row_sd6_amplitude_pA=sd6,mean6_relative_amplitude=meanrel6,mean6_power_pA2=meanpower6,topo_plane_subtracted_pm=d.topo,fft_window=np.hanning(d.nx),q_bragg_cycles_per_nm=fb,q_third_cycles_per_nm=f3)

def csvfile(name,headers,values):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as out:
        writer=csv.writer(out);writer.writerow(headers);writer.writerows(values)

for key,values in series.items():csvfile(f'{key}_FFT_amplitude_pA.csv',['bias_mV',*[f'f_{x:.12g}_cycles_per_nm' for x in f]],np.column_stack([E,values]))
for key,values in [('Mean6_relative',meanrel6),('RowSD6_pA',sd6),('Mean6_power_pA2',meanpower6)]:csvfile(f'{key}.csv',['bias_mV',*[f'f_{x:.12g}_cycles_per_nm' for x in f]],np.column_stack([E,values]))
for j,r in enumerate(ROWS):
    csvfile(f'R{r:03d}_raw_spectra_pA.csv',['bias_mV',*[f'R{r:03d}_S{k+1:04d}' for k in range(d.nx)]],np.column_stack([E,signals[j].T]))
csvfile('frequency_axis.csv',['frequency_index','frequency_cycles_per_nm','period_nm','frequency_over_horizontal_Bragg'],[[i,v,'' if v==0 else 1/v,v/fb] for i,v in enumerate(f)])
csvfile('sample_positions.csv',['row_label','sample_name','distance_nm','display_x_px','display_y_px','source_col','source_row','recorded_x_nm','recorded_y_nm','prior_region'],[[f'R{r:03d}',f'R{r:03d}_S{k+1:04d}',s[k],k,r,k,d.ny-1-r,*lab[j,k],('Transition' if r==67 else 'R1_dark')] for j,r in enumerate(ROWS) for k in range(d.nx)])
csvfile('complex_FFT_long.csv',['display_row','bias_mV','frequency_cycles_per_nm','raw_DFT_real_pA','raw_DFT_imag_pA','amplitude_pA','phase_rad'],[[r,e,x,c.real,c.imag,v,float(np.angle(c))] for j,r in enumerate(ROWS) for e,cs,amps in zip(E,coeff[j],amp[j]) for x,c,v in zip(f,cs,amps)])
shutil.copyfile(ROOT/'research_quicklooks/20261011_dark_row85/frequency_energy/Nanox.pal',OUT/'Nanox.pal')

meta=dict(source_file=str(SOURCE),source_sha256=d.hash,display_pixel_rows=ROWS.tolist(),coordinate_origin='display top-left (0,0), zero-based; y increases down; source_row=98-display_row',display_x_range=[0,98],samples_per_row=d.nx,line_length_nm=float(s[-1]),pixel_step_nm=step,frequency_spacing_cycles_per_nm=float(f[1]),frequency_count_including_DC=len(f),bias_count=len(E),bias_range_mV=[float(E[0]),float(E[-1])],bias_step_mV=float(E[1]-E[0]),row_membership={'67':'Transition','73':'R1_dark','79':'R1_dark','85':'R1_dark','91':'R1_dark','97':'R1_dark'},region_source=str(REGIONS),main_average_rows=ROWS.tolist(),main_average='unweighted arithmetic mean of six per-row one-sided FFT amplitude matrices; each row FFT first, then amplitude mean',dark_only_average_rows=ROWS[1:].tolist(),dark_only_average='same calculation excluding prior transition row 67',row_sd='sample SD of six amplitude matrices, ddof=1; describes row heterogeneity, NOT measurement uncertainty or standard error',supplementary_power_average='arithmetic mean of amplitude**2, pA^2; not spectral-density-normalized PSD',relative_average='first divide each row amplitude by that row raw mean LI X at the same bias; then mean across six rows',fft_axis='spatial x per measured bias, not energy',fft_processing='physical-distance least-squares constant+slope removal; residual mean removal; symmetric Hann; native N=99 rFFT; one-sided coherent-gain-corrected amplitude',amplitude_definition='abs(rfft((processed-mean)*Hann))/sum(Hann); double all positive bins for odd N=99',zero_padding=0,spatial_interpolation=False,energy_interpolation=False,DC_shown_in_plots=False,absolute_color_limits_pA=limits,relative_color_limits=relative_limits,full_color_limits_pA=full_limits,color_limits_policy='all individual and averaged absolute maps share fixed linear scale, P99 of pooled six-row native bins in -30..30 mV and 0<f<=3.5; color clipping does not change values',reference_horizontal_atomic_spacing_nm=cal['Se_column_spacing_nm'],reference_Bragg_frequency_cycles_per_nm=fb,candidate_third_frequency_cycles_per_nm=f3,candidate_third_period_nm=3/fb,reference_lines='1/(3a), 2/(3a), 1/a: reference guides, not detected or fitted peaks',comparison_with_previous_row85='Previously x=0..97, N=98; now all rows including 85 use x=0..98, N=99. Native df changes from about .25255 to .25 cycles/nm; previous files retained.',raw_LI_X_units='pA converted from A, not absolutely calibrated dI/dV',limitations=['Six-line average is not an average of the full 2D dark region.','Row 67 is in the prior transition mask, so the six-line mean is not purely dark.','Magnitude mean cannot establish common spatial phase. Complex coefficients are retained separately.','Averaging does not improve native frequency resolution.','0..10 mV has only six measured biases and is not an estimated gap boundary.','No claim of exact 3a, significance, non-dispersion, or PDW.'],validation=dict(direct_binary_spectra_max_error_pA=raw_error,amplitude_average_max_error_pA=average_error,dark5_average_max_error_pA=dark_average_error,reversal_amplitude_max_error_pA=reverse_error,common_frequency_axis_verified=True,source_hash_verified=True))
(OUT/'methods_and_validation.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')

def freqmap(ax,values,scale=limits,span=(-30,30),full=False,labels=True):
    im=ax.pcolormesh(bin_edges(f[1:]),bin_edges(E),values[:,1:],cmap=NANOX,shading='flat',vmin=scale[0],vmax=scale[1],rasterized=True)
    ax.set(xlim=(0,float(f[-1]) if full else 3.5),ylim=span,xlabel='空间频率 f（cycles/nm）',ylabel='偏压（mV）')
    for x,label in [(f3,'1/(3a)'),(2*f3,'2/(3a)'),(fb,'1/a')]:
        ax.axvline(x,color='#00c9c9',ls='--',lw=.8)
        if labels:ax.text(x+.03,span[1]-.06*(span[1]-span[0]),label,color='#00c9c9',fontsize=8,bbox=dict(fc='black',ec='none',alpha=.55,pad=1))
    ax.axhline(0,color='white',lw=.55,alpha=.55);ax.axhline(10,color='white',ls=':',lw=.65,alpha=.7)
    return im

fig,axes=plt.subplots(2,4,figsize=(15.4,8.7),layout='constrained',sharex=True,sharey=True)
for ax,(key,values) in zip(axes.flat,series.items()):
    im=freqmap(ax,values);ax.set_title(titles[key],fontsize=11)
fig.colorbar(im,ax=axes.ravel().tolist(),label='FFT 幅度（LI X, pA）',fraction=.025,pad=.02)
fig.suptitle('六条单像素线的频率—能量图及幅度平均 · 统一 Nanox 色阶\n每行 99 像素（x=0–98）· 去线性背景 + Hann · Δf≈0.250 cycles/nm',fontsize=12)
for fmt in ['png','pdf']:fig.savefig(OUT/f'six_rows_frequency_energy_overview.{fmt}',dpi=170)
plt.close(fig)

for key,values in series.items():
    fig,ax=plt.subplots(figsize=(7,5.8),layout='constrained');im=freqmap(ax,values);ax.set_title(titles[key]+' · 空间 FFT—偏压\n全宽 x=0–98 · 去线性背景 + Hann',fontsize=10)
    fig.colorbar(im,ax=ax,label='FFT 幅度（LI X, pA）')
    for fmt in ['png','pdf','svg']:fig.savefig(OUT/f'{key}_frequency_energy.{fmt}',dpi=180)
    plt.close(fig)
for key,values,scale,span,full,title in [('Mean6_relative',meanrel6,relative_limits,(-30,30),False,'六行相对幅度平均（低信号处可能放大噪声）'),('Mean6_full',mean6,full_limits,(-80,80),True,'六行幅度平均 · 完整偏压与正频率')]:
    fig,ax=plt.subplots(figsize=(7,5.8),layout='constrained');im=freqmap(ax,values,scale,span,full);ax.set_title(title,fontsize=10);fig.colorbar(im,ax=ax,label='相对幅度（无量纲）' if 'relative' in key else 'FFT 幅度（pA）')
    for fmt in ['png','pdf']:fig.savefig(OUT/f'{key}_frequency_energy.{fmt}',dpi=180)
    plt.close(fig)
fig,ax=plt.subplots(figsize=(6.5,6.1),layout='constrained');lo,hi=np.percentile(d.topo,[1,99]);ax.imshow(d.topo,cmap=NANOX,vmin=lo,vmax=hi,origin='upper',extent=(-.5,98.5,98.5,-.5),interpolation='nearest')
ax.axhspan(65.5,68.5,color='#ffab40',alpha=.25)
for r in ROWS:
    c='#ffb143' if r==67 else '#00e4df';ax.plot([0,98],[r,r],color=c,lw=1.05);ax.text(99.5,r,str(r),color=c,va='center',fontsize=9)
ax.set(xlim=(-.5,107),ylim=(99,-.5),xlabel='显示 x（像素）',ylabel='显示 y（向下，像素）',title='六条取样线：67、73、79、85、91、97\n橙色：此前过渡区；青色：此前 dark 区域')
for fmt in ['png','pdf','svg']:fig.savefig(OUT/f'six_rows_topography_paths.{fmt}',dpi=180)
plt.close(fig)

# Native-bin energy curves retain individual rows to reveal row dominance.
js=[np.flatnonzero(f<f3)[-1],np.flatnonzero(f>f3)[0]]
headers=['bias_mV']
blocks=[E]
for j in js:
    for r,values in zip(ROWS,amp):headers.append(f'R{r:03d}_f{f[j]:.9f}_pA');blocks.append(values[:,j])
    for label,values in [('Mean6',mean6),('Dark5',dark5),('RowSD6',sd6)]:headers.append(f'{label}_f{f[j]:.9f}_pA');blocks.append(values[:,j])
csvfile('candidate_adjacent_frequency_energy_curves.csv',headers,np.column_stack(blocks))
fig,axs=plt.subplots(1,2,figsize=(11.7,4.1),layout='constrained')
for ax,j in zip(axs,js):
    for r,values in zip(ROWS,amp):ax.plot(E,values[:,j],lw=.85,alpha=.6,label=f'{r}')
    ax.plot(E,mean6[:,j],c='black',lw=2,label='六行平均');ax.plot(E,dark5[:,j],c='#db6a28',lw=1.8,ls='--',label='五行 dark 平均')
    ax.axvspan(0,10,color='#d9b56e',alpha=.15);ax.set(xlim=(-30,30),ylim=(0,None),xlabel='偏压（mV）',ylabel='FFT 幅度（pA）',title=f'参考 1/(3a) 两侧频点：f={f[j]:.3f} cycles/nm');ax.grid(alpha=.12)
axs[1].legend(ncol=2,fontsize=8)
for fmt in ['png','pdf']:fig.savefig(OUT/f'candidate_frequency_row_comparison.{fmt}',dpi=180)
plt.close(fig)

readme=f'''# 指定六行的频率—能量图及平均

## 实空间选择

用户指定显示像素行 67、73、79、85、91、97，左上原点 (0,0)，y 向下，从此前网页像素坐标沿用零基编号；不是原子行编号。每行都取 x=0..98，共 99 个实测像素，线长 {s[-1]:.9f} nm，单像素宽，不对相邻未指定行取平均。来源是 {SOURCE.name}，81 个实测偏压 -80..80 mV、步长 2 mV。

依据已确认的区域 mask，67 为过渡区（66–68），其余五行为 dark/1×1（69–98）。主结果保留用户指定的六行；另给 Dark5（73、79、85、91、97）作为排除 67 的敏感性对照。六行不是整块 dark 的二维空间平均，也不是仅 Fe 原子点的平均。

本次每行使用统一完整宽度。上一轮第 85 行使用 x=0..97、98 点；本次第 85 行重新取 x=0..98、99 点。因此频率间隔从约 0.25255 改为 {f[1]:.9f} cycles/nm；旧文件保留，不能直接按原矩阵列逐点相加。

## 计算与平均定义

每行、每个偏压分别沿物理距离拟合并减去常数与线性背景，再减残余均值，乘对称 Hann，做 N=99 的空间 rFFT。单边幅度 = abs(DFT)/sum(Hann)，所有正频率双倍（N 为奇数）。零频表中保留、图中排除。没有补零、空间/能量插值、能量平滑、逐能量颜色自动归一化。平均不会提高原生频率分辨率。

**主平均 Mean6 = 六行分别做 FFT 后，单边幅度逐频率、逐偏压等权算术平均。**不是先对六行实空间信号取平均后再 FFT，也不是复数系数平均；各行相位不同不会在主平均中互相抵消，但主平均也不能证明各行同相。Dark5 使用相同算法、只取后五行。CSV/NPZ 保存每行复数系数供后续相位分析。

RowSD6 为六行幅度的样本标准差（ddof=1），描述行间差异，不称测量误差、标准误或统计置信区间。补充 Mean6_power_pA2 是 mean(A_i^2)，没有做谱密度归一化，不称 PSD。相对幅度平均为每行幅度除以该行同偏压原始沿线 LI X 均值后，再对六行平均；低信号时噪声也会被放大。

## 图和数据

- six_rows_frequency_energy_overview.png/pdf：六行与两种平均，共八幅图，使用共同色阶。
- R067/R073/R079/R085/R091/R097_frequency_energy：各行的单独 PNG/PDF/SVG。
- Mean6_frequency_energy：用户指定六行的主平均；Dark5_frequency_energy：排除过渡行的五行平均。
- Mean6_relative_frequency_energy：相对幅度补充；Mean6_full_frequency_energy：完整频率与偏压。
- six_rows_topography_paths：原始形貌上六条线的实际位置，67 用橙色，其他用青色。
- candidate_frequency_row_comparison：参考 1/(3a) 左右原生频点的逐行及平均能量曲线，便于查看某行是否主导。
- 各行/平均 FFT_amplitude_pA.csv：首列偏压，其余 50 列为原生频率（含 DC）。精确频率见 frequency_axis.csv。
- 各行 raw_spectra_pA.csv、sample_positions.csv：每条谱对应的实测像素、文件行/列和真实坐标。
- six_rows_frequency_energy.npz：完整原始信号、处理背景、单行幅度、复数系数、两种主平均、相对幅度、行间 SD 和平均功率。
- methods_and_validation.json：来源哈希、区域归属、算法、色阶和检查值。

横轴为 cycles/nm，纵轴为偏压 mV，颜色为 LI X 的 FFT 幅度 pA（没有绝对 dI/dV 标定）。a 暂指此前拟合的横向 Se 列间距 {cal['Se_column_spacing_nm']:.9f} nm，青色竖线 1/(3a)={f3:.9f}、2/(3a)、1/a={fb:.9f} 是参考导引，不能当作已拟合出的峰。对应 Fe 次近邻方向需原子配准。白线 0、10 mV 仅指关注窗口，没有估计能隙。

统一 Nanox 色阶：各行及两个主平均均为 0..{limits[1]:.9f} pA，上限取六行合并的 -30..30 mV、0<f≤3.5 原生频点 P99；颜色截断不修改数据。相对图使用自己的固定相对色阶。所有完整矩阵保留 -80..80 mV 和全部原生频率。

## 验证与解释边界

独立直接读取 3DS 二进制记录，6×99×81 个谱值最大误差 {raw_error:g} pA；行翻转、通道与偏压排序验证通过。主幅度平均算术误差 {average_error:g} pA，dark5 平均误差 {dark_average_error:g} pA；线反转幅度最大误差 {reverse_error:.3g} pA。原始 3DS 未修改。

这里只有六条指定单像素线，每条容纳约 {s[-1]/(3/fb):.2f} 个候选三倍周期；Hann 使峰展宽。不能用平均图单独确认严格 3a、非色散、统计显著性或 PDW。67 行的过渡区归属只来自之前 mask；本次未按图像亮暗重新分区。0..10 mV 只有六个实测能量点。尚未把六行配准到相同的 Fe/Se 原子相位。
'''
(OUT/'README_中文.md').write_text(readme,encoding='utf-8')
final_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest();assert final_hash==d.hash
meta['validation']['source_hash_unchanged_after_analysis']=True
(OUT/'methods_and_validation.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(folder=str(OUT),rows=ROWS.tolist(),shape=amp.shape,line_length_nm=float(s[-1]),df=float(f[1]),common_scale=limits,validation=meta['validation']),ensure_ascii=False))
