from pathlib import Path
import csv,json,struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap,Normalize
from core import PALETTE,DISTANCE_STOPS,settings_json,bin_edges

plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'font.size':9})
NANOX=ListedColormap(PALETTE/255)
DISTANCE_CMAP=LinearSegmentedColormap.from_list('line_distance',DISTANCE_STOPS/255,N=256)

def csv_matrix(path,headers,values):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(values)

def draw_path(ax,data,result):
    cfg=result['config'];a,label=data.map_values(cfg['view'],cfg['map_bias_mV'],cfg['topo_plane']);lo,hi=np.percentile(a,[1,99])
    turns=cfg.get('rotation_quarters',0);a=np.rot90(a,-turns);ny,nx=a.shape
    from rotation import rotate_points
    xy=rotate_points(result['xy'],data.nx,data.ny,turns)
    im=ax.imshow(a,origin='upper',extent=(-.5,nx-.5,ny-.5,-.5),cmap=NANOX,vmin=lo,vmax=hi,interpolation='nearest')
    ax.plot(xy[:,0],xy[:,1],c='#00ddd8',lw=1.3)
    for index,text in [(0,'A'),(-1,'B')]:
        ax.scatter(*xy[index],s=24,c='#00ddd8',edgecolors='white',linewidths=.7)
        ax.annotate(text,xy[index],xytext=(5,-10),textcoords='offset points',color='#00b9b6',fontweight='bold',bbox=dict(fc='white',ec='none',alpha=.85,pad=1))
    ax.set(xlim=(-.5,nx-.5),ylim=(ny-.5,-.5),xlabel='显示 x（像素）',ylabel='显示 y（向下，像素）',title=f"{result['line_id']} · {label}")
    return im

def draw_heatmap(ax,result):
    mask=result['energy_mask'];E=result['bias_mV'][mask];s=result['s_nm'];values=result['signal_pA'][:,mask].T
    lo,hi=result['heat_limits_pA']
    im=ax.pcolormesh(result['position_edges_nm'],bin_edges(E),values,shading='flat',cmap=NANOX,vmin=lo,vmax=hi,rasterized=True)
    ax.set(xlim=(0,s[-1]),ylim=(result['config']['energy_min_mV'],result['config']['energy_max_mV']),xlabel='沿 A→B 距离（nm）',ylabel='偏压（mV）',title='能量—位置线谱 · LI X（pA）')
    return im

def draw_height(ax,result):
    y=result['profile_values'];cfg=result['config']
    ax.plot(result['s_nm'],y,c='#a56b2a',lw=1.1)
    ax.grid(alpha=.12)
    mode={'raw':'原始值','subtract_average':'减均值','linear_fit':'去线性背景'}[cfg['profile_mode']]
    channel='Z' if result['profile_channel']=='Z' else f"LI X · {result['profile_bias_mV']:g} mV"
    ax.set(xlim=(0,result['length_nm']),xlabel='沿 A→B 投影距离（nm）',ylabel=f"{result['profile_channel']}（{result['profile_unit']}）",title=f'沿线剖面 · {channel} · {mode}')
    ax.ticklabel_format(axis='y',style='plain',useOffset=True)

def draw_waterfall(ax,result):
    mask=result['waterfall_energy_mask'];E=result['waterfall_bias_mV'][mask];cfg=result['config']
    for j,(index,y) in enumerate(zip(result['waterfall_indices'],result['waterfall'])):
        color=result['waterfall_colors_hex'][j]
        ax.plot(E,y[mask],c=color,lw=.9)
        right=int(np.argmax(E))
        ax.text(E[right],y[mask][right],f' {result["sample_names"][index].split("_")[-1]}',fontsize=6,color=color,va='center')
    unit='LI X（'+result['processing']['unit']+'）'
    if cfg['zero_anchor']:unit+=f" − g({result['zero_anchor_bias_mV']:g}mV)"
    unit+=f" + {cfg['offset']:g}×序号"
    ax.set(xlim=(cfg['energy_min_mV'],cfg['energy_max_mV']),xlabel='偏压（mV）',ylabel=unit,title=f'瀑布图 · {len(result["waterfall_indices"])} 条 / {result["n_samples"]} 点')
    ax.grid(alpha=.12);ax.axvline(0,c='.6',lw=.6,ls=':')
    length=result['length_nm']
    bar=ax.figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,length),cmap=DISTANCE_CMAP),ax=ax,orientation='horizontal',fraction=.035,pad=.15,aspect=35,ticks=[0,length/2,length])
    bar.ax.set_xticklabels(['A · 0',f'{length/2:.3f}',f'B · {length:.3f}'])
    bar.set_label('沿 A→B 距离（nm）',fontsize=8);bar.ax.tick_params(labelsize=7);bar.outline.set_visible(False)

def draw_profile_fft(ax,result):
    f=result['fft']
    if not f['available']:
        ax.text(.5,.5,f['reason'],ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();return
    freq=np.asarray(f['frequency_cycles_per_nm']);amp=np.asarray(f['amplitude'])
    ax.plot(freq[1:],amp[1:],c='#087e82',lw=1.2)
    bias='' if f['bias_mV'] is None else f" · {f['bias_mV']:g} mV"
    ax.set(xlim=(0,f['nyquist_cycles_per_nm']),ylim=(0,max(float(amp[1:].max())*1.12,1e-12)),xlabel='空间频率 f（cycles/nm）',ylabel=f"FFT 幅度（{f['unit']}）",title=f"沿线一维 FFT · {f['channel']}{bias} · {f['window']}")
    ax.grid(alpha=.12)
    if f['peak']:
        p=f['peak'];ax.scatter(p['frequency_cycles_per_nm'],p['amplitude'],s=12,c='#087e82')
        ax.text(.98,.96,f"最大幅值频点\nf={p['frequency_cycles_per_nm']:.4f} cycles/nm\nλ={p['period_nm']:.4f} nm",ha='right',va='top',fontsize=8,transform=ax.transAxes)

def export_images(data,result,folder,formats=('png',)):
    folder=Path(folder);files=[]
    figures={}
    f=plt.figure(figsize=(15,8.5),layout='constrained');gs=f.add_gridspec(3,3,height_ratios=[1.6,1,1]);ax_path=f.add_subplot(gs[0,0]);ax_z=f.add_subplot(gs[1,0]);ax_fft=f.add_subplot(gs[2,0]);ax_heat=f.add_subplot(gs[:,1]);ax_wf=f.add_subplot(gs[:,2]);draw_path(ax_path,data,result);draw_height(ax_z,result);draw_profile_fft(ax_fft,result);im=draw_heatmap(ax_heat,result);f.colorbar(im,ax=ax_heat,label='LI X（pA）');draw_waterfall(ax_wf,result)
    f.suptitle(f'{data.label} | {result["line_id"]} | {result["length_nm"]:.4f} nm',fontsize=10);figures['overview']=f
    for name,draw in [('path',lambda a:draw_path(a,data,result)),('line_profile',lambda a:draw_height(a,result)),('profile_fft',lambda a:draw_profile_fft(a,result)),('energy_position',lambda a:draw_heatmap(a,result)),('waterfall',lambda a:draw_waterfall(a,result))]:
        f,ax=plt.subplots(figsize=(7,5.6),layout='constrained');im=draw(ax)
        if name=='energy_position':f.colorbar(im,ax=ax,label='LI X（pA）')
        figures[name]=f
    try:
        for name,fig in figures.items():
            for fmt in formats:
                p=folder/f'{result["line_id"]}_{name}.{fmt}';fig.savefig(p,dpi=240,bbox_inches='tight');files.append(p.name)
    finally:
        for fig in figures.values():plt.close(fig)
    return files

def save_numeric_bundle(data,result,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    lid=result['line_id'];bias=result['bias_mV'];names=result['sample_names']
    csv_matrix(folder/f'{lid}_full_spectra_used_pA.csv',['bias_mV',*names],np.column_stack([bias,result['signal_pA'].T]))
    csv_matrix(folder/f'{lid}_full_spectra_center_pA.csv',['bias_mV',*names],np.column_stack([bias,result['center_pA'].T]))
    processed_bias=result['waterfall_bias_mV']
    csv_matrix(folder/f'{lid}_processed_spectra.csv',['bias_mV',*names],np.column_stack([processed_bias,result['processed_spectra'].T]))
    if result['normalized'] is not None:csv_matrix(folder/f'{lid}_full_spectra_normalized.csv',['bias_mV',*names],np.column_stack([processed_bias,result['normalized'].T]))
    if result['current_first_pA'] is not None:
        factors=result['current_correction_factor']
        csv_matrix(folder/f'{lid}_current_correction.csv',['sample_name','first_recorded_bias_mV','first_recorded_current_pA','I_sample_div_I_reference'],[[name,result['processing']['current_first_bias_mV'],v,'' if factors is None else factors[i]] for i,(name,v) in enumerate(zip(names,result['current_first_pA']))])
    wfids=[names[i] for i in result['waterfall_indices']]
    csv_matrix(folder/f'{lid}_waterfall_display.csv',['bias_mV',*wfids],np.column_stack([processed_bias,result['waterfall'].T]))
    csv_matrix(folder/f'{lid}_waterfall_colors.csv',['sample_name','distance_nm','fraction_A_to_B','color_hex'],[[names[i],result['s_nm'][i],result['s_nm'][i]/result['length_nm'],color] for i,color in zip(result['waterfall_indices'],result['waterfall_colors_hex'])])
    coords=np.column_stack([np.arange(1,result['n_samples']+1),result['s_nm'],result['xy'],result['xy'][:,0],data.ny-1-result['xy'][:,1],result['lab_xy'],result['valid_strip_samples']])
    csv_matrix(folder/f'{lid}_sample_positions.csv',['sample_name','sample_order','distance_nm','display_x_px','display_y_px','source_col_zero_based','source_row_zero_based','recorded_x_nm','recorded_y_nm','valid_strip_offsets'],[[name,*r] for name,r in zip(names,coords)])
    zkeys=['z_raw_center_pm','z_plane_center_pm','z_raw_used_pm','z_plane_used_pm','z_display_relative_pm']
    csv_matrix(folder/f'{lid}_z_profile.csv',['sample_name','distance_nm',*zkeys],[[name,s,*z] for name,s,z in zip(names,result['s_nm'],np.column_stack([result[k] for k in zkeys]))])
    unit=result['profile_unit']
    csv_matrix(folder/f'{lid}_line_profile.csv',['sample_name','distance_nm','display_x_px','display_y_px',f'raw_{unit}',f'fitted_background_{unit}',f'processed_{unit}'],[[name,s,*xy,raw,trend,used] for name,s,xy,raw,trend,used in zip(names,result['s_nm'],result['xy'],result['profile_raw'],result['profile_trend'],result['profile_values'])])
    meta=settings_json(data,result)
    fft=result['fft'];fft_arrays={}
    if fft['available']:
        csv_matrix(folder/f'{lid}_profile_fft.csv',['frequency_cycles_per_nm','period_nm',f'amplitude_{unit}',f'raw_DFT_real_{unit}',f'raw_DFT_imag_{unit}'],zip(fft['frequency_cycles_per_nm'],['' if v is None else v for v in fft['period_nm']],fft['amplitude'],fft['coefficient_real'],fft['coefficient_imag']))
        csv_matrix(folder/f'{lid}_fft_input.csv',['distance_nm',f'uniform_profile_{unit}',f'demeaned_profile_{unit}','window_weight',f'windowed_input_{unit}'],zip(fft['uniform_distance_nm'],fft['uniform_profile'],fft['demeaned_profile'],fft['window_weights'],np.asarray(fft['demeaned_profile'])*fft['window_weights']))
        array_keys=['uniform_distance_nm','uniform_profile','demeaned_profile','window_weights','frequency_cycles_per_nm','period_nm','amplitude','coefficient_real','coefficient_imag']
        fft_arrays={f'fft_{k}':np.asarray(fft[k]) for k in array_keys if k!='period_nm'}
        meta['fft']={k:v for k,v in fft.items() if k not in array_keys}
    else:meta['fft']=fft
    (folder/f'{lid}_fft_methods.json').write_text(json.dumps(meta['fft'],ensure_ascii=False,indent=2),encoding='utf-8')
    meta['origin_energy_axis']='0-based pixel-sample index; exact projected distance for each pixel is in Coordinates (digital pixel paths have non-uniform distances). Browser/images use exact distance bin edges.'
    (folder/f'{lid}_line.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    A,_=data.map_values(result['config']['view'],result['config']['map_bias_mV'],result['config']['topo_plane'])
    np.savez_compressed(folder/f'{lid}_numeric.npz',bias_mV=bias,s_nm=result['s_nm'],signal_pA=result['signal_pA'],center_pA=result['center_pA'],waterfall=result['waterfall'],waterfall_bias_mV=processed_bias,processed_spectra=result['processed_spectra'],waterfall_indices=result['waterfall_indices'],xy=result['xy'],lab_xy=result['lab_xy'],display_map=A,ny=data.ny,nx=data.nx,position_edges_nm=result['position_edges_nm'],profile_raw=result['profile_raw'],profile_trend=result['profile_trend'],profile_values=result['profile_values'],**fft_arrays,**{k:result[k] for k in zkeys})
    payload=struct.pack('<HH',0x300,256)+b''.join(bytes([*r,0]) for r in PALETTE)
    (folder/'Nanox.pal').write_bytes(b'RIFF'+struct.pack('<I',len(payload)+12)+b'PAL '+b'data'+struct.pack('<I',len(payload))+payload)
    (folder/'README_中文.md').write_text(f'''# {lid}：{data.label}

像素 A={result['config']['start']} → B={result['config']['end']}，显示左上像素中心为 (0,0)，x向右、y向下。共 {result['n_samples']} 个唯一实测像素，每像素 {len(bias)} 个实测偏压。完整名称与整数坐标在 sample_positions.csv。路径采用可精确反转的 Bresenham 算法（8邻接）；斜线会选择最接近直线的连续像素，可能包含对角邻接，没有亚像素插值或带宽平均。

沿线距离是每个像素记录的实验 X/Y 坐标在 A→B 直线上的投影，线长 {result['length_nm']:.6f} nm；它不是数字折线路径的弧长。相邻样本物理距离可能不同，浏览器与图片按真实投影距离的中点分箱。Origin 的原生矩阵列为等间隔像素样本序号（0起），避免伪造均匀 nm 轴；Coordinates 表记录每列精确 nm 距离。

full_spectra_used_pA 与 full_spectra_center_pA 都是直接读取的实测像素谱，数值相同。完整谱保留所有偏压；瀑布显示数只从中均匀选取至多 {len(wfids)} 条（包含两端），不新增或平均点谱。完整原始谱另存；processed_spectra 和 full_spectra_normalized 按下述程序处理规则另存，waterfall_display 在处理谱基础上叠加可选零点对齐和 offset。颜色随A→B实际投影距离渐变，waterfall_colors.csv保存每条线的名称、距离与色值。

line_profile.csv 是本次选中通道的剖面：{result['profile_channel']}，偏压 {result['profile_bias_mV']} mV（Z无偏压），单位 {unit}，处理 {result['config']['profile_mode']}。Raw为原始值；Subtract average减沿线均值；Linear fit对投影距离拟合常数+斜率并减掉拟合背景。表内保存原始值、背景及处理值。此处理独立于Topo显示去平面和瀑布归一化。z_profile.csv始终另存同批像素的原始/全局去平面Z；其中relative字段仅为兼容旧格式，不是当前剖面显示。

能量—位置图最暗/最亮阈值为 {result['heat_limits_pA']} pA，来自当前显示偏压范围的P{result['config']['heat_min_percentile']:g} / P{result['config']['heat_max_percentile']:g}。浏览器、图片和Origin使用这组阈值；只截断显示颜色，不截断原始数据。

一维 FFT 分析本次选中通道的处理后剖面，额外减去 FFT 输入均值。窗口为 {result['config']['fft_window']}，采用原生 N 点 rFFT，不补零。等距剖面直接变换；非等距数字像素路径仅为 FFT 沿物理投影距离线性重采样到同样 N 点，原始点谱、剖面不变，插值可能减弱高频幅度。profile_fft.csv 保存空间频率 f（cycles/nm）、周期 λ=1/f、单边幅度及未归一化复系数的实虚部；fft_input.csv 保存均匀距离、FFT 输入和窗口，fft_methods.json 保存算法参数。单边幅度=|rFFT|/sum(window)，正频率乘2，但偶数N的Nyquist点不乘2。频点间隔是1/(N×Δs)，不是1/两端距离；窗口会影响泄漏和峰宽，最大幅值频点不是连续频率的拟合峰。零频的周期为空，预览不显示零频；该空间FFT本身不能确定PDW。

## 谱线处理（与用户程序实际代码一致）

方法：{result['processing']['label']}，处理单位 {result['processing']['unit']}。顺序是：先对完整能量点谱平滑，再做 x=(bias+offset)×factor，按裁切上下限保留点，最后归一化/缩放。

minmax：[g-min(g)]/[max(g)-min(g)]，逐谱、对平滑裁切后的保留点计算；first：除以同一条处理后谱的第一个保留值；current：g_i×y_factor×I_reference(first)/I_i(first)，首点电流来自原始采集顺序的第一个偏压（并非显示最左端或设定电流），参考为沿线指定像素；none：只乘 y_factor。minmax/first 按原程序忽略 y_factor。旧版外侧均值方法仅为兼容旧记录。

平滑 {result['config']['smoothing_method']}，order={result['config']['smoothing_order']}，窗口=2×order+1，高斯 sigma={result['config']['smoothing_sigma']} 个偏压点。开启平滑时丢弃两端各 order 点；近邻均值忠实使用原程序从左到右原位更新逻辑（在独立副本上操作），并非对称一次卷积。高斯默认 sigma=1 与原程序一致，可自行设置。

裁切 {result['processing']['crop_mV']} mV；x_offset={result['config']['x_offset_mV']} mV，x_factor={result['config']['x_factor']}，y_factor={result['config']['y_factor']}。zero_anchor 是工作台额外选项，原程序未启用零点对齐，因此要完全比较原程序数值时关闭此项。空间数字采样仍为实测像素，不改变原始能量—位置图、沿线剖面或剖面 FFT。processed_spectra / waterfall_display 使用处理后的偏压轴，full_spectra_used_pA / center_pA 保留原始全部偏压轴；两者长度可不同。

Line JSON保存源文件身份、整数端点和全部处理参数。numeric NPZ保存数值、路径背景与分箱边界。导出Origin时包含可编辑原生曲线、矩阵、剖面、坐标及源文件信息。LI Demod 1 X由A转pA，未校准成绝对dI/dV或DOS；能量平滑按所选设置执行，原始完整谱与空间剖面保持不变，原始3DS保持不变。旧版记录恢复后端点会吸附到最近像素并使用新采样规则，旧导出文件不会改写。
''',encoding='utf-8')
    return meta
