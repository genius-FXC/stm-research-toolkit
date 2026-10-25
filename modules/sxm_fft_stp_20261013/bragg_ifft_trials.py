"""Small, reproducible phase-preserving Bragg-filtered iFFT trials.

No source SXM/FFT-STP is modified. Fourier amplitude-only STP is not inverted.
"""
from pathlib import Path
import csv, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from fft_stp_batch import SOURCE, HERE, read_sxm, plane_and_fft, read_stp, digest, GwyddionReader
sys.path.insert(0, str(HERE.parent / 'stm_line_spectrum_tool'))
from core import PALETTE

OUT = HERE / 'ifft_trials' / 'results'
FILES = ['2026_00237.sxm', '2026_00341.sxm', '2026_00370.sxm']
RADII = [.25, .40]
plt.rcParams['font.family'] = 'Microsoft YaHei'
CMAP = ListedColormap(PALETTE / 255.)

def transform(s):
    r=plane_and_fft(s.channels[0].data,*s.scan_range,s.channels[0].unit)
    p=r['fft_shape'][0]; ny,nx=r['source_shape']
    # The prior WSxM-modulus STP uses a reversed native input. Pad in that
    # convention, then reverse both axes to get the displayed real-space grid.
    padded=np.zeros((p,p)); padded[:ny,:nx]=r['residual']*1e12
    displayed=padded[::-1,::-1]
    complex_fft=np.fft.fftshift(np.fft.fft2(displayed))
    assert np.allclose(abs(complex_fft)*r['pixel_area_nm2'],r['canonical_fft'],rtol=1e-10,atol=1e-10)
    return r,displayed,complex_fft

def paired(mask):
    """Exact discrete conjugate partner in FFT-shifted pixel coordinates."""
    n=mask.shape[0]
    unshift=np.fft.ifftshift(mask)
    idx=(-np.arange(n))%n
    sym=np.maximum(unshift,unshift[np.ix_(idx,idx)])
    return np.fft.fftshift(sym)

def make_mask(r, centers, radius):
    x,y=np.meshgrid(r['kx'],r['ky']); m=np.zeros_like(x)
    for c in centers:
        for sign in [-1,1]:
            d=np.hypot(x-sign*c['kx_nm_inv'],y-sign*c['ky_nm_inv'])/radius
            # Unity central disk, cosine rolloff: no hard cutoff ringing.
            w=np.where(d<=.6,1.,np.where(d<1.,.5*(1+np.cos(np.pi*(d-.6)/.4)),0.))
            m=np.maximum(m,w)
    m=paired(m)
    assert m[m.shape[0]//2,m.shape[1]//2]==0
    return m

def inverse(F,mask,shape):
    z=np.fft.ifft2(np.fft.ifftshift(F*mask))
    rel=float(np.max(abs(z.imag))/max(np.max(abs(z.real)),1e-30))
    assert rel<1e-12, rel
    ny,nx=shape
    # Data occupies the bottom-right after reversing the native padded grid.
    return z.real[-ny:,-nx:],rel

def find_centers(r):
    a,x,y=r['canonical_fft'],r['kx'],r['ky']; xx,yy=np.meshgrid(x,y)
    def maximum(allowed,label,pred=None):
        i,j=np.unravel_index(np.argmax(np.where(allowed,a,-1)),a.shape)
        c={'label':label,'kx_nm_inv':float(x[j]),'ky_nm_inv':float(y[i]),'fft_x_0based':int(j),'fft_y_display_0based':int(i)}
        if pred is not None: c['basis_prediction_nm_inv']=pred.tolist()
        return c
    ca=maximum((xx>2.2)&(xx<3.5)&(abs(yy)<.7),'(1,0)')
    cb=maximum((yy>2.2)&(yy<3.5)&(abs(xx)<.7),'(0,1)')
    ga=np.array([ca['kx_nm_inv'],ca['ky_nm_inv']]); gb=np.array([cb['kx_nm_inv'],cb['ky_nm_inv']])
    extra=[]
    for v,name in [(ga+gb,'(1,1)'),(-ga+gb,'(-1,1)')]:
        extra.append(maximum(np.hypot(xx-v[0],yy-v[1])<=.25,name,v))
    return [ca,cb]+extra

def peak_metrics(r,centers):
    a=r['canonical_fft'];x,y=np.meshgrid(r['kx'],r['ky']); out=[]
    for c in centers:
        d=np.hypot(x-c['kx_nm_inv'],y-c['ky_nm_inv'])
        i,j=c['fft_y_display_0based'],c['fft_x_0based']
        background=float(np.median(a[(d>.45)&(d<.8)]))
        out.append(dict(label=c['label'],center_modulus=float(a[i,j]),local_peak_modulus=float(a[d<.2].max()),background_ring_median=background,local_peak_to_background_ratio=float(a[d<.2].max()/max(background,1e-30))))
    return out

def write_real_stp(path,display,range_nm,unit,comment):
    ny,nx=display.shape; disk=display[::-1,::-1]
    lo,hi=float(disk.min()),float(disk.max());span=max(hi-lo,1e-30)
    lines=['WSxM file copyright Nanotec Electronica','SxM Image file','Image header size: 99999','','[Control]','',
           '    Angle: 0',f'    X Amplitude: {range_nm[0]:.17g} nm',f'    Y Amplitude: {range_nm[1]:.17g} nm','','[General Info]','',
           '    Image Data Type: double','    Acquisition channel: Bragg iFFT','    Image processes: converted, plane, FFT, iFFT',
           f'    Number of columns: {nx}',f'    Number of rows: {ny}','    X scanning direction: forward','    Y scanning direction: down',
           f'    Z Amplitude: {span:.17g} {unit}','','[Miscellaneous]','',f'    Comments: {comment}',f'    Maximum: {hi:.17g}',f'    Minimum: {lo:.17g}',
           '    Relative Z value: No','    Z Scale Factor: 1','    Z Scale Offset: 0','    Version: 1.0 (December 2003)','','[Header end]','']
    header='\r\n'.join(lines).encode('ascii',errors='backslashreplace')
    header=header.replace(b'99999',f'{len(header):5d}'.encode('ascii'),1)
    with path.open('xb') as f: f.write(header);f.write(np.asarray(disk,dtype='<f8').tobytes())
    disk_read,_=read_stp(path); assert np.array_equal(disk_read,disk)

def validate_real_stp(reader,path,display,range_nm):
    import ctypes as ct
    err=ct.c_void_p(); con=reader.module.gwy_file_load(str(path).encode('utf-8'),4,ct.byref(err));assert con
    try:
        field=reader.core.gwy_container_get_object(con,reader.glib.g_quark_from_string(b'/0/data')); assert field
        ny,nx=display.shape
        assert reader.proc.gwy_data_field_get_xres(field)==nx and reader.proc.gwy_data_field_get_yres(field)==ny
        extent=[reader.proc.gwy_data_field_get_xreal(field),reader.proc.gwy_data_field_get_yreal(field)]
        assert np.allclose(extent,np.array(range_nm)*1e-9,rtol=1e-12,atol=0)
        v=np.ctypeslib.as_array(reader.proc.gwy_data_field_get_data(field),shape=(nx*ny,)).reshape(ny,nx)
        rel=float(np.max(abs(v-display*1e-12))/max(np.max(abs(display*1e-12)),1e-30));assert rel<1e-12
        return dict(native_import=True,orientation_and_units=True,max_relative_error=rel)
    finally: reader.gobject.g_object_unref(con)

def show_map(ax,data,size,unit,title):
    lo,hi=np.percentile(data,[1,99]); bound=max(abs(lo),abs(hi),1e-20)
    im=ax.imshow(data,cmap=CMAP,vmin=-bound,vmax=bound,extent=[0,size[0],0,size[1]],origin='upper',interpolation='nearest')
    ax.set(title=title,xlabel='x（nm）',ylabel='从图像下边缘起 y（nm）')
    plt.colorbar(im,ax=ax,shrink=.8,label=unit)

def show_fft(ax,r,centers,radius,title,group):
    a=r['canonical_fft'];kx,ky=r['kx'],r['ky'];xx,yy=np.meshgrid(kx,ky)
    roi=(np.hypot(xx,yy)>1)&(abs(xx)<4.5)&(abs(yy)<4.5)
    dx=kx[1]-kx[0];dy=ky[0]-ky[1]
    ax.imshow(a,cmap=CMAP,vmin=0,vmax=np.percentile(a[roi],99.5),extent=[kx[0]-dx/2,kx[-1]+dx/2,ky[-1]-dy/2,ky[0]+dy/2])
    for c in centers:
        color='#00eeee' if c['label'] in ['(1,0)','(0,1)'] else '#ff77cc'
        for sign in [-1,1]:
            p=(sign*c['kx_nm_inv'],sign*c['ky_nm_inv']);ax.add_patch(Circle(p,radius,fill=False,color=color,lw=1.2))
        ax.text(c['kx_nm_inv']+.08,c['ky_nm_inv']+.25,c['label'],color=color,fontsize=9)
    ax.set(xlim=(-4.4,4.4),ylim=(-4.4,4.4),title=title,xlabel='kx（1/nm）',ylabel='ky（1/nm）')

def synthetic_check():
    n=64; x,y=np.meshgrid(np.arange(n),np.arange(n))
    wave=np.cos(2*np.pi*7*x/n)+.3*np.sin(2*np.pi*5*y/n)
    F=np.fft.fftshift(np.fft.fft2(wave));m=np.zeros((n,n));m[n//2,n//2+7]=1;m=paired(m)
    actual,imag=inverse(F,m,wave.shape)
    assert np.max(abs(actual-np.cos(2*np.pi*7*x/n)))<1e-12
    return dict(phase_preservation_and_wave_pass_reject_test=True,error=float(np.max(abs(actual-np.cos(2*np.pi*7*x/n)))),imaginary_fraction=imag)

def main():
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError('Refusing to overwrite trial results')
    OUT.mkdir(parents=True,exist_ok=True)
    for sub in ['images','stp','csv','arrays']: (OUT/sub).mkdir()
    reader=GwyddionReader(); checks={'synthetic':synthetic_check(),'samples':[]}
    with (SOURCE/'index_and_bias.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    refs={}; allpoints=[];index=[]; summary=[]
    for filename in FILES:
        row=next(r for r in rows if r['filename']==filename)
        source=SOURCE/filename;assert digest(source)==row['sxm_sha256']
        s=read_sxm(source);r,original,F=transform(s);ny,nx=r['source_shape']; display=original[-ny:,-nx:]
        group=row['source_3ds_title'];size=np.array(s.scan_range)*1e9;unit='pm' if s.channels[0].unit=='m' else 'pA'
        if group not in refs:
            topo=next(t for t in rows if t['source_3ds_title']==group and t['image_kind']=='Topo_Z')
            sr=read_sxm(SOURCE/topo['filename']);rr,_,_=transform(sr)
            refs[group]=dict(reference_topo=topo['filename'],centers=find_centers(rr))
        centers=refs[group]['centers'];meta={'source_sxm':filename,'source_3ds':group,'source_sha256':row['sxm_sha256'],'bias_mV':row['bias_mV'],'channel':s.channels[0].name,'range_nm':size.tolist(),'shape_yx':[ny,nx],'reference':refs[group],'peak_metrics':peak_metrics(r,centers),'radii_nm_inv':RADII,'mask':'disk: unity r<=0.6R; cosine rolloff 0.6R<r<R; zero outside; conjugate symmetric; exclude DC','axes':'original acquisition grid, not laboratory axes; k=1/wavelength','padding':r['padded']}
        full,imag=inverse(F,np.ones(F.shape),r['source_shape']); allpass=float(np.max(abs(full-display)));assert allpass<max(np.max(abs(display)),1.)*1e-12
        sample_check={'source':filename,'allpass_error_display_units':allpass,'exports':[]}
        maps={}; masks={}; arraydata={'original_plane_removed':display,'complex_fft':F,'kx_nm_inv':r['kx'],'ky_nm_inv':r['ky']}
        for radius in RADII:
            for name,cs in [('ab',centers[:2]),('FeXY',centers[2:]),('ab_FeXY',centers)]:
                mask=make_mask(r,cs,radius); filtered,im=inverse(F,mask,r['source_shape']);maps[radius,name]=filtered;masks[radius,name]=mask
                tag=f'R{round(radius*100):03d}';stem=f'iFFT{source.stem}_{name}_{tag}'
                comment=f'Source SXM: {filename}; 3DS: {group}; bias_mV: {row["bias_mV"] or "N/A topo"}; channel: {s.channels[0].name}; plane removed; complex FFT phase retained; selected {name}; R={radius} 1/nm; cosine-edge disk; conjugates included; DC excluded; fixed topo reference={refs[group]["reference_topo"]}; details in metadata JSON; raw unchanged.'
                path=OUT/'stp'/f'{stem}.stp';write_real_stp(path,filtered,size,unit,comment)
                native=validate_real_stp(reader,path,filtered,size)
                np.savetxt(OUT/'csv'/f'{stem}.csv',filtered,delimiter=',',header=f'display top-down rows; unit={unit}; x_columns_0based; source={filename}',comments='# ')
                arraydata[f'{name}_{tag}']=filtered;arraydata[f'mask_{name}_{tag}']=mask
                rms=float(np.sqrt(np.mean(filtered**2)));rowindex=dict(filename=path.name,source_sxm=filename,source_3ds=group,bias_mV=row['bias_mV'],channel=row['channel'],selection=name,radius_nm_inv=radius,unit=unit,rms=rms,sha256=digest(path))
                index.append(rowindex);sample_check['exports'].append(dict(filename=path.name,imaginary_fraction=im,**native))
        np.savez_compressed(OUT/'arrays'/f'{source.stem}_phase_and_reconstruction.npz',**arraydata)
        (OUT/'arrays'/f'{source.stem}_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        title=f'{filename} · {group} · '+('形貌 Z' if row['image_kind']=='Topo_Z' else f'LI X / {float(row["bias_mV"]):+.3f} mV')
        fig,ax=plt.subplots(2,3,figsize=(13.5,9),layout='constrained');fig.suptitle(title+'\n主试验：R=0.25 1/nm；Nanox 色表，实空间各图单独对称对比度',fontsize=13)
        show_map(ax[0,0],display,size,unit,'原图：仅扣二维平面')
        show_fft(ax[0,1],r,centers[:2],.25,'a/b 选点 + 共轭点','ab')
        show_fft(ax[0,2],r,centers[2:],.25,'对角选点 + 共轭点','FeXY')
        for axs,name,title2 in zip(ax[1],['ab','FeXY','ab_FeXY'],['iFFT：a/b 基本 Bragg 点','iFFT：(1,1)、(-1,1) 对角点','iFFT：两组同时保留']): show_map(axs,maps[.25,name],size,unit,title2)
        fig.savefig(OUT/'images'/f'{source.stem}_iFFT_preview.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(2,3,figsize=(13.5,9),layout='constrained');fig.suptitle(title+'\n选区半径比较：同一列两幅图使用相同色阶',fontsize=13)
        for j,name in enumerate(['ab','FeXY','ab_FeXY']):
            bound=max(float(np.percentile(abs(maps[rad,name]),99)) for rad in RADII)
            for i,rad in enumerate(RADII):
                im=ax[i,j].imshow(maps[rad,name],cmap=CMAP,vmin=-bound,vmax=bound,extent=[0,size[0],0,size[1]],origin='upper',interpolation='nearest')
                ax[i,j].set(title=f'{name} · R={rad:.2f} 1/nm',xlabel='x（nm）',ylabel='y（nm）');plt.colorbar(im,ax=ax[i,j],shrink=.8,label=unit)
        fig.savefig(OUT/'images'/f'{source.stem}_radius_comparison.png',dpi=150);plt.close(fig)
        for c in centers:
            for sign in [1,-1]: allpoints.append(dict(source_sxm=filename,reference_topo=refs[group]['reference_topo'],label=c['label'],conjugate=sign==-1,kx_nm_inv=sign*c['kx_nm_inv'],ky_nm_inv=sign*c['ky_nm_inv']))
        assert digest(source)==row['sxm_sha256'];checks['samples'].append(sample_check);summary.append((row,r,display,size,unit,maps))
        print('Completed',filename,'bias',row['bias_mV'],'peak ratios',[round(m['local_peak_to_background_ratio'],2) for m in meta['peak_metrics']],flush=True)
    for name,data in [('output_index.csv',index),('selected_peaks.csv',allpoints)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f: w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(3,4,figsize=(15,11.5),layout='constrained');fig.suptitle('Bragg iFFT 试验总览 · R=0.25 1/nm\n各面板单独对比度；信号幅值请以色条、CSV 与 STP 为准',fontsize=14)
    for ax,(row,r,display,size,unit,maps) in zip(axes,summary):
        show_map(ax[0],display,size,unit,row['filename']+'\n原图（扣平面）')
        for a,name,label in zip(ax[1:],['ab','FeXY','ab_FeXY'],['a/b Bragg','对角 Bragg','两组同时保留']): show_map(a,maps[.25,name],size,unit,label)
    fig.savefig(OUT/'iFFT_summary.png',dpi=145);plt.close(fig)
    write_readme(summary,refs,checks)
    print('Results',OUT,'STP',len(index),flush=True)

def write_readme(summary,refs,checks):
    lines=['# Bragg 选点 iFFT 少量试验','', '本次只对 3 张图试验，原始 SXM 和已有 FFT STP 均未修改。','', '## 打开什么','', '`iFFT_summary.png` 为总览：每行依次为扣平面原图、a/b 点 iFFT、对角点 iFFT、两组同时保留的 iFFT。', '`images/*_iFFT_preview.png` 含原图、具体选区圆圈与重建；`*_radius_comparison.png` 比较两种窗口半径。', '`stp/` 有 18 张真实 WSxM STP：3 张输入 × 3 种选区 × 2 种半径。`csv/` 是同一实空间矩阵，可导入 Origin。','', '## 选图和坐标','']
    for row,r,display,size,unit,maps in summary:
        lines.append(f'- `{row["filename"]}`：`{row["source_3ds_title"]}`；'+('Z 形貌（采集偏压未记录）' if not row['bias_mV'] else f'LI X，偏压 {float(row["bias_mV"]):+.6f} mV')+f'；{size[0]:g} × {size[1]:g} nm。')
    lines+=['','从 17 组形貌预览中选取 136、138，两者基本和对角晶格峰可见；138 的 +69.33333 mV 图在候选三组的偏压比较中，四个峰相对附近背景均突出。此筛选为可视化演示，不是能量依赖性统计。', '按每组形貌识别 a*=(1,0)、b*=(0,1)，分别搜索 kx 正方向和 ky 正方向 2.2–3.5 nm⁻¹、横向偏离不超过 0.7 nm⁻¹ 的离散幅度极大值；随后在 a*+b*、-a*+b* 预测位置 0.25 nm⁻¹ 内取离散极大值。未强制晶格正交或漂移校正。','138 的偏压图沿用 138 形貌的全部选点，未按偏压重新找峰。精确记录见 `selected_peaks.csv` 与 `arrays/*_metadata.json`。','坐标采用扫描网格方向，kx 向右、ky 向上，单位 1/nm（不含 2π）。频率坐标不是实验室坐标；实空间 y=0 在图像下边缘，CSV 第一行是图像最上边缘。','']
    for group,ref in refs.items():
        lines.append(f'### {group}：参考 {ref["reference_topo"]}')
        for c in ref['centers']: lines.append(f'- {c["label"]}：({c["kx_nm_inv"]:.6f}, {c["ky_nm_inv"]:.6f}) nm⁻¹；另保留其负向共轭点。')
        lines.append('')
    lines+=['## 计算方法和半径','', '二维最小二乘平面扣除 → 复数 FFT → 共轭对称频域掩膜 → 复数逆 FFT，取通过虚部检查的实部。保留原始相位，未对 FFT 幅度 STP 做逆变换，未调整相位以强行对齐原子。', '三种掩膜：`ab` 保留 ±(1,0)、±(0,1)；`FeXY` 保留 ±(1,1)、±(-1,1)；`ab_FeXY` 同时保留八个点。均不保留中心 DC 或低频背景。','圆形选区外半径 R=0.25、0.40 nm⁻¹。圆心至 0.6R 权重为 1，0.6R–R 余弦平滑下降，R 外为 0；同组掩膜取最大值而非叠加。平滑边缘减小硬截断的振铃，但仍属于频率滤波。','R 越小，所保留波矢更单一、图案通常更规则，局部变形和振幅包络更易被抹平；R 越大，保留更多峰附近变化和背景。两个半径试验用于判断结果对选区的敏感性。','这些输入均为 128×128，无补零、插值、重采样；输出尺寸、像素与源图严格对应。136 的频率格点间距 0.1 nm⁻¹，138 为 1/12≈0.083333 nm⁻¹。','STP 数据为有正负值的晶格频带贡献，单位 topo 为 pm、LI X 为 pA。无平面/背景加回，不能把重建值当成原始绝对高度或完整 dI/dV。CSV 保留双精度；NPZ 保存复数 FFT、掩膜和结果。','', '## 命名','', '`iFFT2026_00237_ab_R025.stp` = 2026_00237 的 a/b 组，外半径 0.25 nm⁻¹。','`FeXY` 沿用用户对两个对角频带的命名，只表示频域选点类别；它不等于已完成 Fe/Se 化学位点识别。`R040` 表示外半径 0.40 nm⁻¹。','', '## 如何理解结果','', 'a/b 选点重建基本二维晶格贡献；对角选点重建两个对角方向的晶格谐波贡献；联合结果是二者之和。滤波会天然产生周期网格，不能单凭规则图案证明原子物种、PDW 或新周期。Fe/Se 位点需要原始形貌原子配准、晶胞与结构依据；本次尚未做化学赋值。','预览使用 Nanox 色表。总览和主预览各图独立对称色阶以显示空间结构，视觉亮度不能跨图比较；半径比较中每一列采用同一色阶；定量看色条、CSV 和 STP。','边缘周期延拓和窗口选择可能影响 iFFT，尤其边缘处，不宜作为缺陷/新调制证据。','', '## 校验与验证边界','', '合成双频测试验证指定频率保留、其它频率剔除和相位保留；全通掩膜逆变换逐像素重现扣平面输入；每个结果共轭对称、虚部相对最大实部 < 1e-12。','18 张 STP 均通过自读逐位一致性和独立 Gwyddion 原生 WSxM STP 导入模块验证（尺寸、方向、物理范围、单位、数值相对误差 < 1e-12）。源 SXM 处理前后 SHA256 不变。','未逐张在 WSxM GUI 中打开；GUI 调色显示与本预览可不同。精确数值检查见 `validation.json`。','']
    (OUT/'README_中文.md').write_text('\n'.join(lines),encoding='utf-8')

def render_existing():
    """Only redraw previews; do not change any numerical outputs or sources."""
    samples=[]
    for filename in FILES:
        stem=Path(filename).stem
        meta=json.loads((OUT/'arrays'/f'{stem}_metadata.json').read_text(encoding='utf-8'))
        z=np.load(OUT/'arrays'/f'{stem}_phase_and_reconstruction.npz')
        data=z['original_plane_removed'];size=meta['range_nm'];unit='pm' if meta['channel']=='Z' else 'pA'
        r=dict(kx=z['kx_nm_inv'],ky=z['ky_nm_inv'],canonical_fft=abs(z['complex_fft'])*(size[0]/data.shape[1])*(size[1]/data.shape[0]))
        centers=meta['reference']['centers'];maps={(rad,name):z[f'{name}_R{round(rad*100):03d}'] for rad in RADII for name in ['ab','FeXY','ab_FeXY']}
        signal='形貌 Z' if meta['channel']=='Z' else f'LI X / {float(meta["bias_mV"]):+.3f} mV'
        title=f'{filename} · {meta["source_3ds"]} · {signal}'
        fig,ax=plt.subplots(2,3,figsize=(13.5,9),layout='constrained');fig.suptitle(title+'\n主试验：R=0.25 1/nm；Nanox 色表，实空间各图单独对称对比度',fontsize=13)
        show_map(ax[0,0],data,size,unit,'原图：仅扣二维平面')
        show_fft(ax[0,1],r,centers[:2],.25,'a/b 选点 + 共轭点','ab')
        show_fft(ax[0,2],r,centers[2:],.25,'对角选点 + 共轭点','FeXY')
        for axs,name,t in zip(ax[1],['ab','FeXY','ab_FeXY'],['iFFT：a/b 基本 Bragg 点','iFFT：(1,1)、(-1,1) 对角点','iFFT：两组同时保留']): show_map(axs,maps[.25,name],size,unit,t)
        fig.savefig(OUT/'images'/f'{stem}_iFFT_preview.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(2,3,figsize=(13.5,9),layout='constrained');fig.suptitle(title+'\n选区半径比较：同一列两幅图使用相同色阶',fontsize=13)
        for j,name in enumerate(['ab','FeXY','ab_FeXY']):
            bound=max(float(np.percentile(abs(maps[rad,name]),99)) for rad in RADII)
            for i,rad in enumerate(RADII):
                im=ax[i,j].imshow(maps[rad,name],cmap=CMAP,vmin=-bound,vmax=bound,extent=[0,size[0],0,size[1]],origin='upper',interpolation='nearest')
                ax[i,j].set(title=f'{name} · R={rad:.2f} 1/nm',xlabel='x（nm）',ylabel='y（nm）');plt.colorbar(im,ax=ax[i,j],shrink=.8,label=unit)
        fig.savefig(OUT/'images'/f'{stem}_radius_comparison.png',dpi=150);plt.close(fig)
        samples.append((filename,signal,r,data,size,unit,maps))
    fig,axes=plt.subplots(3,4,figsize=(15,11.5),layout='constrained');fig.suptitle('Bragg iFFT 试验总览 · R=0.25 1/nm\n各面板单独对比度；信号幅值请以色条、CSV 与 STP 为准',fontsize=14)
    for ax,(filename,signal,r,data,size,unit,maps) in zip(axes,samples):
        show_map(ax[0],data,size,unit,filename+'\n'+signal+' · 扣平面')
        for a,name,label in zip(ax[1:],['ab','FeXY','ab_FeXY'],['a/b Bragg','对角 Bragg','两组同时保留']): show_map(a,maps[.25,name],size,unit,label)
    fig.savefig(OUT/'iFFT_summary.png',dpi=145);plt.close(fig)
    print('Previews redrawn; numerical data unchanged')

if __name__=='__main__':
    if '--render-only' in sys.argv: render_existing()
    else: main()
