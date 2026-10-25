from pathlib import Path
import sys,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from core import GridDataset
from profile_fft import profile_fft
BASE=HERE.parent/'20261011_Se_rows070_076_082_088_094/results'
meta=json.loads((BASE/'methods_and_validation.json').read_text(encoding='utf-8'))
a=np.load(BASE/'Se_rows_frequency_energy.npz');d=GridDataset(Path(meta['source_file']));assert d.hash==meta['source_sha256']
OUT=HERE/'Z_line_FFT';OUT.mkdir(exist_ok=True)
rows=a['pixel_rows'];x=a['distance_nm'];raw=d.z[rows].copy(); labels=[f'SeR{11+i}_y{r}' for i,r in enumerate(rows)]
X=np.column_stack([np.ones(len(x)),x]);z=raw-(X@np.linalg.lstsq(X,raw.T,rcond=None)[0]).T
results=[profile_fft(x,t) for t in z];f=np.array(results[0]['frequency_cycles_per_nm']);amp=np.array([t['amplitude'] for t in results])
plain=np.array([profile_fft(x,t,window='none')['amplitude'] for t in z]);mean=amp.mean(axis=0);sd=amp.std(axis=0,ddof=1)
def save(name,header,data):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as h:
        w=csv.writer(h);w.writerow(header);w.writerows(data)
save('Z_raw_pm.csv',['distance_nm',*labels],np.column_stack([x,raw.T]))
save('Z_detrended_pm.csv',['distance_nm',*labels],np.column_stack([x,z.T]))
save('Z_FFT_Hann_pm.csv',['frequency_cycles_per_nm',*labels,'Mean5','Row_SD'],np.column_stack([f,amp.T,mean,sd]))
save('Z_FFT_no_window_pm.csv',['frequency_cycles_per_nm',*labels],np.column_stack([f,plain.T]))
qb=float(a['q_bragg_cycles_per_nm']);q3=float(a['q_third_cycles_per_nm'])
fig,ax=plt.subplots(3,2,figsize=(12,10),layout='constrained');colors=plt.cm.viridis(np.linspace(.1,.85,5))
for i,(r,c) in enumerate(zip(rows,colors)):
    ax[0,0].plot(x,z[i]+i*25,color=c,label=labels[i]);ax[0,1].plot(f,amp[i],color=c,label=labels[i])
    ax[1,0].plot(f,amp[i],color=c);ax[2,0].plot(f,plain[i],color=c)
ax[0,0].set(xlabel='Distance (nm)',ylabel='Z minus linear trend + 25 pm / row');ax[0,0].legend(fontsize=8)
ax[1,1].plot(f,mean,color='black');ax[1,1].fill_between(f,np.maximum(0,mean-sd),mean+sd,alpha=.2)
ax[2,1].plot(f,mean,label='Hann');ax[2,1].plot(f,plain.mean(axis=0),label='No window');ax[2,1].legend()
for axes in [ax[0,1],ax[1,0],ax[1,1],ax[2,0],ax[2,1]]:
    axes.set(xlabel='Spatial frequency (cycles/nm)',ylabel='Z FFT amplitude (pm)',xlim=(0,3.5),ylim=(0,None));axes.axvline(q3,color='red',ls='--',lw=1);axes.axvline(qb,color='gray',ls='--',lw=1)
for axes in [ax[1,0],ax[2,0]]:axes.set_xlim(.4,1.5)
for axes,title in zip(ax.flat,['Measured Z profiles (display offsets only)','Five individual FFTs | Hann','Low-frequency zoom | Hann','Mean amplitude +/- between-row SD','Low-frequency zoom | no window','Mean amplitude: window comparison']):axes.set_title(title,fontsize=10)
fig.suptitle('Dark Se paths: direct Z(x) FFT | rows 70,76,82,88,94\nRed: 1/(3a) reference; gray: 1/a | 99 measured pixels per line; no interpolation')
fig.savefig(OUT/'Se_Z_FFT_overview.png',dpi=180);fig.savefig(OUT/'Se_Z_FFT_overview.pdf')
np.savez(OUT/'Z_FFT_arrays.npz',x=x,f=f,raw=raw,z=z,amp=amp,plain=plain,mean=mean,sd=sd,rows=rows)
print('f, mean pm:',[(round(float(f[k]),3),round(float(mean[k]),3)) for k in range(1,15)])
print('individual q-neighbor amps .75 / 1.0:',amp[:,3:5].tolist())
(OUT/'README_中文.md').write_text('直接从原始 3DS 的 Z 参数读取显示像素行 70、76、82、88、94，每行 x=0..98，共 99 个实测像素。原始值保存于 Z_raw_pm.csv。每行去除线性趋势，Hann 窗后一侧幅度 FFT；无窗结果作为对照。未按原子散点取样、未插值、未补零。频率单位 cycles/nm，高度和频谱幅度单位 pm。平均为五行幅度的算术平均，不是复系数平均；阴影为行间标准差。图中实空间曲线每行错开 25 pm，仅用于显示。参考 q3='+str(q3)+'，qB='+str(qb)+'。原生频率步长约 0.25 cycles/nm，只有约 3.6 个候选周期。形貌高度的三倍周期分量不等于相邻原子间距三倍周期；亮斑高度/形状变化也可产生信号。原始数据及此前工程未修改。\n',encoding='utf-8')
if '--origin' in sys.argv:
    from origin_export import op,sheet_csv,title_export
    op.set_show(False);op.new();book=op.new_book('w',lname='Direct measured Se-path Z and FFT');book.name='SeZ'
    sh=sheet_csv(book,'Z_profile',OUT/'Z_detrended_pm.csv',True);fr=sheet_csv(book,'FFT_Hann',OUT/'Z_FFT_Hann_pm.csv');nw=sheet_csv(book,'FFT_NoWindow',OUT/'Z_FFT_no_window_pm.csv');sheet_csv(book,'Raw_Z',OUT/'Z_raw_pm.csv')
    for name,sheet,cols,freq in [('G_Se_Z_Profile',sh,range(1,6),False),('G_Se_Z_FFT',fr,range(1,6),True),('G_Se_Z_FFT_Mean',fr,[6],True),('G_Se_Z_FFT_NoWindow',nw,range(1,6),True)]:
        gp=op.new_graph(template='Origin',lname=name);gp.name=name;gl=gp[0]
        for j,col in enumerate(cols):
            p=gl.add_plot(sheet,coly=col,colx=0,type='l');p.color=matplotlib.colors.to_hex(colors[j])
        gl.rescale();gl.axis('x').title='Spatial frequency (cycles/nm)' if freq else 'Distance (nm)';gl.axis('y').title='Z FFT amplitude (pm)' if freq else 'Z minus linear trend (pm)';gl.set_xlim(0,3.5 if freq else float(x[-1]))
        if freq:
            gl.set_ylim(0,float(np.max(mean[f<=3.5]) if cols==[6] else np.max((plain if sheet==nw else amp)[:,f<=3.5]))*1.15)
        title_export(gp,OUT)
    path=OUT/'Se_Z_profiles_FFT_editable.opju';assert op.save(str(path));assert op.open(str(path),readonly=True,asksave=False)
    b=op.find_book('w','SeZ');err=0.
    for name,arr in [('Raw_Z',raw),('Z_profile',z),('FFT_Hann',amp),('FFT_NoWindow',plain)]:
        sheet=next(t for t in b if t.name==name)
        for j in range(5):err=max(err,float(np.max(np.abs(np.array(sheet.to_list(j+1),float)-arr[j]))))
    assert err==0; (OUT/'Origin_validation.json').write_text(json.dumps({'max_abs_error':err,'graphs':4,'raw_source_hash':d.hash}),encoding='utf-8');op.exit();print(path)
