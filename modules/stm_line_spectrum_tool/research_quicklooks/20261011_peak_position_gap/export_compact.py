from pathlib import Path
import sys,json,csv
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from origin_export import op,sheet_csv,title_export
OUT=HERE/'results/06_origin_compact';OUT.mkdir(exist_ok=True)
a=np.load(HERE/'results/04_tables/analysis_arrays.npz');x=a['distance_nm'];f=a['frequency_cycles_per_nm']
colors=['#46327e','#3e689d','#21918c','#5cc863','#bc9e19']
def main():
    op.set_show(False);op.new();bk=op.new_book('w',lname='Dark Se-Se and interleaved Fe-path metrics');bk.name='Metrics'
    index=[];expected={};first=True
    for group,inds in [('Se',list(range(6,11))),('Fe',list(range(1,6)))]:
        rows=a['rows'][inds];names=[f'{group} path y={r}' for r in rows]
        for key,short,desc in [('pos_quad5','Pos','Positive fitted peak'),('neg_quad5','Neg','Negative fitted peak'),('gap_width','W50','Half-depth full width')]:
            for freq in [False,True]:
                name=f'G_{group}_{short}_'+('FFT' if freq else 'Line');sheetname=name[2:]
                axis=f if freq else x
                vals=(a['gap_width_HannFFT'] if key=='gap_width' else a['freq_'+key]) if freq else a[key]
                vals=vals[inds];expected[sheetname]=vals
                file=OUT/(sheetname+'.csv')
                with file.open('w',newline='',encoding='utf-8-sig') as h:
                    w=csv.writer(h);w.writerow(['frequency_cycles_per_nm' if freq else 'distance_nm',*names])
                    for xx,v in zip(axis,vals.T):w.writerow([xx,*[float(t) if np.isfinite(t) else '' for t in v]])
                sh=sheet_csv(bk,sheetname,file,first);first=False
                for c in range(1,6):sh.set_label(c,'meV','U')
                method='Hann FFT' if key=='gap_width' else 'Spatial sine regression; missing samples excluded'
                gp=op.new_graph(template='Origin',lname=f'{group} paths | {desc}'+(' | '+method if freq else ''));gp.name=name;gl=gp[0]
                counts=np.isfinite(a[key][inds]).sum(axis=1).tolist();good=np.isfinite(vals)
                for j in range(5):
                    if good[j].any():
                        p=gl.add_plot(sh,coly=j+1,colx=0,type='l' if freq or key=='gap_width' else 'y');p.color=colors[j]
                        if not freq and key!='gap_width':p.symbol_size=2
                gl.rescale();gl.set_xlim(0,3.5 if freq else float(x[-1]),.5)
                gl.axis('x').title='Spatial frequency (cycles/nm)' if freq else 'Distance (nm)';gl.axis('y').title='Amplitude (meV)' if freq else ('W50 (meV)' if key=='gap_width' else 'Peak energy (meV)')
                vv=vals[:,f<=3.5] if freq else vals;vv=vv[np.isfinite(vv)]
                if vv.size:
                    lo,hi=float(vv.min()),float(vv.max());pad=max((hi-lo)*.12,.3)
                    gl.set_ylim(0,hi*1.2) if freq else gl.set_ylim(lo-pad,hi+pad)
                else:gl.set_ylim(0,1) if freq else gl.set_ylim(6,18) if key=='pos_quad5' else gl.set_ylim(-24,-8)
                legend=gl.label('Legend')
                if legend:legend.remove()
                ymax=gl.get_float('y.to');ymin=gl.get_float('y.from');xmax=3.5 if freq else float(x[-1])
                for j,label in enumerate(names):
                    lab=gl.add_label(f'y={rows[j]}, n={counts[j]}',xmax*1.01,ymax-(j+1)*.075*(ymax-ymin));lab.set_int('fsize',8);lab.set_int('color',op.ocolor(colors[j]))
                if not vv.size:
                    msg='No resolved positive peaks' if not freq else 'Insufficient valid peak positions\nSpectrum not calculated'
                    lab=gl.add_label(msg,xmax*.15,(ymin+ymax)/2);lab.set_int('fsize',12)
                elif freq:
                    guide=bk.add_sheet(sheetname+'_ref');guide.cols=3;guide.from_list(0,[0,ymax],axis='Y');guide.from_list(1,[float(a['q3'])]*2,axis='X');guide.from_list(2,[float(a['qBragg'])]*2,axis='X')
                    for col,color,label in [(1,'#bf5b00','1/(3a)'),(2,'#888888','1/a')]:
                        p=gl.add_plot(guide,coly=0,colx=col,type='l');p.color=color;gl.add_label(label,float(a['q3'] if col==1 else a['qBragg'])+.02,ymax*.93)
                title_export(gp,OUT);index.append({'graph':name,'group':group,'metric':key,'domain':'frequency' if freq else 'real','method':method if freq else '5-point quadratic peak fit' if key!='gap_width' else 'normalized suppression half-depth full width','valid_counts':counts,'available':bool(vv.size)})
                print(name,flush=True)
    path=OUT/'dark_Se_Fe_peak_width_compact.opju';assert op.save(str(path));assert op.open(str(path),readonly=True,asksave=False)
    book=op.find_book('w','Metrics');err=0
    for name,arr in expected.items():
        sh=next(t for t in book if t.name==name)
        for j in range(5):
            got=np.array([float(t) if t not in (None,'') else np.nan for t in sh.to_list(j+1)])
            mask=np.isfinite(arr[j]);assert np.array_equal(np.isfinite(got),mask)
            if mask.any():err=max(err,float(np.max(abs(got[mask]-arr[j,mask]))))
    assert err==0
    (OUT/'validation.json').write_text(json.dumps({'graphs':12,'max_numeric_error':err,'missing_masks_preserved':True,'saved_and_reopened':True}),encoding='utf-8')
    (OUT/'graph_index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    (OUT/'README_中文.md').write_text('精简版：只含 dark 区域两组路径的拟合正峰位、拟合负峰位、半深度全宽 W50，以及各自频率页面，共 12 页。Se：像素行 70/76/82/88/94；Fe：交错路径 73/79/85/91/97。Fe 为路径名称，不表示沿线每个像素是 Fe。原有工程不覆盖。\n\n实空间采用折线；稀疏峰位加极小符号以显示孤立有效点，缺失点不补值。W50 是归一化谱凹陷的半深度全宽，不是相干峰自身的 FWHM。\n\nG_Se_Pos_Line/FFT、G_Se_Neg_Line/FFT、G_Se_W50_Line/FFT；G_Fe_Pos_Line/FFT、G_Fe_Neg_Line/FFT、G_Fe_W50_Line/FFT。\n\nW50 使用常规去趋势 Hann FFT。Se 负峰存在缺失，频率页面实际是沿用已验证的空间正弦回归幅度谱，图标题已注明，不是完整等间隔 FFT。其余峰位频谱数据不足，页面明确标注不计算；Se 正峰全部缺失，实空间页面同样明确说明。每条线的有效数列在图例中。所有处理、搜索窗与质控沿用此前分析，不重新放宽阈值来填满曲线。原始数据和全部旧结果保持不变。\n',encoding='utf-8')
    print(path)
if __name__=='__main__':
    try:main()
    finally:op.exit()
