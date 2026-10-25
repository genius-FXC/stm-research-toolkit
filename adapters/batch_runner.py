"""Parameter adapters; numerical functions are imported from retained modules."""
from pathlib import Path
import sys,json,csv,hashlib,datetime,traceback,importlib.util
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'modules/nanonis_converter/scripts'),str(ROOT/'modules/sxm_fft_stp_20261013'),str(ROOT/'modules/stm_line_spectrum_tool')]
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(job):
    if job.get('tool_id') not in ['3ds-sxm','fft-stp','batch-ifft']:raise ValueError('未知工具')
    src=Path(job['input']).resolve();out=Path(job['output']).resolve()
    if not src.exists():raise ValueError('输入不存在')
    if out==src or (src.is_dir() and src in out.parents):raise ValueError('输出不能是输入目录或其子目录')
    if out.exists() and any(out.iterdir()):raise ValueError('输出目录非空，拒绝覆盖')
    ext='.3ds' if job['tool_id']=='3ds-sxm' else '.sxm'
    files=sorted(src.glob('*'+ext)) if src.is_dir() else [src]
    if not files or any(p.suffix.lower()!=ext for p in files):raise ValueError('没有匹配的输入文件：'+ext)
    if len({p.stem.casefold() for p in files})!=len(files):raise ValueError('输入文件名发生大小写冲突')
    if job['tool_id']=='batch-ifft':
        centers=job.get('centers',[]);radius=float(job.get('radius',0))
        if not centers or not np.isfinite(radius) or radius<=0:raise ValueError('需要有效 Q 坐标与正 mask 半径')
        for pair in centers:
            if len(pair)!=2 or not all(np.isfinite(pair)):raise ValueError('Q 应为有限的 qx,qy')
            if np.hypot(*pair)<=radius:raise ValueError('Mask 包含零频，请减小半径或调整 Q')
    return files,out

def export_grid(source,destination,job):
    import export_3ds_to_sxm as writer
    # The line tool's maintained reader supports compact headers as well.
    from nanonis_reader import read_grid
    grid=read_grid(source);destination.mkdir();rows=[]
    topo=float(grid['fields'].get('Bias>Bias (V)','nan'))
    maps=[('Z','m',topo,None,grid['z'])]
    order=np.argsort(grid['bias'],kind='stable') if job.get('order','ascending')=='ascending' else range(grid['npts'])
    maps += [('LI Demod 1 X','A',float(grid['bias'][i]),int(i),grid['lix'][:,:,i]) for i in order]
    for n,(name,unit,bias,index,data) in enumerate(maps):
        path=destination/f'{n:04d}_{"Topo_Z" if index is None else "LI_X_"+format(bias*1000,"+.6g")+"mV"}.sxm'
        writer.write_sxm(path,grid,data,name,unit,bias,index)
        writer.verify_nanonis_header(path)
        from _sxm_reader import read_sxm
        loaded=read_sxm(path);actual=loaded.channels[0].data
        if not np.array_equal(actual,np.asarray(data,dtype=np.float32).astype(float),equal_nan=True):raise ValueError('像素重读校验失败')
        if loaded.scan_pixels!=(grid['nx'],grid['ny']) or not np.allclose(loaded.scan_range,grid['settings'][2:4],rtol=1e-12,atol=0):raise ValueError('空间范围重读校验失败')
        if (loaded.channels[0].name,loaded.channels[0].unit)!=(name,unit):raise ValueError('通道或单位重读校验失败')
        rows.append({'file':path.name,'channel':name,'unit':unit,'bias_V':bias,'source_index':index,'sha256':sha(path)})
    (destination/'index.json').write_text(json.dumps({'source':str(source),'sha256':sha(source),'grid_settings':grid['settings'],'files':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    return {'files':len(rows),'pixel_reread_verified':True}

def run(job):
    files,out=prepare(job);out.mkdir(parents=True,exist_ok=True);report={'job':job,'status':'running','started':datetime.datetime.now().isoformat(),'files':[],'numerical_core':'existing modules; adapter changes IO only','native_gui_validation':False}
    report_path=out/'job_report.json'
    try:
        for i,src in enumerate(files,1):
            before=sha(src)
            if job['tool_id']=='3ds-sxm':result=export_grid(src,out/src.stem,job)
            else:
                import fft_stp_batch as fft
                s=fft.read_sxm(src)
                if len(s.channels)!=1 or s.channels[0].direction!='forward' or s.header_fields.get('SCAN_DIR')!=['up']:raise ValueError('原 FFT 核心要求单通道 forward、SCAN_DIR=up 的 SXM；请先用既有转换器导出。')
                if s.scan_range is None:raise ValueError('缺少扫描范围')
                if job['tool_id']=='fft-stp':
                    r=fft.plane_and_fft(s.channels[0].data,*s.scan_range,s.channels[0].unit);p=out/('FFT'+src.stem+'.stp')
                    fft.write_stp(p,r,{'source_sxm':src.name,'source_3ds_title':'not supplied','bias_mV':None,'source_sha256':before,'source_channel':s.channels[0].name,'source_scan_angle_deg':s.header_fields.get('SCAN_ANGLE',['unknown'])[0]})
                    if not np.array_equal(fft.read_stp(p)[0],r['native_fft']):raise ValueError('STP 重读不一致')
                    result={'file':p.name,'sha256':sha(p),'reread_verified':True,'padding':r['padded'],'fft_shape':r['fft_shape']}
                else:
                    import bragg_ifft_trials as inv
                    r,display,F=inv.transform(s)
                    if any(abs(x)>max(abs(r['kx'])) or abs(y)>max(abs(r['ky'])) for x,y in job['centers']):raise ValueError('Q 超出可采样频率范围')
                    centers=[{'kx_nm_inv':x,'ky_nm_inv':y} for x,y in job['centers']]
                    mask=inv.make_mask(r,centers,float(job['radius']))
                    if not np.any(mask):raise ValueError('Mask 未覆盖任何频率像素')
                    filtered,imag=inv.inverse(F,mask,r['source_shape']);size=np.array(s.scan_range)*1e9;unit='pm' if s.channels[0].unit=='m' else 'pA'
                    formats=job.get('formats',['stp','npz']);written=[]
                    for fmt in formats:
                        p=out/(src.stem+'_iFFT.'+fmt)
                        if fmt=='stp':inv.write_real_stp(p,filtered,size,unit,'ToolHub; source='+src.name+'; phase-preserving paired cosine mask')
                        elif fmt=='npz':np.savez_compressed(p,reconstruction=filtered,mask=mask,complex_fft=F,kx=r['kx'],ky=r['ky'],range_nm=size,unit=unit)
                        elif fmt=='csv':np.savetxt(p,filtered,delimiter=',',header='unit='+unit,comments='# ')
                        else:raise ValueError('不支持该输出格式')
                        written.append(p.name)
                    result={'files':written,'imaginary_fraction':imag,'padding':r['padded'],'unit':unit}
            if before!=sha(src):raise RuntimeError('源文件在处理期间变化')
            report['files'].append({'source':str(src),'source_sha256':before,**result});print(f'{i}/{len(files)} 完成：{src.name}',flush=True)
        report['status']='complete'
    except Exception as e:report['status']='failed';report['error']=str(e);raise
    finally:report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
if __name__=='__main__':
    job=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));run(job)
