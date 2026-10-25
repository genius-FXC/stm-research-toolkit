"""Deterministic, read-only grid sampling. Display (0,0) is top left."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import io,json,configparser,hashlib
import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image
from nanonis_reader import read_grid
from profile_fft import profile_fft
from spectrum_processing import process_spectra

ROOT=Path(__file__).resolve().parent

def palette():
    ini=configparser.ConfigParser();ini.read(ROOT/'assets/Nanox.cpal')
    ids=[int(v) for v in ini['ramp']['indexes'].split(',')]
    hexes=ini['ramp']['colors'].split(',')[2:]
    rgb=np.array([[int(h.zfill(6)[j:j+2],16) for j in (0,2,4)] for h in hexes])
    return np.rint(np.array([np.interp(np.linspace(0,252,256),ids,rgb[:,c]) for c in range(3)]).T).astype(np.uint8)

PALETTE=palette()

# A common, non-cycling distance ramp for browser, images and Origin.
DISTANCE_STOPS=np.array([[35,101,166],[42,131,144],[185,123,25],[180,66,66]])

def distance_colors(fractions):
    t=np.clip(np.asarray(fractions,dtype=float),0,1)
    rgb=np.rint(np.column_stack([np.interp(t,np.linspace(0,1,len(DISTANCE_STOPS)),DISTANCE_STOPS[:,c]) for c in range(3)])).astype(np.uint8)
    return ['#'+''.join(f'{v:02x}' for v in row) for row in rgb]

DISTANCE_PALETTE=distance_colors(np.linspace(0,1,256))

@dataclass
class LineSettings:
    start: list[float]
    end: list[float]
    step_px: float=1.0
    width_px: float=0.0
    curves: int=20
    normalize: bool=True
    norm_lo_mV: float=60.0
    norm_hi_mV: float=80.0
    zero_anchor: bool=True
    offset: float=0.024
    energy_min_mV: float=-40.0
    energy_max_mV: float=40.0
    view: str='topo'
    map_bias_mV: float=0.0
    topo_plane: bool=True
    profile_mode: str='raw'
    heat_min_percentile: float=1.0
    heat_max_percentile: float=99.0
    fft_window: str='hann'
    normalization_mode: str|None=None
    smoothing_method: str='none'
    smoothing_order: int=3
    smoothing_sigma: float=1.0
    x_offset_mV: float=0.0
    x_factor: float=1.0
    y_factor: float=1.0
    spectrum_min_mV: float|None=None
    spectrum_max_mV: float|None=None
    current_reference_sample: int=0
    rotation_quarters: int=0

    @classmethod
    def from_dict(cls,obj):
        allowed=cls.__dataclass_fields__
        result=cls(**{k:v for k,v in obj.items() if k in allowed})
        result.start=[float(v) for v in result.start];result.end=[float(v) for v in result.end]
        for k in ['norm_lo_mV','norm_hi_mV','offset','energy_min_mV','energy_max_mV','map_bias_mV','heat_min_percentile','heat_max_percentile']:setattr(result,k,float(getattr(result,k)))
        # Old saved lines are migrated to direct measured pixels, never interpolated.
        result.step_px=1.0;result.width_px=0.0
        if result.rotation_quarters not in (0,1,2,3):raise ValueError("旋转角度必须为 0/90/180/270 度")
        result.rotation_quarters=int(result.rotation_quarters)
        result.curves=int(result.curves)
        for k in ['smoothing_sigma','x_offset_mV','x_factor','y_factor','spectrum_min_mV','spectrum_max_mV']:
            if getattr(result,k) is not None:setattr(result,k,float(getattr(result,k)))
        for k in ['smoothing_order','current_reference_sample']:
            val=float(getattr(result,k))
            if not np.isfinite(val) or val!=int(val):raise ValueError('平滑 order 和参考谱序号必须是整数。')
            setattr(result,k,int(val))
        return result

class GridDataset:
    def __init__(self,path):
        self.path=Path(path).resolve();raw=read_grid(self.path)
        order=np.argsort(raw['bias']);self.bias=np.round(raw['bias'][order]*1000,9)
        self.acquisition_bias_order=np.argsort(order)
        self.g=raw['lix'][::-1,:,order].astype(np.float64)*1e12
        self.current_first_pA=None if raw['current'] is None else raw['current'][::-1,:,0].astype(np.float64)*1e12
        self.current_first_bias_mV=float(raw['bias'][0]*1000)
        self.current_channel=raw['current_channel']
        # Advanced indexing order is explicit; retain (display_y, display_x, bias).
        assert self.g.shape==(raw['ny'],raw['nx'],raw['npts'])
        self.z=raw['z'][::-1].astype(np.float64)*1e12
        self.ny,self.nx=self.z.shape;yy,xx=np.indices(self.z.shape)
        A=np.column_stack([np.ones(self.z.size),xx.ravel(),yy.ravel()])
        self.topo=self.z-(A@np.linalg.lstsq(A,self.z.ravel(),rcond=None)[0]).reshape(self.z.shape)
        lab=np.column_stack([raw['x'][::-1].ravel(),raw['y'][::-1].ravel()])*1e9
        self.transform=np.linalg.lstsq(A,lab,rcond=None)[0]
        self.hash=raw['sha256'];self.header=raw['fields'];self.settings=raw['settings']
        self.bias_endpoint_sources=raw['bias_endpoint_sources']
        self.label=self.path.name

    def info(self):
        return dict(filename=self.label,source_path=str(self.path),nx=self.nx,ny=self.ny,n_bias=len(self.bias),bias_mV=self.bias.tolist(),bias_endpoint_sources=self.bias_endpoint_sources,pixel_step_nm=np.linalg.norm(self.transform[1:],axis=1).tolist(),channel='LI Demod 1 X',signal_unit='pA',palette=PALETTE.tolist(),distance_palette=DISTANCE_PALETTE,source_sha256=self.hash,coordinate_convention='display top-left; source row=ny-1-display_y; source col=display_x',source_topo_parameter='Z (m)',current_available=self.current_first_pA is not None,current_channel=self.current_channel,current_first_bias_mV=self.current_first_bias_mV)

    def map_values(self,view='topo',bias_mV=0,plane=True):
        if view=='topo':return (self.topo if plane else self.z),'Z minus plane (pm)' if plane else 'Z (pm)'
        if view!='mapping':raise ValueError('未知图像类型。')
        k=int(np.argmin(abs(self.bias-float(bias_mV))))
        return self.g[:,:,k],f'LI X {self.bias[k]:+g} mV (pA)'

    def image_png(self,view='topo',bias_mV=0,plane=True):
        a,label=self.map_values(view,bias_mV,plane)
        lo,hi=np.percentile(a,[1,99]);span=max(float(hi-lo),1e-12)
        rgb=PALETTE[np.rint(np.clip((a-lo)/span,0,1)*255).astype(int)]
        f=io.BytesIO();Image.fromarray(rgb).save(f,format='PNG')
        return f.getvalue(),dict(min=float(lo),max=float(hi),label=label)

    def lab(self,xy):
        xy=np.asarray(xy);return np.column_stack([np.ones(len(xy)),xy])@self.transform

    def sample(self,xy):
        xy=np.asarray(xy)
        return np.column_stack([map_coordinates(self.g[:,:,i],[xy[:,1],xy[:,0]],order=1,mode='nearest') for i in range(len(self.bias))])

    def sample_height(self,xy):
        xy=np.asarray(xy)
        return np.column_stack([map_coordinates(image,[xy[:,1],xy[:,0]],order=1,mode='nearest') for image in (self.z,self.topo)])

def pixel_path(a,b):
    """Canonical Bresenham: unique 8-connected pixels; reversing A/B reverses exactly."""
    a=tuple(int(v) for v in a);b=tuple(int(v) for v in b)
    reverse=a>b
    (x,y),(bx,by)=(b,a) if reverse else (a,b)
    dx=abs(bx-x);dy=-abs(by-y);sx=1 if x<bx else -1;sy=1 if y<by else -1;err=dx+dy;points=[]
    while True:
        points.append((x,y))
        if (x,y)==(bx,by):break
        e2=2*err
        if e2>=dy:err+=dy;x+=sx
        if e2<=dx:err+=dx;y+=sy
    return np.asarray(points[::-1] if reverse else points,dtype=int)

def bin_edges(centers):
    centers=np.asarray(centers,float)
    return np.r_[centers[0]-(centers[1]-centers[0])/2,(centers[:-1]+centers[1:])/2,centers[-1]+(centers[-1]-centers[-2])/2]

def extract_line(data: GridDataset,cfg: LineSettings,line_id='L001'):
    a=np.asarray(cfg.start,float);b=np.asarray(cfg.end,float)
    if a.shape!=(2,) or b.shape!=(2,) or not np.isfinite(np.r_[a,b]).all():raise ValueError('请指定有限的两个端点。')
    if not (0<=a[0]<=data.nx-1 and 0<=b[0]<=data.nx-1 and 0<=a[1]<=data.ny-1 and 0<=b[1]<=data.ny-1):raise ValueError('端点需位于图像像素中心范围内。')
    numeric=[cfg.offset,cfg.norm_lo_mV,cfg.norm_hi_mV,cfg.energy_min_mV,cfg.energy_max_mV,cfg.map_bias_mV,cfg.heat_min_percentile,cfg.heat_max_percentile]
    if not np.isfinite(numeric).all():raise ValueError('参数必须是有限数字。')
    a=np.floor(a+.5).astype(int);b=np.floor(b+.5).astype(int)
    cfg.start=a.tolist();cfg.end=b.tolist();cfg.step_px=1.0;cfg.width_px=0.0
    if np.array_equal(a,b):raise ValueError('起点和终点必须是两个不同的像素。')
    if not 2<=cfg.curves<=2000:raise ValueError('瀑布显示数范围为 2–2000；最多显示沿线实际像素数。')
    if cfg.view not in ('topo','mapping'):raise ValueError('未知图像类型。')
    if cfg.profile_mode not in ('raw','subtract_average','linear_fit'):raise ValueError('未知剖面处理方式。')
    if cfg.fft_window not in ('hann','none'):raise ValueError('未知 FFT 窗函数。')
    if not 0<=cfg.heat_min_percentile<cfg.heat_max_percentile<=100:raise ValueError('色标百分位需满足 0 ≤ 最暗 < 最亮 ≤ 100。')
    if cfg.energy_min_mV>=cfg.energy_max_mV:raise ValueError('偏压下限必须小于上限。')
    energy_mask=(data.bias>=cfg.energy_min_mV-1e-8)&(data.bias<=cfg.energy_max_mV+1e-8)
    if energy_mask.sum()<2:raise ValueError('显示范围至少需要两个实测偏压点。')
    if cfg.energy_min_mV<data.bias.min()-1e-7 or cfg.energy_max_mV>data.bias.max()+1e-7:raise ValueError('显示范围不能超出文件的偏压范围。')
    xy=pixel_path(a,b);n=len(xy)
    if n>2000:raise ValueError('沿线像素数超过 2000；请缩短线段。')
    lab=data.lab(xy);direction=lab[-1]-lab[0];length=float(np.linalg.norm(direction))
    if length<=0:raise ValueError('空间坐标校准给出的线长为零。')
    s=(lab-lab[0])@(direction/length)
    if np.any(np.diff(s)<=0):raise ValueError('空间坐标校准导致投影距离非递增，请检查文件的 X/Y 参数。')
    center=data.g[xy[:,1],xy[:,0]].copy();signal=center.copy()
    height_center=np.column_stack([data.z[xy[:,1],xy[:,0]],data.topo[xy[:,1],xy[:,0]]]);height=height_center.copy();valid_counts=np.ones(n,dtype=int)
    strip_offsets=np.array([0.])
    current=getattr(data,'current_first_pA',None)
    sampled_current=None if current is None else current[xy[:,1],xy[:,0]].copy()
    acquisition_order=getattr(data,'acquisition_bias_order',np.arange(len(data.bias)))
    processed=process_spectra(data.bias[acquisition_order],signal[:,acquisition_order],cfg,sampled_current,getattr(data,'current_first_bias_mV',None))
    cfg.normalization_mode=processed['metadata']['method'];cfg.normalize=cfg.normalization_mode!='none'
    denominator=processed['denominator'];normalized=processed['values'] if cfg.normalize else None
    base=processed['values'];wf_bias=processed['bias_mV']
    wf_mask=(wf_bias>=cfg.energy_min_mV-1e-8)&(wf_bias<=cfg.energy_max_mV+1e-8)
    if wf_mask.sum()<2:raise ValueError('处理后瀑布谱在显示范围内不足两个偏压点，请调整裁切/显示范围。')
    zero=int(np.argmin(abs(wf_bias)))
    indices=np.unique(np.rint(np.linspace(0,n-1,min(n,cfg.curves))).astype(int))
    wf=base[indices].copy()
    if cfg.zero_anchor:wf-=base[indices][:,[zero]]
    wf+=np.arange(len(indices))[:,None]*cfg.offset
    k=int(np.argmin(abs(data.bias-cfg.map_bias_mV)))
    profile_raw=height[:,0].copy() if cfg.view=='topo' else signal[:,k].copy()
    trend=np.zeros(n)
    if cfg.profile_mode=='subtract_average':trend[:]=profile_raw.mean()
    elif cfg.profile_mode=='linear_fit':
        A=np.column_stack([np.ones(n),s]);mean=profile_raw.mean()
        trend=mean+A@np.linalg.lstsq(A,profile_raw-mean,rcond=None)[0]
    profile=profile_raw-trend
    fft=profile_fft(s,profile,cfg.fft_window)
    fft.update(channel='Z' if cfg.view=='topo' else 'LI X',unit='pm' if cfg.view=='topo' else 'pA',bias_mV=None if cfg.view=='topo' else float(data.bias[k]),profile_mode=cfg.profile_mode)
    lo,hi=np.percentile(signal[:,energy_mask],[cfg.heat_min_percentile,cfg.heat_max_percentile])
    hi=max(float(hi),float(lo)+max(abs(float(lo))*1e-12,1e-12))
    return dict(line_id=line_id,config=asdict(cfg),xy=xy,lab_xy=lab,s_nm=s,position_edges_nm=bin_edges(s),bias_mV=data.bias.copy(),signal_pA=signal,center_pA=center,z_raw_center_pm=height_center[:,0],z_plane_center_pm=height_center[:,1],z_raw_used_pm=height[:,0],z_plane_used_pm=height[:,1],z_display_relative_pm=height[:,0]-height[0,0],profile_raw=profile_raw,profile_values=profile,profile_trend=trend,profile_unit='pm' if cfg.view=='topo' else 'pA',profile_channel='Z' if cfg.view=='topo' else 'LI X',profile_bias_mV=None if cfg.view=='topo' else float(data.bias[k]),fft=fft,heat_limits_pA=[float(lo),hi],normalized=normalized,norm_denominator_pA=denominator,processed_spectra=base,waterfall_bias_mV=wf_bias,waterfall_energy_mask=wf_mask,processing=processed['metadata'],current_first_pA=sampled_current,current_correction_factor=processed['current_factor'],waterfall=wf,waterfall_indices=indices,waterfall_colors_hex=distance_colors(s[indices]/s[-1]),energy_mask=energy_mask,valid_strip_samples=valid_counts,strip_offsets_px=strip_offsets,zero_anchor_bias_mV=float(wf_bias[zero]),length_nm=length,n_samples=n,sample_names=[f'{line_id}_S{i+1:04d}' for i in range(n)])

def preview_payload(result):
    keys=['line_id','config','xy','lab_xy','s_nm','position_edges_nm','bias_mV','signal_pA','z_raw_center_pm','z_plane_center_pm','z_raw_used_pm','z_plane_used_pm','z_display_relative_pm','profile_raw','profile_values','profile_trend','profile_unit','profile_channel','profile_bias_mV','fft','heat_limits_pA','waterfall','waterfall_bias_mV','processing','waterfall_indices','waterfall_colors_hex','zero_anchor_bias_mV','length_nm','n_samples','sample_names','valid_strip_samples']
    return {k:(result[k].tolist() if isinstance(result[k],np.ndarray) else result[k]) for k in keys}

def settings_json(data,result):
    cfg=result['config']
    meta=dict(format='stm-line-spectrum/v1',sampling_version='measured-pixels/v2',source_file=str(data.path),source_name=data.label,source_sha256=data.hash,line_id=result['line_id'],settings=cfg,length_nm=result['length_nm'],n_samples=result['n_samples'],waterfall_sample_names=[result['sample_names'][i] for i in result['waterfall_indices']],waterfall_colors_hex=result['waterfall_colors_hex'],waterfall_distances_nm=result['s_nm'][result['waterfall_indices']].tolist(),waterfall_color_mapping='continuous RGB gradient of projected distance/total_length from A (blue) to B (red), no cycling',zero_anchor_bias_mV=result['zero_anchor_bias_mV'],display_coordinate_origin='upper-left pixel center (0,0), integer coordinates',source_row_formula='ny-1-display_y',source_col_formula='display_x',display_to_recorded_lab_nm=data.transform.tolist(),bias_axis='nominal Sweep Start/End, sorted ascending; no repeated bias offset correction',bias_endpoint_sources=data.bias_endpoint_sources,units='raw LI X A converted to pA, not calibrated absolute dI/dV',no_energy_smoothing=True,spatial_sampling='canonical Bresenham unique 8-connected measured pixels; no interpolation or width averaging; exact reverse on A/B reversal',distance_axis='recorded lab coordinates projected on straight A-to-B direction, not zigzag arc length; adjacent pixel distances can differ',waterfall_policy='at most requested display count, uniformly selected pixel indices including A and B; all measured spectra always exported',heat_limits_pA=result['heat_limits_pA'],heat_percentile_population='all selected pixels within displayed measured bias window',profile=dict(channel=result['profile_channel'],bias_mV=result['profile_bias_mV'],unit=result['profile_unit'],mode=cfg['profile_mode'],source='raw Z fixed parameter or raw selected-bias LI X; independent of topo-plane display and waterfall normalization',linear_fit='least-squares constant+slope versus projected distance; subtract fitted line',no_smoothing=True),normalization='each pixel divided by mean in configured absolute-bias window' if cfg['normalize'] else 'disabled')

    meta.update(processing=result['processing'],no_energy_smoothing=cfg['smoothing_method']=='none',normalization=result['processing']['label'],waterfall_bias_axis='processed bias: (raw bias + x_offset) * x_factor, then crop; original acquisition order retained',raw_bias_order='raw heat-map spectra sorted ascending; processing restores original acquisition order; current correction uses first original acquisition point')
    return meta
