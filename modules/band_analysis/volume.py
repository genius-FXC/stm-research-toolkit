"""Band Analysis: physical FFT volume and shared contrast. Sources are read only."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
HUB = HERE.parents[1]
sys.path.insert(0, str(HUB / 'modules/2d-lockin'))
sys.path.insert(0, str(HUB / 'modules/nanonis_converter/scripts'))
from stmfft import analysis as original_fft
from stmfft.sxm import read_sxm
from export_3ds_to_sxm import read_grid

def groups(manifest):
    with Path(manifest).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return list(dict.fromkeys(r['source_3ds_title'] for r in rows if r['image_kind'] == 'LI_X'))

@dataclass
class Volume:
    energy: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    amplitude: np.ndarray
    meta: dict

    def __post_init__(self):
        self.energy, self.qx, self.qy = [np.asarray(a, dtype=float) for a in (self.energy, self.qx, self.qy)]
        self.amplitude = np.asarray(self.amplitude, dtype=float)
        if self.amplitude.shape != (len(self.energy), len(self.qy), len(self.qx)):
            raise ValueError('Volume shape does not match E, qy, qx axes.')
        for a in (self.energy, self.qx, self.qy):
            if len(a) < 2 or not np.all(np.isfinite(a)) or not np.all(np.diff(a) > 0):
                raise ValueError('Each axis must have at least two strictly increasing finite samples.')
        if not np.all(np.isfinite(self.amplitude)) or np.any(self.amplitude < 0):
            raise ValueError('Incomplete or negative amplitude volume; no automatic filling is allowed.')
        self._interps = {};self._power=None

    def scalar(self, metric='amplitude'):
        if metric=='amplitude':return self.amplitude
        if self._power is None:self._power=self.amplitude ** 2
        return self._power

    def sample(self, energy, qx, qy, metric='amplitude', interpolation='linear'):
        if metric not in self._interps:
            self._interps[metric] = RegularGridInterpolator((self.energy, self.qy, self.qx), self.scalar(metric), bounds_error=False, fill_value=np.nan)
        ee, xx, yy = np.broadcast_arrays(energy, qx, qy)
        points = np.stack((ee, yy, xx), axis=-1)
        return self._interps[metric](points, method=interpolation)

    def tilted(self, e0, sx=0, sy=0, metric='amplitude', interpolation='linear'):
        xx, yy = np.meshgrid(self.qx, self.qy)
        ee = e0 + sx * xx + sy * yy
        return Cut(self.sample(ee, xx, yy, metric, interpolation), self.qx, self.qy, 'qx (cycles/nm)', 'qy (cycles/nm)',
                   {'mode':'tilted', 'E0_mV':e0, 'dE_dqx':sx, 'dE_dqy':sy, 'metric':metric, 'interpolation':interpolation},
                   {'energy_mV':ee, 'qx':xx, 'qy':yy})

    def path(self, vertices, width=0, metric='amplitude', interpolation='linear', samples=400):
        p = np.asarray(vertices, float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.all(np.isfinite(p)):
            raise ValueError('请在等能量图上点击至少两个路径点。')
        lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError('路径有重复的相邻点，请删除重复点。')
        if width < 0 or not np.isfinite(width):
            raise ValueError('路径宽度必须为非负有限数值。')
        corners = np.r_[0., np.cumsum(lengths)]
        # Segment endpoints are included exactly; no interpolating across a corner.
        coords, distances, normals = [], [], []
        for j, length in enumerate(lengths):
            n = max(2, int(np.ceil(samples * length / corners[-1])) + 1)
            t = np.linspace(0, 1, n)
            if j < len(lengths)-1: t = t[:-1]
            direction = (p[j+1]-p[j])/length
            coords.append(p[j] + t[:, None]*(p[j+1]-p[j]))
            distances.append(corners[j] + t*length)
            normals.append(np.tile([-direction[1], direction[0]], (len(t), 1)))
        q = np.concatenate(coords); distance = np.concatenate(distances); normal = np.concatenate(normals)
        dq = min(np.min(np.diff(self.qx)), np.min(np.diff(self.qy)))
        offsets = [0.] if width == 0 else np.linspace(-width/2, width/2, max(3, int(np.ceil(width/dq))*2+1))
        planes = []
        for offset in offsets:
            qq = q + offset*normal
            planes.append(self.sample(self.energy[:, None], qq[None,:,0], qq[None,:,1], metric, interpolation))
        # Full width mean; any out-of-bounds part is missing, never silently narrowed.
        values = np.mean(planes, axis=0)
        return Cut(values, distance, self.energy, 'Path distance (cycles/nm)', 'Bias (mV)',
                   {'mode':'path', 'vertices':p.tolist(), 'corner_distances':corners.tolist(), 'full_width':width,
                    'width_samples':len(offsets), 'metric':metric, 'interpolation':interpolation,
                    'corner_normal':'outgoing segment at internal vertex; incoming at final vertex'},
                   {'path_qx':q[:,0], 'path_qy':q[:,1]})

    def plane(self, angle, offset, metric='amplitude', interpolation='linear', width=0):
        theta = np.deg2rad(angle); t = np.array([np.cos(theta),np.sin(theta)])
        n = np.array([-t[1], t[0]]); origin = offset*n
        lo, hi = -np.inf, np.inf
        for k, axis in enumerate((self.qx,self.qy)):
            if abs(t[k]) < 1e-12:
                if origin[k] < axis[0] or origin[k] > axis[-1]: raise ValueError('切面未穿过数据体。')
            else:
                a,b = sorted(((axis[0]-origin[k])/t[k], (axis[-1]-origin[k])/t[k]))
                lo,hi=max(lo,a),min(hi,b)
        if hi<=lo: raise ValueError('切面未穿过数据体。')
        cut = self.path([origin+lo*t,origin+hi*t], width, metric, interpolation)
        cut.x += lo
        cut.xlabel = 'q along plane (cycles/nm)'
        cut.params.update(mode='plane', angle_deg=angle, offset_cycles_nm=offset)
        return cut

@dataclass
class Cut:
    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    xlabel: str
    ylabel: str
    params: dict
    extra: dict

@dataclass
class Contrast:
    low: float = 0
    high: float = 1
    gamma: float = 1
    logarithmic: bool = True
    floor: float = 1e-12
    cmap: str = 'gray_r'

    def transform(self, value):
        return np.log10(np.maximum(value, self.floor)) if self.logarithmic else np.asarray(value)

    def fit(self, entire_volume, lower=1, upper=99.5):
        v = np.asarray(entire_volume)
        positive = v[np.isfinite(v) & (v > 0)]
        self.floor = float(positive.min()) if len(positive) else 1e-12
        a = self.transform(v); good = a[np.isfinite(a)]
        self.low,self.high = map(float,np.percentile(good,[lower,upper]))
        if self.high<=self.low: self.high=self.low+1

    def normalized(self, value):
        if not self.high>self.low: raise ValueError('色标上限必须大于下限。')
        v = self.transform(value)
        return np.clip((v-self.low)/(self.high-self.low),0,1) ** (1/self.gamma)

    def rgba(self, value):
        from matplotlib import colormaps
        norm = self.normalized(value)
        out = colormaps[self.cmap](np.nan_to_num(norm,nan=0), bytes=True)
        out[~np.isfinite(value),3]=0
        return np.ascontiguousarray(out)

def build_volume(images, energies, size_nm, meta, preprocess='plane', window='hann', progress=lambda x:None):
    images=np.asarray(images,float); energies=np.asarray(energies,float)
    if images.ndim!=3 or len(images)!=len(energies) or not np.all(np.isfinite(images)):
        raise ValueError('原始图像不完整或存在缺失像素；停止而不补值。')
    order=np.argsort(energies); result=[]
    for j,i in enumerate(order):
        clean,_ = original_fft.preprocess_image(images[i],preprocess)
        fft=original_fft.compute_fft(clean,*size_nm,window=window)
        result.append(np.abs(fft.spectrum));progress(int(100*(j+1)/len(order)))
    meta=dict(meta,preprocess=preprocess,window=window,fft_core=str(Path(original_fft.__file__).resolve()),
              fft_core_sha256=hashlib.sha256(Path(original_fft.__file__).read_bytes()).hexdigest(),
              fft_normalization='original unnormalized DFT sum; no per-frame scaling',
              coordinates='cycles/nm in recorded scan axes; no lab-axis rotation',width_nm=size_nm[0],height_nm=size_nm[1])
    return Volume(energies[order],fft.qx,fft.qy,np.stack(result),meta)

def load_3ds(path, preprocess='plane', window='hann', progress=lambda x:None):
    path=Path(path)
    try:g=read_grid(path)
    except KeyError as exc:
        if str(exc.args[0]) not in ('Bias Spectroscopy>Sweep Start (V)','Bias Spectroscopy>Sweep End (V)'):raise
        g=read_recorded_bias_grid(path)
    return build_volume(np.moveaxis(g['lix'].astype(float),-1,0)*1e12,g['bias']*1000,
                        (g['settings'][2]*1e9,g['settings'][3]*1e9),
                        {'source':str(path),'source_type':'3ds','title':path.name,'channel':'LI Demod 1 X','input_unit':'pA',
                         'source_sha256':g['sha256'],'scan_angle_deg':g['settings'][4],
                         'comment':g['fields'].get('Comment',''),
                         'bias_axis_source':g.get('bias_axis_source','verified header endpoints')},preprocess,window,progress)

def read_recorded_bias_grid(path):
    """Adapter for minimal Linear headers with a fully recorded Bias (V) channel.

    Original exporter is unchanged. Here the bias channel is used verbatim;
    identical axes at every spatial pixel and linear spacing are required.
    """
    import re
    raw=Path(path).read_bytes();marker=b':HEADER_END:';end=raw.index(marker)+len(marker);offset=end
    for ending in (b'\r\r\n',b'\r\n',b'\n'):
        if raw[offset:offset+len(ending)]==ending:offset+=len(ending);break
    else:raise ValueError('Unsupported 3DS header line ending.')
    fields={}
    for line in raw[:end].decode('latin1').splitlines():
        if '=' in line:
            k,value=line.split('=',1);fields[k.strip()]=value.strip().strip('"')
    nx,ny=map(int,re.fullmatch(r'(\d+) x (\d+)',fields['Grid dim']).groups())
    npts=int(fields['Points']);npar=int(fields['# Parameters (4 byte)'])
    channels=fields['Channels'].split(';');params=fields['Fixed parameters'].split(';')+fields['Experiment parameters'].split(';')
    if fields['Filetype'].lower()!='linear' or fields['Sweep Signal']!='Bias (V)' or npts<2 or min(nx,ny)<2:
        raise ValueError('Only complete Linear Bias (V) grids are supported.')
    if len(params)!=npar or len(set(channels))!=len(channels):raise ValueError('3DS channel/parameter count mismatch.')
    stride=npar+len(channels)*npts;payload=raw[offset:]
    if len(payload)!=nx*ny*stride*4 or int(fields['Experiment size (bytes)'])!=len(channels)*npts*4:
        raise ValueError('Incomplete 3DS payload; no filling is allowed.')
    records=np.frombuffer(payload,dtype='>f4').reshape(ny,nx,stride)
    def channel(name):
        i=channels.index(name);return records[:,:,npar+i*npts:npar+(i+1)*npts]
    biases=channel('Bias (V)');bias=biases[0,0].astype(float)
    if not np.all(np.isfinite(biases)) or not np.all(biases==biases[0,0]):raise ValueError('Recorded bias axis differs between spatial pixels.')
    if not (np.all(np.diff(bias)>0) or np.all(np.diff(bias)<0)) or not np.allclose(bias,np.linspace(bias[0],bias[-1],npts),rtol=1e-6,atol=1e-9):
        raise ValueError('Recorded bias axis is not a monotonic linear sweep.')
    endpoints=records[:,:,[params.index('Sweep Start'),params.index('Sweep End')]]
    if not np.allclose(endpoints,[bias[0],bias[-1]],rtol=1e-6,atol=1e-9):raise ValueError('Sweep parameters and recorded bias disagree.')
    lix=channel('LI Demod 1 X (A)');z=records[:,:,params.index('Z (m)')]
    if not np.all(np.isfinite(lix)) or not np.all(np.isfinite(z)):raise ValueError('Invalid pixels; no automatic filling.')
    settings=tuple(map(float,fields['Grid settings'].split(';')))
    if len(settings)!=5 or not np.all(np.isfinite(settings)) or min(settings[2:4])<=0:raise ValueError('Invalid 3DS geometry.')
    x=records[:,:,params.index('X (m)')].astype(float);y=records[:,:,params.index('Y (m)')].astype(float)
    theta=np.deg2rad(settings[4]);u=x*np.cos(theta)+y*np.sin(theta);v=-x*np.sin(theta)+y*np.cos(theta)
    if not np.all(np.isfinite(x+y)) or not np.all(np.diff(u,axis=1)>0) or not np.all(np.diff(v,axis=0)>0):raise ValueError('Unexpected spatial ordering.')
    return dict(lix=lix,z=z,bias=bias,settings=settings,fields=fields,sha256=hashlib.sha256(raw).hexdigest(),bias_axis_source='recorded Bias (V) channel, identical at every pixel; no snapping')

def load_manifest(path, group, preprocess='plane', window='hann', progress=lambda x:None):
    path=Path(path)
    with path.open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['source_3ds_title']==group and r['image_kind']=='LI_X']
    if not rows: raise ValueError('该组没有 LI X 能量图像。')
    images=[]; energies=[]; hashes=[];size=None;geometry=None
    for r in rows:
        p=path.parent/r['filename'];s=read_sxm(p)
        if len(s.channels)!=1 or s.channels[0].unit!='A' or s.channels[0].direction!='forward':
            raise ValueError('只接受同一来源的单通道 forward LI X SXM。')
        if int(r['missing_pixels'])!=0:raise ValueError('存在缺失像素。')
        current_geometry={key:s.header_fields.get(key,[]) for key in ('SCAN_ANGLE','SCAN_DIR','SCAN_OFFSET','SCAN_PIXELS','SCAN_RANGE')}
        if geometry is not None and geometry!=current_geometry:raise ValueError('组内扫描几何或方向不同。')
        geometry=current_geometry;size=s.scan_range_nm
        if size is None:raise ValueError('缺少扫描尺寸。')
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if r.get('sxm_sha256') and digest!=r['sxm_sha256']:raise ValueError('SXM 校验值与索引不一致：'+p.name)
        hashes.append({'file':str(p),'sha256':digest})
        images.append(s.channels[0].data*1e12);energies.append(float(r['bias_mV']))
    return build_volume(images,energies,size,{'source':str(path),'group':group,'source_type':'manifest','title':group,
                       'channel':'LI Demod 1 X','input_unit':'pA','source_files':hashes,'scan_geometry':geometry},preprocess,window,progress)

def save_volume(path, volume):
    np.savez_compressed(path,energy_mV=volume.energy,qx=volume.qx,qy=volume.qy,amplitude=volume.amplitude,meta=json.dumps(volume.meta,ensure_ascii=False))

def load_volume(path):
    with np.load(path,allow_pickle=False) as z:
        return Volume(z['energy_mV'],z['qx'],z['qy'],z['amplitude'],json.loads(str(z['meta'])))

def export_cut(folder, cut, volume, contrast, project):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    metadata={'source':volume.meta,'cut':cut.params,'contrast':asdict(contrast),'project':project,
              'raw_values':'Amplitude in pA DFT sum or power in pA^2; not display-normalized.',
              'axes':{'x':cut.xlabel,'y':cut.ylabel},'missing':'NaN means outside acquired volume; no extrapolation',
              'energy_convention':'recorded sample bias in mV; no sign inversion or physical band assignment'}
    (folder/'parameters.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(folder/'section.npz',values=cut.values,x=cut.x,y=cut.y,**cut.extra)
    with (folder/'section.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow([cut.ylabel+' / '+cut.xlabel,*cut.x]);w.writerows(np.column_stack((cut.y,cut.values)))
    if 'path_qx' in cut.extra:
        with (folder/'path_coordinates.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow([cut.xlabel,'qx_cycles_nm','qy_cycles_nm']);w.writerows(zip(cut.x,cut.extra['path_qx'],cut.extra['path_qy']))
    return folder

def render_cut_png(path, cut, volume, contrast):
    """Export a scientific figure using the same global display mapping as the UI."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import PowerNorm
    def edge(a):return np.r_[a[0]-(a[1]-a[0])/2,(a[:-1]+a[1:])/2,a[-1]+(a[-1]-a[-2])/2]
    c=contrast;fig=Figure(figsize=(10,6),dpi=160,layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
    norm=PowerNorm(gamma=1/c.gamma,vmin=c.low,vmax=c.high,clip=True)
    im=ax.pcolormesh(edge(cut.x),edge(cut.y),c.transform(cut.values),cmap=c.cmap,norm=norm,shading='flat')
    ax.set_xlabel(cut.xlabel);ax.set_ylabel(cut.ylabel);ax.set_title(volume.meta['title']+' · '+('constant-energy q-q map' if cut.params['mode']=='energy' else 'q-E scattering section (not a direct band)'))
    if cut.params['mode'] in ('energy','tilted'):ax.set_aspect('equal')
    if cut.params['mode']=='path':
        for j,s in enumerate(cut.params['corner_distances']):
            ax.axvline(s,color='#ee9922',lw=.7)
            ax.text(s,.985,f' P{j}',transform=ax.get_xaxis_transform(),va='top',color='#b35d00',fontsize=10,
                    bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1})
    fig.colorbar(im,ax=ax,label=('log10 ' if c.logarithmic else '')+('FFT amplitude (pA)' if cut.params['metric']=='amplitude' else 'FFT power (pA²)'))
    fig.savefig(path)
