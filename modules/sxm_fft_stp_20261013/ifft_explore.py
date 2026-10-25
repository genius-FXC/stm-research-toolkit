"""Read-only source survey for phase-preserving Bragg iFFT trials."""
from pathlib import Path
import csv, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import maximum_filter
from fft_stp_batch import read_sxm, plane_and_fft, SOURCE, HERE
sys.path.insert(0, str(HERE.parent / 'stm_line_spectrum_tool'))
from core import PALETTE

OUT = HERE / 'ifft_trials'
OUT.mkdir(exist_ok=True)
plt.rcParams['font.family'] = 'Microsoft YaHei'
cmap = ListedColormap(PALETTE / 255.)
with (SOURCE / 'index_and_bias.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
topos = [r for r in rows if r['image_kind'] == 'Topo_Z']
fig, axes = plt.subplots(len(topos), 2, figsize=(8, 3*len(topos)), layout='constrained')
info = []
for row, ax in zip(topos, axes):
    s = read_sxm(SOURCE / row['filename'])
    r = plane_and_fft(s.channels[0].data, *s.scan_range, s.channels[0].unit)
    raw = r['residual'][::-1, ::-1]*1e12
    lo, hi = np.percentile(raw, [2,98])
    ax[0].imshow(raw, cmap=cmap, vmin=lo, vmax=hi)
    ax[0].set_title(row['filename']+' | '+row['source_3ds_title']+' | topo')
    a, kx, ky = r['canonical_fft'], r['kx'], r['ky']
    xx, yy = np.meshgrid(kx,ky)
    roi = (abs(xx)<6)&(abs(yy)<6)&(np.hypot(xx,yy)>.8)
    v = np.percentile(a[roi],98.5)
    dx, dy = kx[1]-kx[0],ky[0]-ky[1]
    ax[1].imshow(a,cmap=cmap,vmin=0,vmax=v,extent=[kx[0]-dx/2,kx[-1]+dx/2,ky[-1]-dy/2,ky[0]+dy/2])
    ax[1].set(xlim=(-6,6),ylim=(-6,6),xlabel='kx (1/nm)',ylabel='ky (1/nm)')
    peaks = (a==maximum_filter(a, size=3))&roi
    inds = np.argwhere(peaks)
    inds = sorted(inds,key=lambda p:a[tuple(p)],reverse=True)[:30]
    info.append(dict(filename=row['filename'],source=row['source_3ds_title'],size_nm=np.array(s.scan_range).tolist(),df=[dx,dy],peaks=[dict(kx=float(kx[j]),ky=float(ky[i]),amplitude=float(a[i,j])) for i,j in inds]))
fig.savefig(OUT / 'topo_survey.jpg',dpi=110)
plt.close(fig)
(OUT / 'topo_peaks.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
print('Survey:', OUT / 'topo_survey.jpg')
