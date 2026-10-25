from ifft_explore import rows, OUT, cmap, read_sxm, plane_and_fft, SOURCE
import json
import numpy as np
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
selected=['2026_00013.sxm','2026_00237.sxm','2026_00341.sxm']
fig,axes=plt.subplots(3,2,figsize=(11,14),layout='constrained')
bases={}
for filename,axs in zip(selected,axes):
    row=next(r for r in rows if r['filename']==filename)
    s=read_sxm(SOURCE/filename); r=plane_and_fft(s.channels[0].data,*s.scan_range,s.channels[0].unit)
    a,kx,ky=r['canonical_fft'],r['kx'],r['ky']; xx,yy=np.meshgrid(kx,ky)
    vectors=[]
    for cond in [(xx>2.2)&(xx<3.5)&(abs(yy)<.7),(yy>2.2)&(yy<3.5)&(abs(xx)<.7)]:
        i,j=np.unravel_index(np.argmax(np.where(cond,a,-1)),a.shape)
        vectors.append(np.array([kx[j],ky[i]]))
    bases[row['source_3ds_title']]=[v.tolist() for v in vectors]
    raw=r['residual'][::-1,::-1]*1e12
    axs[0].imshow(raw,cmap=cmap,vmin=np.percentile(raw,2),vmax=np.percentile(raw,98))
    axs[0].set_title(filename+' '+row['source_3ds_title'])
    roi=(abs(xx)<5)&(abs(yy)<5)&(np.hypot(xx,yy)>1)
    dx=kx[1]-kx[0]; dy=ky[0]-ky[1]
    axs[1].imshow(a,cmap=cmap,vmin=0,vmax=np.percentile(a[roi],99),extent=[kx[0]-dx/2,kx[-1]+dx/2,ky[-1]-dy/2,ky[0]+dy/2])
    axs[1].set(xlim=(-4.5,4.5),ylim=(-4.5,4.5),xlabel='kx (1/nm)',ylabel='ky (1/nm)')
    ga,gb=vectors
    for name,v in [('(1,0)',ga),('(0,1)',gb),('(1,1)',ga+gb),('(-1,1)',-ga+gb)]:
        for sign in [-1,1]:
            target=v*sign
            axs[1].plot(*target,'o',mfc='none',mec='cyan',ms=12)
        axs[1].text(*(v+.1),name,color='cyan',fontsize=10)
    print(filename,'basis',vectors)
fig.savefig(OUT/'selected_topos.png',dpi=150); plt.close(fig)
(OUT/'bases.json').write_text(json.dumps(bases,indent=2),encoding='utf-8')
results=[]
for row in rows:
    if row['source_3ds_title'] not in bases or row['image_kind']!='LI_X': continue
    s=read_sxm(SOURCE/row['filename']);r=plane_and_fft(s.channels[0].data,*s.scan_range,s.channels[0].unit)
    a,kx,ky=r['canonical_fft'],r['kx'],r['ky']; xx,yy=np.meshgrid(kx,ky)
    ga,gb=map(np.array,bases[row['source_3ds_title']])
    metrics=[]
    for v in [ga,gb,ga+gb,-ga+gb]:
        dist=np.hypot(xx-v[0],yy-v[1]);peak=a[dist<.22].max(); background=np.median(a[(dist>.35)&(dist<.7)])
        metrics.append(float(peak/max(background,1e-30)))
    results.append(dict(filename=row['filename'],source=row['source_3ds_title'],bias_mV=row['bias_mV'],ratios=metrics,score=float(np.prod(metrics)**.25)))
results.sort(key=lambda r:r['score'],reverse=True)
(OUT/'mapping_candidates.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('Best mappings:',json.dumps(results[:15]))
