from pathlib import Path
import numpy as np
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
a=np.load(HERE.parent/'20261011_Se_rows070_076_082_088_094/results/Se_rows_frequency_energy.npz')
xy=a['Se_calibrated_xy_px'].reshape(5,11,2)
z=a['topo_plane_subtracted_pm']
step=float(np.diff(a['distance_nm']).mean())
q=float(a['q_third_cycles_per_nm'])
def refine(rad):
    out=np.full_like(xy,np.nan)
    for r in range(5):
        for c in range(11):
            px,py=xy[r,c]; ix,iy=np.rint([px,py]).astype(int)
            if ix-rad<0 or ix+rad>=99 or iy-rad<0 or iy+rad>=99: continue
            yy,xx=np.mgrid[-rad:rad+1,-rad:rad+1]
            D=np.column_stack([np.ones(xx.size),xx.ravel(),yy.ravel(),xx.ravel()**2,xx.ravel()*yy.ravel(),yy.ravel()**2])
            b=np.linalg.lstsq(D,z[iy-rad:iy+rad+1,ix-rad:ix+rad+1].ravel(),rcond=None)[0]
            H=np.array([[2*b[3],b[4]],[b[4],2*b[5]]])
            if np.max(np.linalg.eigvalsh(H))>=0: continue
            v=-np.linalg.solve(H,b[1:3]); pos=v+[ix,iy]
            if np.linalg.norm(pos-[px,py])>2 or np.max(abs(v))>rad:continue
            out[r,c]=pos
    return out
sets={'Manual':xy,'Topo3x3':refine(1),'Topo5x5':refine(2)}
records=[];summary=[]
fig,axes=plt.subplots(5,2,figsize=(12,13),layout='constrained')
for name,p in sets.items():
    dd=np.linalg.norm(np.diff(p,axis=1),axis=2)*step
    xm=(p[:,:-1,0]+p[:,1:,0])/2*step
    for r in range(5):
        valid=np.isfinite(dd[r]); x=xm[r,valid]; y=dd[r,valid]*1000
        amp=phase=None
        if len(y)>=7:
            X=np.column_stack([np.ones(len(x)),x-x.mean(),np.cos(2*np.pi*q*x),np.sin(2*np.pi*q*x)])
            b=np.linalg.lstsq(X,y,rcond=None)[0];amp=float(np.hypot(*b[2:]));phase=float(np.arctan2(-b[3],b[2]))
        summary.append(dict(method=name,Se_row=r+11,valid_bonds=int(valid.sum()),mean_pm=float(np.nanmean(dd[r])*1000),sd_pm=float(np.nanstd(dd[r],ddof=1)*1000),min_pm=float(np.nanmin(dd[r])*1000),max_pm=float(np.nanmax(dd[r])*1000),fixed_q3_amplitude_pm=amp,phase_rad=phase))
        axes[r,0].plot(np.arange(1,11)+.5,dd[r]*1000,'o-',ms=3,label=name)
        axes[r,0].set(ylabel=f'Se row {r+11}\nSpacing (pm)',xlabel='Bond midpoint (atom column)')
        if len(y)>=7:
            fs=np.sort(np.r_[np.arange(1,5)/(10*np.nanmean(dd[r])),q]); amps=[]
            for f in fs:
                X=np.column_stack([np.ones(len(x)),x-x.mean(),np.cos(2*np.pi*f*x),np.sin(2*np.pi*f*x)])
                b=np.linalg.lstsq(X,y,rcond=None)[0];amps.append(np.hypot(*b[2:]))
            axes[r,1].plot(fs,amps,'o-',ms=3,label=name)
        for c in range(10): records.append([name,r+11,c+1,c+2,xm[r,c],dd[r,c]*1000])
for ax in axes[:,1]:
    ax.axvline(q,color='gray',ls='--',label='1/(3a) reference');ax.set(xlabel='Spatial frequency (cycles/nm)',ylabel='Sine-regression amplitude (pm)',ylim=(0,None))
axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
fig.suptitle('Dark Se rows: horizontal adjacent-site spacing\nExploratory: manual positions vs local topography quadratic centers; 10 bonds per row')
fig.savefig(HERE/'Se_spacing_check.png',dpi=170)
with (HERE/'Se_spacing_bonds.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['method','Se_row','left_col','right_col','midpoint_x_nm','spacing_pm']);w.writerows(records)
with (HERE/'Se_spacing_summary.json').open('w',encoding='utf-8') as f:json.dump(summary,f,indent=2)
np.savez(HERE/'Se_centers_check.npz',**sets)
print(json.dumps(summary,indent=2));print('mean shift between 3x3 and 5x5 (pm):',np.nanmean(np.linalg.norm(sets['Topo3x3']-sets['Topo5x5'],axis=2))*step*1000)
print('Manual vertical row-to-row spacing nm:',np.linalg.norm(np.diff(xy,axis=0),axis=2).mean(axis=1)*step)
print('Common-bond q3 amplitudes pm (Manual,Topo3x3,Topo5x5):')
for r in range(5):
    valid=np.ones(10,dtype=bool)
    for p in sets.values(): valid &= np.isfinite(np.diff(p[r],axis=0)).all(axis=1)
    vals=[]
    for p in sets.values():
        x=(p[r,:-1,0]+p[r,1:,0])[valid]/2*step
        y=np.linalg.norm(np.diff(p[r],axis=0),axis=1)[valid]*step*1000
        X=np.column_stack([np.ones(len(x)),x-x.mean(),np.cos(2*np.pi*q*x),np.sin(2*np.pi*q*x)])
        b=np.linalg.lstsq(X,y,rcond=None)[0];vals.append(round(float(np.hypot(*b[2:])),3))
    print(r+11,int(valid.sum()),vals)
