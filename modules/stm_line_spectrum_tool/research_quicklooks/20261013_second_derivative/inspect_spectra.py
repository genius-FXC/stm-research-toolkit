from pathlib import Path
import sys,numpy as np
from scipy.signal import savgol_filter,find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from core import GridDataset
d=GridDataset(Path(__import__('os').environ['STM_DATA_FILE']))
E=d.bias;g=d.g;fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
samples=[('Dark mean',g[69:].mean((0,1))),('Se paths mean',g[[70,76,82,88,94]].mean((0,1))),('Fe paths mean',g[[73,79,85,91,97]].mean((0,1)))]
samples += [(f'x={x},y={y}',g[y,x]) for x,y in [(15,70),(50,70),(50,85),(50,94),(80,94),(20,79)]]
for ax,(name,s) in zip(axes.flat,samples):
    right=ax.twinx();ax.plot(E,s,color='gray');ax.set(xlim=(-40,40),title=name,ylabel='g (pA)');right.set_ylabel('D (pA/mV^2)')
    for w in [5,7,9]:
        D=savgol_filter(s,w,3,deriv=2,delta=2);right.plot(E,D,label=f'SG {w}');idx=find_peaks(-D)[0];idx=idx[(abs(E[idx])>=4)&(abs(E[idx])<=38)&(D[idx]<0)];print(name,w,E[idx].tolist())
    right.legend(fontsize=7)
fig.savefig(HERE/'representative_second_derivatives.png',dpi=140)
np.savez(HERE/'source_arrays.npz',g=g,E=E,z=d.z,transform=d.transform,sha256=d.hash)
