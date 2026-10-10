"""Regenerate README walnut comparisons from archived upstream arrays.

No reconstruction or PSNR is recomputed. Uses the same grayscale limits and
2-row composition as examples/_common.py. TV is the regularizer, Adam the solver.
Run: uv run --no-project --with matplotlib python docs/assets/make_reconstruction_figures.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).parent
DATA=HERE.parent/'video'


def figure(filename, data, panels, title):
    background,ink,muted='#111418','#ECEAE6','#9AA3AD'
    fig,axes=plt.subplots(2,len(panels),figsize=(2.7*len(panels),6.3),squeeze=False,facecolor=background)
    for column,(key,name,detail) in enumerate(panels):
        for row,section in enumerate(('axial','coronal')):
            ax=axes[row,column]
            ax.imshow(data[f'{key}_{section}'],cmap='gray',vmin=0,vmax=1)
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if column==0:ax.set_ylabel(section,color=muted,fontsize=10)
        axes[0,column].set_title(name,color=ink,fontsize=11,pad=16 if detail else 6)
        if detail:
            axes[0,column].text(.5,1.02,detail,transform=axes[0,column].transAxes,
                               ha='center',va='bottom',color=muted,fontsize=9)
    fig.suptitle(title,color=ink,fontsize=12,x=.012,ha='left')
    fig.tight_layout(rect=(0,0,1,.97),w_pad=.4,h_pad=.4)
    fig.savefig(HERE/filename,dpi=200,facecolor=background)
    plt.close(fig)
    print('saved',filename)


def main():
    measured=np.load(DATA/'walnut_measured.npz')
    figure('walnut_measured.png',measured,[('fdk','FDK','Hann window'),
        ('sirt','SIRT','200 iterations'),('cgls','CGLS','20 iterations'),
        ('tv','TV-regularized','Adam, 300 it; '+r'$\lambda=0.3$')],
        'Measured walnut: 240 views, circular cone beam, 256³')
    simulated=np.load(DATA/'recon_slices.npz')
    psnr=json.loads((DATA/'recon_psnr.json').read_text())
    panels=[('phantom','Ground truth','')]
    for key,name,steps in [('fdk','FDK',None),('cgls','CGLS',30),('sirt','SIRT',200),('tv','TV-regularized',200)]:
        detail=f'{psnr[key]:.2f} dB'
        if steps:detail=f'{steps} it; '+detail
        if key=='tv':detail='Adam, '+detail
        panels.append((key,name,detail))
    figure('walnut_helical.png',simulated,panels,'Simulated helical walnut scan: 720 views, 256³, 1% noise')


if __name__=='__main__':main()
