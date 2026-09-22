"""Reproduce descriptive continuity figures from retained, sanitized measurements.

One correlated trajectory per arm. No independent-sample error bars or GPU work.
Dependencies: matplotlib, numpy.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    data=json.loads((ROOT/'evidence/continuity-ablation.json').read_text())
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':'#172c3a','text.color':'#172c3a',
        'axes.edgecolor':'#a5b4c0','grid.color':'#e0e5e9','svg.fonttype':'none',
        'svg.hashsalt':'h3-continuity-20260922'})
    arms=['off','video','av'];labels=['Unprotected','Video prefix','AV prefix'];colors=['#c56934','#2369a8','#16887b']
    seqs=[[r['seam_mae_255'] for r in data['arms'][k]['rows'] if 'seam_mae_255' in r] for k in arms]
    assert all(len(x)==59 for x in seqs)
    fig,(a,b)=plt.subplots(1,2,figsize=(10.8,4.6),layout='constrained')
    bp=a.boxplot(seqs,positions=[1,2,3],widths=.47,patch_artist=True,showfliers=False,
                 medianprops={'color':'#111','linewidth':1.7})
    for i,(ys,color,box) in enumerate(zip(seqs,colors,bp['boxes'])):
        box.set(facecolor=color,alpha=.18)
        # Deterministic horizontal offsets for visibility; not uncertainty.
        x=(i+1)+.1*np.sin(np.arange(len(ys))*2.4)
        a.scatter(x,ys,s=13,alpha=.48,color=color,zorder=3,edgecolors='none')
    a.set(xticks=[1,2,3],xticklabels=labels,ylabel='Adjacent-frame RGB MAE (0–255 scale)')
    a.set_title('(a) Boundary distributions, n = 59 per arm',fontsize=11)
    a.grid(axis='y',alpha=.7);a.set_axisbelow(True)
    for ys,label,color in zip(seqs,labels,colors):
        b.step(np.sort(ys),np.arange(1,len(ys)+1)/len(ys),where='post',label=label,color=color,linewidth=1.7)
    b.set(xlabel='Adjacent-frame RGB MAE (0–255 scale)',ylabel='Empirical cumulative proportion',ylim=(0,1.04),xlim=(0,None))
    b.set_title('(b) Empirical CDF of all boundaries',fontsize=11)
    b.grid(alpha=.5);b.legend(loc='lower right',fontsize=9)
    fig.suptitle('Prefix protection: descriptive statistics from one trajectory per arm',fontsize=12)
    for ext in ['svg','png']:fig.savefig(figdir/('continuity-seams.'+ext),dpi=180,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10.8,6.0),sharex=True,sharey=True,layout='constrained')
    for a,key,label,color in zip(axes,arms,labels,colors):
        rows=data['arms'][key]['rows'];summary=data['arms'][key]['summary']
        a.plot([r['ordinal']+1 for r in rows],[r['max_top_black_rows'] for r in rows],color=color,marker='o',markersize=3,linewidth=1.3)
        a.axhline(3,color='#777',linestyle=':',linewidth=1)
        a.text(.015,.82,f"{label}: {summary['clips_with_sampled_band_over_3_rows']}/60 clips > 3 rows",transform=a.transAxes,fontsize=10)
        a.set(ylabel='Dark rows',ylim=(-1.5,26));a.grid(axis='y',alpha=.6)
    axes[-1].set(xlabel='Generated clip index',xlim=(.5,60.5),xticks=[1,10,20,30,40,50,60])
    fig.suptitle('Top-border behavior: maximum over three sampled frames per clip',fontsize=12)
    for ext in ['svg','png']:fig.savefig(figdir/('continuity-black-bands.'+ext),dpi=180,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    for name in ['continuity-seams.svg', 'continuity-black-bands.svg']:
        path=figdir/name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print('Built two continuity figures from 180 clips / 177 boundaries; CPU only.')

if __name__=='__main__':main()
