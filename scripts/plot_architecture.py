#!/usr/bin/env python3
"""Standalone figures from completed resource and compiler-composition runs."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"reports/architecture"


def main():
    resource=list(csv.DictReader((FOLDER/"resource/comparison.csv").open()))
    composition=list(csv.DictReader((FOLDER/"composition/comparison.csv").open()))
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8))
    presets=["flat3_128","bank4_packed128","bank4_padded128","bank4_padded192","bank4_lanes4_128"]
    values=[next(r for r in resource if r['model']=='linear_16x8' and r['frames']=='8' and r['preset']==p) for p in presets]
    labels=['3-read flat\n128B, 2 lanes','4 banks packed\n128B, 2 lanes','4 banks padded\n128B, 2 lanes','4 banks padded\n192B, 2 lanes','4 banks padded\n128B, 4 lanes']
    bars=axes[0].bar(range(5),[float(v['time_us']) for v in values],color=['#527898','#ce8c54','#438c76','#438c76','#8d7aaa'])
    for bar,value in zip(bars,values):
        axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{float(value['time_us']):.2f}\nconflict={value['conflict_cycles']}",ha='center',fontsize=8)
    axes[0].set_xticks(range(5),labels,fontsize=8)
    axes[0].set(ylabel='Cold batch modeled time (us)',title='M=8, K=16, batch=8: ports and layout',ylim=(0,94))
    x=np.arange(3);width=.34
    for offset,fused,color,label in ((-width/2,'False','#ce8c54','Separate Q/K/V'),(width/2,'True','#438c76','Fused QKV')):
        selected=[next(r for r in composition if r['model']=='qkv_16x8' and r['frames']=='8' and r['preset']==p and r['fused_qkv']==fused) for p in ('lite64','flat3_128','bank4_128')]
        heights=[float(r['with_1us_per_partition_us']) for r in selected]
        axes[1].bar(x+offset,heights,width=width,color=color,label=label)
        for pos,height in zip(x+offset,heights):axes[1].text(pos,height+3,f'{height:.1f}',ha='center',fontsize=8)
    axes[1].set_xticks(x,['1-read / 1 lane\n64B','3-read / 2 lanes\n128B','4 banks / 2 lanes\n128B'])
    axes[1].set(ylabel='Modeled time + 1us / partition (us)',title='QKV fusion depends on the hardware layout',ylim=(0,310))
    axes[1].legend(frameon=False,fontsize=8,loc='upper right')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Tiny-LPU: compiler and architecture co-design',fontsize=13)
    fig.text(.5,.015,'100MHz / 10MB/s scenario; registered memory reads, byte copies and conflicts counted. Not silicon timing.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94))
    fig.savefig(FOLDER/'architecture_tradeoffs.png',dpi=180)
    fig.savefig(FOLDER/'architecture_tradeoffs.svg')


if __name__=='__main__':main()
