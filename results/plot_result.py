"""
Plot simulation results
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt

def plot_std(ax, fname, color=None):
    data = np.load(fname)
    x = np.arange(data['ave'].shape[1]) + 1
    num_data = data['ave'].shape[0]

    for n in range(num_data):
        if color is None:
            col = f'C{n}'
        else:
            col = color
        ax.plot(x, data['ave'][n], color=col)
        ax.fill_between(x, data['ave'][n]-data['std'][n], data['ave'][n]+data['std'][n], color=col, alpha=0.2)

    ax.set_xlabel('number of episodes')
    if 'aveRW' in fname:
        ax.set_ylabel('return (total reward)')
        ax.set_ylim(-600, 700)
    else:
        ax.set_ylabel('estimated consistency level')
        ax.set_ylim(0, 1)
        
    return True

def plot_pct(ax, fname, pct=50, color='C0'):
    data = np.load(fname)
    x = np.arange(data['ave'].shape[1]) + 1
    num_data = data['ave'].shape[0]
    p = np.percentile(data['raw'], pct, axis=2)
    n = 0
    ax.plot(x, p[n,:], color=color)

    ax.set_xlabel('number of episodes')
    if 'aveRW' in fname:
        ax.set_ylabel('return (total reward)')
        ax.set_ylim(-600, 700)
    else:
        ax.set_ylabel('estimated consistency level')
        ax.set_ylim(0, 1)
        
    return True



files = glob.glob('./results/aveRW*npz')
for fname in files:
    fig, ax = plt.subplots(1)
    plot_std(ax, fname)

    fig_fname, ext = os.path.splitext(fname)
    fig.savefig(fig_fname+'.pdf')
    

# fname = "results/aveRW_tabQL_Cest_em_org_t1_C=p2p9_Tr8_L=p2.npz"
# fig, ax = plt.subplots(1)
# plot_pct(ax, fname, 5)
# fig_fname, ext = os.path.splitext(fname)
# fig.savefig(fig_fname+'.pdf')


