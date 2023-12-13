"""
Plot simulation results
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, UnivariateSpline, splrep, BSpline

def plot_std(ax, fname, color=None,master=False):
    data = np.load(fname)
    x = np.arange(data['ave'].shape[1]) + 1
    num_data = data['ave'].shape[0]

    for n in range(num_data):
        if color is None:
            col = f'C{n}'
        else:
            col = color 

        resline = ax.plot(x, data['ave'][n], color=col,label=fname)
        ax.fill_between(x, data['ave'][n]-data['std'][n], data['ave'][n]+data['std'][n], color=col, alpha=0.2)

    ax.set_xlabel('number of episodes')
    if 'aveRW' in fname:
        ax.set_ylabel('return (total reward)')
        ax.set_ylim(-600, 700)
    else:
        ax.set_ylabel('estimated consistency level')
        ax.set_ylim(0, 1)
    
    if master:
        ax.legend(loc='best',fontsize=7)
        resline[0].remove()
        # Increase the number of points for smoother interpolation
        x_new = np.linspace(min(x), max(x), 200)

        # Apply spline interpolation to smooth the line
        # spl = make_interp_spline(x, data['ave'][n], k=0)
        # y_smooth = spl(x_new)
        tck = splrep(x, data['ave'][n], s=200)
        y_smooth = BSpline(*tck)(x_new)
        resline = ax.plot(x_new, y_smooth, color=col,label=fname)
        # resline[0].set_alpha(0.1)
        
    else:
        ax.legend()
        ax.get_legend().remove()
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


m_fig,m_ax = plt.subplots(1)

files = glob.glob('./results/aveRW*window.npz')
colours = sns.color_palette("Set1",len(files))
for i,fname in enumerate(files):
    fig, ax = plt.subplots(1)
    plot_std(ax, fname)
    plot_std(m_ax, fname,color=colours[i],master=True)
    fig_fname, ext = os.path.splitext(fname)
    fig.savefig(fig_fname+'.pdf')

m_fig.savefig('./results/all_results.pdf')

# fname = "results/aveRW_tabQL_Cest_em_org_t1_C=p2p9_Tr8_L=p2.npz"
# fig, ax = plt.subplots(1)
# plot_pct(ax, fname, 5)
# fig_fname, ext = os.path.splitext(fname)
# fig.savefig(fig_fname+'.pdf')


