"""
    Simple utility learning top level script

"""
import numpy as np
import matplotlib.pyplot as plt
import os, datetime, pickle
from simple_utility_learning import main

def ma(d, len): # Moving Average function
    if len==1 or len is None:
        return d
    prePadLen = len//2
    posPadLen = len - prePadLen
    d_ = np.append(d[0:prePadLen], d)
    d_ = np.append(d_, d[-posPadLen:])
    cs = np.cumsum(d_)
    ma = (cs[len:] - cs[:-len]) / len
    return ma

def plot_std(ax, data, color=None, label=None, ma_len=1):
    data = np.array(data)
    num_trials = data.shape[0]
    num_timesteps = data.shape[1]
    num_data = data.shape[2]

    x = np.arange(num_timesteps)
    for n in range(num_data):
        if color is None:
            col = f'C{n}'
        else:
            if hasattr(color, '__len__'):
                col = color[n]
            else:
                col = color
        if hasattr(label, '__len__'):
            lab = label[n]
        else:
            lab = label
        ave = np.mean(data[:,:,n], axis=0)
        std = np.std(data[:,:,n],  axis=0)
        ax.plot(x, ma(ave, ma_len), color=col, label=lab, alpha=0.7)
        ax.fill_between(x, ma(ave-std, ma_len), ma(ave+std, ma_len), color=col, alpha=0.2)        
    return


def top(
    util = np.log([1.0, 1.1, 1.2, 0.8]),
    beta = np.array([1.0, 1.5, 0.3]),
    lr = 2e-3,
    sgd_all = False,
    max_time_steps = 10000,
    num_feedback_per_timestep = 100, # number of preferences per iteration per trainer
    num_iterations_per_timestep = 1,
    num_SGD_steps_per_iteration = 1,
    dir = 'results',
    num_trials = 10,
    fname_tailer = None
    ):
    start_time = datetime.datetime.now()
    
    # generate file name
    alg = 'SGD' if sgd_all else 'SGD+MM'
    lr_str = f"{lr:1.0e}"
    u_str = str(np.exp(util)).replace(' ', '_')
    b_str = str(beta).replace(' ', '_')
    fname = f'{alg}_u{u_str}_b{b_str}_lr{lr_str}_nFb{num_feedback_per_timestep}_nIt{num_iterations_per_timestep}_nSGDs{num_SGD_steps_per_iteration}'    
    if fname_tailer is not None:
        fname = fname + fname_tailer
    
    print(f"start ({start_time}) :: {fname}")
    
    u_list = []
    b_list = []
    for n in range(num_trials):
        print(f'trials {n+1}/{num_trials}')
        u_, b_, _, _ = main(
                            util = util,
                            beta = beta,
                            lr = lr,
                            sgd_all = sgd_all,
                            max_time_steps = max_time_steps,
                            num_feedback_per_timestep = num_feedback_per_timestep,
                            num_iterations_per_timestep = num_iterations_per_timestep,
                            num_SGD_steps_per_iteration= num_SGD_steps_per_iteration,
                            )
        u_list.append(u_)
        b_list.append(b_)
        
    # save result
    data = {
        'util': util,
        'beta': beta,
        'u_list': u_list,
        'b_list': b_list,
        'lr': lr,
        'sgd_all': sgd_all,
        'max_time_steps': max_time_steps,
        'num_feedback_per_timestep' : num_feedback_per_timestep,
        'num_iterations_per_timestep': num_iterations_per_timestep,
        'num_SGD_steps_per_iteration': num_SGD_steps_per_iteration,
        'start_dt': str(start_time),
        'end_dt': str(datetime.datetime.now()),
    }
    
    with open(os.path.join(dir, f'simple_pb_data_{fname}.pkl'), 'wb') as fid:
        pickle.dump(data, fid)    

if __name__ == '__main__':
    top()