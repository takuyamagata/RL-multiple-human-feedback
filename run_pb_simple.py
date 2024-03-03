# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:33:08 2018

@author: taku.yamagata
"""
import numpy as np
from simple_utility_learning_top import top

nTrials = 5 # 100
u = np.array([1.0, 1.1, 1.2, 0.7])
b = np.array([1.0, .5, 2.0])
b_fxd = False # b
nFB_st = 100
nFB_pt = 0
batch_size = 1000

top(
    util = u,
    beta = b,
    beta_fixed = b_fxd,
    lr = 2e-3,
    sgd_all = True,
    max_time_steps = 20000,
    num_feedback_at_start = nFB_st,       # number of preferences at the start
    num_feedback_per_timestep = nFB_pt, # number of preferences per iteration per trainer
    sgd_sample_size = batch_size,          # batch size for SGD
    num_iterations_per_timestep = 1,
    num_SGD_steps_per_iteration = 1,
    dir = 'results',
    num_trials = nTrials,
    fname_tailer = None
)

b_fxd = b
top(
    util = u,
    beta = b,
    beta_fixed = b_fxd,
    lr = 2e-3,
    sgd_all = True,
    max_time_steps = 20000,
    num_feedback_at_start = nFB_st,       # number of preferences at the start
    num_feedback_per_timestep = nFB_pt, # number of preferences per iteration per trainer
    sgd_sample_size = batch_size,          # batch size for SGD
    num_iterations_per_timestep = 1,
    num_SGD_steps_per_iteration = 1,
    dir = 'results',
    num_trials = nTrials,
    fname_tailer = None
)

b_fxd = np.array([1.0, 1.0, 1.0])
top(
    util = u,
    beta = b,
    beta_fixed = b_fxd,
    lr = 2e-3,
    sgd_all = True,
    max_time_steps = 20000,
    num_feedback_at_start = nFB_st,       # number of preferences at the start
    num_feedback_per_timestep = nFB_pt, # number of preferences per iteration per trainer
    sgd_sample_size = batch_size,          # batch size for SGD
    num_iterations_per_timestep = 1,
    num_SGD_steps_per_iteration = 1,
    dir = 'results',
    num_trials = nTrials,
    fname_tailer = None
)