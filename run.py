# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:33:08 2018

@author: taku.yamagata
"""
import numpy as np
import main_oracle

main_oracle.main(
         algID   = 'tabQL_Cest_em_t2',
         simInfo='_C=p8_Tr8_L=p2', # Filename header
         trial_count = 100,          # number of learning trial
         episode_count = 2000,       # number of episodes for a single learning trial
         L  = np.array([0.2]),     # probability to give a feedback
         C  = np.array([0.8])      # Human feedback confidence level)
         )