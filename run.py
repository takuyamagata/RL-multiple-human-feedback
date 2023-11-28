# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:33:08 2018

@author: taku.yamagata
"""
import numpy as np
import main_oracle

main_oracle.main(
         algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
                                        # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
                                        # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         simInfo='_C=p2p9_Tr15_L=p01_a=100_b=100', # Filename header
         env_size = 'small',         # environment size ('small' or 'medium')
         trial_count = 5,          # number of learning trial
         episode_count = 2000,       # number of episodes for a single learning trial
         L  = np.ones(15) * 1e-2,    # probability to give a feedback
         C  = np.array([0.9,  0.9,  0.8,  0.8,  0.7,  0.7,  0.6,  0.6,  0.5,  0.4,  0.4,  0.3,  0.3,  0.2,  0.2]),      # Human feedback confidence level)
         a  = 100,
         b  = 100,
         no_reward = False,          # True: learns without reward
         C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
         )
