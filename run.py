# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:33:08 2018

@author: taku.yamagata
"""
import numpy as np
import main_oracle


# Setup common parameters
# C = np.array([0.9,  0.9,  0.8,  0.8,  0.7,  0.7,  0.6,  0.6,  0.5,  0.4,  0.4,  0.3,  0.3,  0.2,  0.2])
C = np.array([0.9,  0.8,])
L = np.ones(len(C)) * 5e-2
env_size = 'small'         # environment size ('small' or 'medium')
trial_count = 100          # number of learning trial
episode_count = 2000       # number of episodes for a single learning trial



# main_oracle.main(
#          algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
#                                         # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
#                                         # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
#          simInfo='_C=p9p8_Tr2_L=p01_a=40_b=10', # Filename header
#          env_size = env_size,           # environment size ('small' or 'medium')
#          trial_count = trial_count,     # number of learning trial
#          episode_count = episode_count, # number of episodes for a single learning trial
#          L  = L,                        # probability to give a feedback
#          C  = C,                        # Human feedback confidence level)
#          a  = 40,
#          b  = 10,
#          no_reward = False,          # True: learns without reward
#          C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
#          active_feedback_type=None, #'count',
#          )

main_oracle.main(
         algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
                                        # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
                                        # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         simInfo='_C=p9p8_Tr2_L=p01_a=40_b=10_ALcount', # Filename header
         env_size = env_size,           # environment size ('small' or 'medium')
         trial_count = trial_count,     # number of learning trial
         episode_count = episode_count, # number of episodes for a single learning trial
         L  = L,                        # probability to give a feedback
         C  = C,                        # Human feedback confidence level)
         a  = 40,
         b  = 10,
         no_reward = False,          # True: learns without reward
         C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
         active_feedback_type='count',
         )

# main_oracle.main(
#          algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
#                                         # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
#                                         # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
#          simInfo='_C=p9p8_Tr2_L=p01_a=40_b=10_noR', # Filename header
#          env_size = env_size,           # environment size ('small' or 'medium')
#          trial_count = trial_count,     # number of learning trial
#          episode_count = episode_count, # number of episodes for a single learning trial
#          L  = L,                        # probability to give a feedback
#          C  = C,                        # Human feedback confidence level)
#          a  = 40,
#          b  = 10,
#          no_reward = True,          # True: learns without reward
#          C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
#          active_feedback_type=None, #'count',
#          )

# main_oracle.main(
#          algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
#                                         # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
#                                         # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
#          simInfo='_C=p9p8_Tr2_L=p01_a=40_b=10_ALcount_noR', # Filename header
#          env_size = env_size,           # environment size ('small' or 'medium')
#          trial_count = trial_count,     # number of learning trial
#          episode_count = episode_count, # number of episodes for a single learning trial
#          L  = L,                        # probability to give a feedback
#          C  = C,                        # Human feedback confidence level)
#          a  = 40,
#          b  = 10,
#          no_reward = True,          # True: learns without reward
#          C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
#          active_feedback_type='count',
#          )

