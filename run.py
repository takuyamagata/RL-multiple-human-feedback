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
L = np.ones(len(C)) * 2e-1
env_size = 'small'         # environment size ('small' or 'medium')
trial_count = 50           # number of learning trial
episode_count = 1500       # number of episodes for a single learning trial
update_Cest_interval = 10
alpha = 9
beta = 1

"""
No feedback
"""
# main_oracle.main(
#          algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
#                                         # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
#                                         # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
#          simInfo='_no_feedback',        # Filename header
#          env_size = env_size,           # environment size ('small' or 'medium')
#          trial_count = trial_count,     # number of learning trial
#          episode_count = episode_count, # number of episodes for a single learning trial
#          L  = np.zeros(1),              # probability to give a feedback
#          C  = np.ones(1),               # Human feedback confidence level)
#          prior_alpha  = 9,
#          prior_beta   = 1,
#          no_reward = False,          # True: learns without reward
#          C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
#          active_feedback_type=None, #'count',
#          update_Cest_interval=update_Cest_interval,
#          )

"""
Random timing feedback
"""
# main_oracle.main(
#          algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
#                                         # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
#                                         # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
#          simInfo='_C=p9p8_Tr2_L=p2_a=9_b=1_ALrandom', # Filename header
#          env_size = env_size,           # environment size ('small' or 'medium')
#          trial_count = trial_count,     # number of learning trial
#          episode_count = episode_count, # number of episodes for a single learning trial
#          L  = L,                        # probability to give a feedback
#          C  = C,                        # Human feedback confidence level)
#          prior_alpha  = 9,
#          prior_beta   = 1,
#          no_reward = False,          # True: learns without reward
#          C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
#          active_feedback_type='random', #'count',
#          update_Cest_interval=update_Cest_interval,
#          )

"""
Count based active feedback
"""
main_oracle.main(
         algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
                                        # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
                                        # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         simInfo='_C=p9p8_Tr2_L=p2_a=9_b=1_ALcount', # Filename header
         env_size = env_size,           # environment size ('small' or 'medium')
         trial_count = trial_count,     # number of learning trial
         episode_count = episode_count, # number of episodes for a single learning trial
         L  = L,                        # probability to give a feedback
         C  = C,                        # Human feedback confidence level)
         prior_alpha  = 9,
         prior_beta   = 1,
         no_reward = False,          # True: learns without reward
         C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
         active_feedback_type='count',
         update_Cest_interval=update_Cest_interval,
         )

main_oracle.main(
         algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
                                        # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
                                        # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         simInfo='_C=p9p8_Tr2_L=p2_a=9_b=1_ALcount_last3', # Filename header
         env_size = env_size,           # environment size ('small' or 'medium')
         trial_count = trial_count,     # number of learning trial
         episode_count = episode_count, # number of episodes for a single learning trial
         L  = L,                        # probability to give a feedback
         C  = C,                        # Human feedback confidence level)
         prior_alpha  = 9,
         prior_beta   = 1,
         no_reward = False,          # True: learns without reward
         C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
         active_feedback_type='count_last3',
         update_Cest_interval=update_Cest_interval,
         )

main_oracle.main(
         algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
                                        # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
                                        # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         simInfo='_C=p9p8_Tr2_L=p2_a=9_b=1_ALvalue', # Filename header
         env_size = env_size,           # environment size ('small' or 'medium')
         trial_count = trial_count,     # number of learning trial
         episode_count = episode_count, # number of episodes for a single learning trial
         L  = L,                        # probability to give a feedback
         C  = C,                        # Human feedback confidence level)
         prior_alpha  = 9,
         prior_beta   = 1,
         no_reward = False,          # True: learns without reward
         C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
         active_feedback_type='value',
         update_Cest_interval=update_Cest_interval,
         )

main_oracle.main(
         algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
                                        # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
                                        # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         simInfo='_C=p9p8_Tr2_L=p2_a=9_b=1_ALvalue_last3', # Filename header
         env_size = env_size,           # environment size ('small' or 'medium')
         trial_count = trial_count,     # number of learning trial
         episode_count = episode_count, # number of episodes for a single learning trial
         L  = L,                        # probability to give a feedback
         C  = C,                        # Human feedback confidence level)
         prior_alpha  = 9,
         prior_beta   = 1,
         no_reward = False,          # True: learns without reward
         C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
         active_feedback_type='value_last3',
         update_Cest_interval=update_Cest_interval,
         )

main_oracle.main(
         algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
                                        # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
                                        # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         simInfo='_C=p9p8_Tr2_L=p2_a=9_b=1_ALideal', # Filename header
         env_size = env_size,           # environment size ('small' or 'medium')
         trial_count = trial_count,     # number of learning trial
         episode_count = episode_count, # number of episodes for a single learning trial
         L  = L,                        # probability to give a feedback
         C  = C,                        # Human feedback confidence level)
         prior_alpha  = 9,
         prior_beta   = 1,
         no_reward = False,          # True: learns without reward
         C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
         active_feedback_type='ideal',
         update_Cest_interval=update_Cest_interval,
         )


"""
No Reward
"""

# main_oracle.main(
#          algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
#                                         # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
#                                         # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
#          simInfo='_C=p9p8_Tr2_L=p2_a=9_b=1_ALrandom_noR', # Filename header
#          env_size = env_size,           # environment size ('small' or 'medium')
#          trial_count = trial_count,     # number of learning trial
#          episode_count = episode_count, # number of episodes for a single learning trial
#          L  = L,                        # probability to give a feedback
#          C  = C,                        # Human feedback confidence level)
#          prior_alpha  = 9,
#          prior_beta   = 1,
#          no_reward = True,          # True: learns without reward
#          C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
#          active_feedback_type='random', #'count',
#          update_Cest_interval=update_Cest_interval,
#          )

main_oracle.main(
         algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
                                        # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
                                        # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         simInfo='_C=p9p8_Tr2_L=p2_a=9_b=1_ALcount_noR', # Filename header
         env_size = env_size,           # environment size ('small' or 'medium')
         trial_count = trial_count,     # number of learning trial
         episode_count = episode_count, # number of episodes for a single learning trial
         L  = L,                        # probability to give a feedback
         C  = C,                        # Human feedback confidence level)
         prior_alpha  = 9,
         prior_beta   = 1,
         no_reward = True,          # True: learns without reward
         C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
         active_feedback_type='count',
         update_Cest_interval=update_Cest_interval,
         )

main_oracle.main(
         algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
                                        # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
                                        # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         simInfo='_C=p9p8_Tr2_L=p2_a=9_b=1_ALcount_last3_noR', # Filename header
         env_size = env_size,           # environment size ('small' or 'medium')
         trial_count = trial_count,     # number of learning trial
         episode_count = episode_count, # number of episodes for a single learning trial
         L  = L,                        # probability to give a feedback
         C  = C,                        # Human feedback confidence level)
         prior_alpha  = 9,
         prior_beta   = 1,
         no_reward = True,          # True: learns without reward
         C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
         active_feedback_type='count_last3',
         update_Cest_interval=update_Cest_interval,
         )

main_oracle.main(
         algID   = 'tabQL_Cest_vi_t2',  # 'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2'
                                        # 'tabQL_Cest_em_t1', 'tabQL_Cest_em_t2'
                                        # 'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         simInfo='_C=p9p8_Tr2_L=p2_a=9_b=1_ALideal_noR', # Filename header
         env_size = env_size,           # environment size ('small' or 'medium')
         trial_count = trial_count,     # number of learning trial
         episode_count = episode_count, # number of episodes for a single learning trial
         L  = L,                        # probability to give a feedback
         C  = C,                        # Human feedback confidence level)
         prior_alpha  = 9,
         prior_beta   = 1,
         no_reward = True,          # True: learns without reward
         C_fixed = None,             # np.array(num_trainers): fix C value, None: learns C value
         active_feedback_type='ideal',
         update_Cest_interval=update_Cest_interval,
         )

