import numpy as onp
import jax
import jax.numpy as np
from functools import partial
# from config import parameters, environment, rl, setup
import config
import library.utilities as ut


# @profile
# @partial(jax.jit,static_argnums = (9,))
# @jax.jit
def tabQL_Cest_em(Q, hp, hm, Ce, prev_obs, prev_act, obs, rw, done, update_Cest = False):
    """
    Tabular Q-learning with Policy shaping
    """
    # Boltzmann exploration policy
    l_pr = Q[obs,:]/config.parameters['tempConst']
    d = hp - hm
    
    # policy shaping
    # d = hp - hm # for VI need sum of pos and sum of neg
    
    
        # type2 (only one optimal action)
    mask = np.array([ut.mask_array(marray) for marray in d[:,obs]])
    l_pr += d[:,obs]*np.log(Ce)+ np.sum(mask*np.log(1-Ce))

    pr = np.exp(l_pr - ut.logsum(l_pr))
    action = np.argmax(pr)
    
    # check if this is the first time...
    if prev_obs is not None:
        # one step TD algorithm
        td_err = rw + config.parameters['gamma'] * Q[obs, action] * (1-done) - Q[prev_obs, prev_act]
        Q = Q.at[prev_obs, prev_act].set(Q[prev_obs, prev_act] + config.parameters['alpha'] * td_err)

        # Human feedback updates
        # d = d.at[np.arange(config.rl['nTrainer']), prev_obs, prev_act].set(d[np.arange(config.rl['nTrainer']), prev_obs, prev_act] + (2*fb - 1))
        # TODO: modify for non-binary case
        
    # action = np.array(list(environment['actions'].values())[np.argmax(Q[obs,:])])
    pr =np.argmax(Q[obs,:])
    if update_Cest:
        Ce = Cest(Q,hp,hm,Ce)
    # decide action based on pr[] probability distribution
    return pr, Q, hp, hm, Ce


# @partial(jax.jit,static_argnums = (5,))



# @jax.jit
def Cest(Q,hp,hm,Ce):
    """
    consistency level (C) estimation
    The consistency level estimation is based on EM algorithm 
    """
    # Update C estimations    
    d = hp - hm

    # Check if any element in each row is non-zero along axis 1
    valid_trainers,valid_states = np.where(np.any(d != 0,axis=2))
    valid_states = np.unique(valid_states)
    valid_trainers = np.unique(valid_trainers)

    # prepare piror of Optimality flag O (P1/P0) from Q-learning
    # given S,a pairs - if optimal ==1 , else ==0
    ln_pr = Q/config.parameters['tempConst']
    max_ln_pr = np.amax(ln_pr,axis=1).reshape(-1,1)
    log_sum_exp = np.log(np.sum(np.exp(ln_pr - max_ln_pr), axis=1, keepdims=True))
    ln_P_Q1 = ln_pr - (max_ln_pr + log_sum_exp)
    ln_P_Q0 = np.log(1.0 - np.exp(ln_P_Q1))
    
    for k in range(20): # EM iteration
        Ce_old = Ce
        Ce = ut.calcCe(ln_P_Q1[valid_states,:],
                              d[:,valid_states,:],
                              hp[valid_trainers][:,valid_states],
                              hm[valid_trainers][:,valid_states],
                              Ce)
        
        if np.max(np.abs(Ce - Ce_old)) < 1e-3:
            break; # if Ce does not change much, stop EM iterations

    # set the new Ce (avoid 0.0 and 1.0)
    # agent.Ce = np.clip(Ce, 0.001, 0.999)
    
    return Ce

 # Tabular one step Temporal Difference

# @profile
# @partial(jax.jit,static_argnums = (9,))
@jax.jit
def tabQL_greedy(Q, hp, hm, Ce, prev_obs, prev_act, obs, rw, done, update_Cest=None):
    # check if this is the first time...
    # if prev_obs is not None:
        # one step TD algorithm
    td_err = (rw + config.parameters['gamma'] * np.max(Q[obs])) - Q[prev_obs, prev_act] 
    Q = Q.at[prev_obs, prev_act].set(Q[prev_obs, prev_act] + config.parameters['alpha'] * td_err)
    
    # Greedy policy (always select the best action)
    pr =np.argmax(Q[obs,:])
    # action = np.array(list(environment['actions'].values())[pr])
    
    return pr, Q, hp, hm, Ce 

algorithms = {
    'tabQL_greedy':tabQL_greedy,
    'tabQL_Cest_em':tabQL_Cest_em
}
