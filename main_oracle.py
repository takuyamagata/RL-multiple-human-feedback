import numpy as np

import envPacMan as environment 
from agent import agent

from RLmon import RLmon

class feedback():
    def __init__(self, state=[], good_actions=[], conf_good_actions=[], bad_actions=[], conf_bad_actions=[]):
        good_actions = np.array(good_actions)
        bad_actions = np.array(bad_actions)
        
        # check inputs
        if len(good_actions.shape) == 1:
            good_actions = good_actions.reshape((1,-1))
        if len(bad_actions.shape) == 1:
            bad_actions = bad_actions.reshape((1,-1))
        if not hasattr(conf_good_actions, '__len__'):
            conf_good_actions = [conf_good_actions]
        if not hasattr(conf_bad_actions, '__len__'):
            conf_bad_actions = [conf_bad_actions]
        
        self.state = state
        self.good_actions = good_actions
        self.conf_good_actions = conf_good_actions
        self.bad_actions = bad_actions
        self.conf_bad_actions = conf_bad_actions
        
        
def generate_feedback(state, nActions, C, right_actions, type='binary-feedback', action=None):
    # generate human feedback
    #   nActions = number of actions in the environment
    #   C = consistency level (probability of giving a right feedback)
    #   type = feedback type -- 'binary-feedback', 'soft-feedback', 'crisp-set', 'soft-set'
    #   right_actions = list of actions for the optimal actions
    #   action = action for giving a feedback
    
    if type=='binary-feedback':
        # right or wrong (original Adivce algorithm)
        if np.random.rand() < C:
            # right feedback
            if action in right_actions:
                ret = feedback(state=state, good_actions=[action], conf_good_actions=1.0)
            else:
                ret = feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
        else:
            # wrong feedback
            if action in right_actions:
                ret = feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
            else:
                ret = feedback(state=state, good_actions=[action], conf_good_actions=1.0)
    elif type=='soft-feedback':
        # right or wrong with the confidence level [0,1]
        sampled_C = np.random.beta(a=C*10, b=(10-C*10)) # sample C from beta distribution to keep the expectation to be the given C
        confidence = np.abs(sampled_C - 0.5) * 2.0
        if np.random.rand() < sampled_C:
            # right feedback
            if action in right_actions:
                ret = feedback(state=state, good_actions=[action], conf_good_actions=confidence)
            else:
                ret = feedback(state=state, bad_actions=[action], conf_bad_actions=confidence)
        else:
            # wrong feedback
            if action in right_actions:
                ret = feedback(state=state, bad_actions=[action], conf_bad_actions=confidence)
            else:
                ret = feedback(state=state, good_actions=[action], conf_good_actions=confidence)
    elif type=='crisp-set':
        # set of good and/or bad actions      
        num_feedback_actions = 2
        a_list = np.random.choice(nActions, num_feedback_actions, replace=False)
        good_actions, bad_actions = [], []
        for a in a_list:
            if np.random.rand() < C:
                # right feedback
                if a in right_actions:
                    good_actions.append(a)
                else:
                    bad_actions.append(a)
            else:
                # wrong feedback
                if a in right_actions:
                    bad_actions.append(a)
                else:
                    good_actions.append(a)
        conf_good_actions = 1.0 if len(good_actions) > 0 else []
        conf_bad_actions = 1.0 if len(bad_actions) > 0 else []
        ret = feedback(state=state, good_actions=good_actions, conf_good_actions=conf_good_actions,
                                    bad_actions=bad_actions,   conf_bad_actions=conf_bad_actions)
    elif type=='soft-set':
        # set of good and/or bad actions      
        num_feedback_actions = 2
        a_list = np.random.choice(nActions, num_feedback_actions, replace=False)
        good_actions, bad_actions = [], []
        sampled_C = np.random.beta(a=C*10, b=(10-C*10)) # sample C from beta distribution to keep the expectation to be the given C
        confidence = np.abs(sampled_C - 0.5) * 2.0
        for a in a_list:
            if np.random.rand() < sampled_C:
                # right feedback
                if a in right_actions:
                    good_actions.append(a)
                else:
                    bad_actions.append(a)
            else:
                # wrong feedback
                if a in right_actions:
                    bad_actions.append(a)
                else:
                    good_actions.append(a)
        conf_good_actions = confidence if len(good_actions) > 0 else []
        conf_bad_actions = confidence if len(bad_actions) > 0 else []
        ret = feedback(state=state, good_actions=good_actions, conf_good_actions=conf_good_actions,
                                    bad_actions=bad_actions,   conf_bad_actions=conf_bad_actions)         
                
    return ret

# ==================================================================================================

def main(algID   = 'tabQL_Cest_em_t2',  # Agent Algorithm   'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2', 
                                        #                   'tabQL_Cest_em_t1', 'tabQL_Cest_em_org_t2', 
                                        #                   'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
         feedback_type = 'binary-feedback', # feedback type 'binary-feedback', 'soft-feedback', 'crisp-set', 'soft-set'
         simInfo = '_tmp',              # Filename header
         env_size = 'small',            # Pacman environment size 'small' or 'medium'
         trial_count = 100,             # number of learning trial
         episode_count = 2000,          # number of episodes to learn
         max_steps = 500,               # max. number of steps in a episode
         L  = np.array([1.0]),          # probability to give a feedback
         C  = np.array([0.2]),          # Human feedback confidence level
         a  = 1.0,                      # alpha for C prior
         b  = 1.0,                      # beta  for C prior
         no_reward = False,             # agent learns the policy without reward (feedback only)
         C_fixed = None,                # None: learn C, np.array(): fixed C (fixed C only works with "tabQL_Cest_em_org_t1" or "tabQL_Cest_em_org_t2")
         update_Cest_interval = 4,      # Cest update interval (number of espisodes)
         ):

    print(f"start--{algID} {simInfo}")
    dispON = False
    
    # prepare RL monitor module
    legendStr = []
    for n in range(len(C)):
        legendStr.append('L={0},C={1}'.format(L[n], C[n]))
    mon = RLmon(trial_count, episode_count, 1)
    monC = RLmon(trial_count, episode_count, len(C))
    monAlpha = RLmon(trial_count, episode_count, len(C))
    monBeta  = RLmon(trial_count, episode_count, len(C))
    
    env_h = environment.env(env_size)            
    for k in range(trial_count):
        print('trial: {0}'.format(k))
        
        env_h.reset()
        agent_h  = agent(algID, env_h.nStates(), len(env_h.action_list()), 
                         a=a, b=b, 
                         C_fixed=C_fixed, 
                        )
        
        # Setup ORACLE
        oracle_h = agent('tabQLgreedy', env_h.nStates(), len(env_h.action_list()))
        if env_size == 'small': # load pre-learned Q function 
            oracle_h.load('learnedStates/pacman_tabQL_oracle.pkl')
        elif env_size == 'medium':
            oracle_h.load('learnedStates/pacman_medium_tabQL_oracle.pkl')
        else:
            raise ValueError(f"nvalid env_size value - must be 'small' or 'medium': {env_size}")
        oracle_h.alpha = 0                          # set learning rate to zero (no learning)
        
        action_list = env_h.action_list()
        action = 0 
        ob = env_h.st2ob()            # observation
        rw = 0                        # reward
        totRW = 0                     # total reward in this episode
        done = False                  # episode completion flag
        fb = [feedback() for n in range(len(C))] # Human feedback
        update_Cest = False
        
        for i in range(episode_count):
            
            for j in range(max_steps):
                
                if dispON:
                    print('action:{0}'.format(action_list[action]))
                    d = env_h.display()
                    for n in range(len(d)):
                        print(d[n])
                        
                    tmp = input('>>')
                    if tmp == 'stop':
                        dispON = False
        
                # call agent
                action = agent_h.act(action, ob, rw, done, fb, 0.5, update_Cest=False)
                # call oracle to get 'right' action
                if np.any(L > 0.0):
                    rightAction = oracle_h.act(action, ob, rw, done, fb, C)
                    ob_for_feedback = ob
                    
                # call environment
                ob, rw, done = env_h.step(action_list[action])

                # accumrate total reward
                totRW += rw

                # set reward zero when simulating without reward scase
                if no_reward:
                    rw = 0.0
                
                # 'human' feedback generation (by using ORACLE)
                for trainerIdx in np.arange(len(fb)):
                    if np.random.rand() < L[trainerIdx]:
                        fb[trainerIdx] = generate_feedback(ob_for_feedback, len(env_h.action_list()), C[trainerIdx], [rightAction], type=feedback_type, action=action) # Right feedback
                    else:
                        fb[trainerIdx] = feedback() # no feedback
                
                
                # if done==True, call agent once more to learn 
                # the final transition, then finish this episode.
                if done:
                    update_Cest = ((i+1) % update_Cest_interval == 0)
                    agent_h.act(action, ob, rw, done, fb, C, update_Cest=update_Cest)
                    break
            
            if i % 100 == 0:
                print(f"{k}, {i}: Ce: {agent_h.Ce} \t total reward: {totRW}")
            
            # store result
            mon.store(i, k, totRW)
            monC.store(i, k, agent_h.Ce)
            if hasattr(agent_h, 'sum_of_right_feedback'):
                # store VI algorithm parameters
                monAlpha.store(i, k, agent_h.sum_of_right_feedback + agent_h.a)
                monBeta.store(i, k,  agent_h.sum_of_wrong_feedback + agent_h.b)
                
            # Reset environment
            env_h.reset()
            agent_h.prev_obs = None
            ob = env_h.st2ob()
            rw = 0
            totRW = 0
            done = False
        
        # save model
        # agent_h.save('learnedStates/pacman_medium_tabQL_oracle')
                        
        # Clear agent class except the last trial
        if k < trial_count-1:
            del agent_h
            del oracle_h

    # Save results
    fname = 'results/aveRW_' + 'Env_' + str(env_size) + '_' + str(algID) + str(simInfo)
    mon.saveData(fname)
    fname = 'results/aveC_' + 'Env_' + str(env_size) + '_' + str(algID) + str(simInfo)
    monC.saveData(fname)
    if hasattr(agent_h, 'sum_of_right_feedback'):
        fname = 'results/aveAlpha_' + 'Env_' + str(env_size) + '_' + str(algID) + str(simInfo)
        monAlpha.saveData(fname)
        fname = 'results/aveBeta_' + 'Env_' + str(env_size) + '_' + str(algID) + str(simInfo)
        monBeta.saveData(fname)
        
    #fname = 'results/plot_' + str(algID) + str(simInfo)
    #mon.savePlot(fname)
    
    agent_h.save('learnedStates/pacman_' + str(algID))

if __name__ == '__main__':
    main()
