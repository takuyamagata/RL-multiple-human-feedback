import numpy as np

import envPacMan as environment 
from agent import agent

from RLmon import RLmon
        
# -----------------------------------------------------
def main(algID   = 'tabQL_Cest_em_t2',  # Agent Algorithm   'tabQL_Cest_em_org_t1', 'tabQL_Cest_em_org_t2', 
                                        #                   'tabQL_Cest_em_t1', 'tabQL_Cest_em_org_t2', 
                                        #                   'tabQL_Cest_vi_t1', 'tabQL_Cest_vi_t2'
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
        agent_h  = agent(algID, env_h.nStates(), len(env_h.action_list()), a=a, b=b, C_fixed=C_fixed)
        
        # Setup ORACLE
        oracle_h = agent('tabQLgreedy', env_h.nStates(), len(env_h.action_list()))
        oracle_h.load('learnedStates/pacman_tabQL_oracle.pkl')   # load pre-learned Q function    
        oracle_h.alpha = 0                          # set learning rate to zero (no learning)
        
        action_list = env_h.action_list()
        action = 0 
        ob = env_h.st2ob()            # observation
        rw = 0                        # reward
        totRW = 0                     # total reward in this episode
        done = False                  # episode completion flag
        fb = np.ones(len(C)) * np.NaN # Human feedback
        
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
                action = agent_h.act(action, ob, rw, done, fb, 0.5)
                # call oracle to get 'right' action
                if np.any(L > 0.0):
                    rightAction = oracle_h.act(action, ob, rw, done, fb, C)
                    
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
                        if np.random.rand() < C[trainerIdx]:
                            fb[trainerIdx] = (action == rightAction)     # Right feedback
                        else:
                            fb[trainerIdx] = not (action == rightAction) # Wrong feedback
                    else:
                        fb[trainerIdx] = np.NaN # no feedback
                
                
                # if done==True, call agent once more to learn 
                # the final transition, then finish this episode.
                if done:
                    agent_h.act(action, ob, rw, done, fb, C)
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
