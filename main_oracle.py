import numpy as np

import envPacMan as environment 
from agent import agent
from RLmon import RLmon
from feedback import *

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
         prior_alpha  = 1.0,            # alpha for C prior
         prior_beta   = 1.0,            # beta  for C prior
         no_reward = False,             # agent learns the policy without reward (feedback only)
         C_fixed = None,                # None: learn C, np.array(): fixed C (fixed C only works with "tabQL_Cest_em_org_t1" or "tabQL_Cest_em_org_t2")
         update_Cest_interval = 5,      # Cest update interval (number of espisodes)
         active_feedback_type = None,   # active feedback type (None: no active feedback, 'count') 
         ):

    print(f"start--{algID} {simInfo}")
    dispON = False
    
    # prepare RL monitor module
    legendStr = []
    for n in range(len(C)):
        legendStr.append('L={0},C={1}'.format(L[n], C[n]))
    
    monitor = RLmon(['return', 'Cest', 'alpha', 'beta'])
    trajectory = Trajectory()
    
    env_h = environment.env(env_size)            
    for k in range(trial_count):
        print('trial: {0}'.format(k))
        
        env_h.reset()
        agent_h  = agent(algID, env_h.nStates(), len(env_h.action_list()), 
                         a=prior_alpha, b=prior_beta, 
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
        fb = [[] for n in range(len(C))] # Human feedback
        update_Cest = False

        totalRW_list = []
        
        for i in range(episode_count):

            trajectory.reset() # store trajectory for generating active feedback (generate feedback at the end of the episode)
            
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

                # store the trajectory for generating active feedback (generate feedback at the end of the episode)
                if np.any(L > 0.0):
                    trajectory.append(state=ob_for_feedback, action=action, optimal_action=rightAction, reward=rw, done=done)

                # set reward zero when simulating without reward scase
                if no_reward:
                    rw = 0.0
                
                # 'human' feedback generation (by using ORACLE)
                if np.any(L > 0.0):
                    # generate feedbacks
                    fb = generate_feedback(
                            trajectory=trajectory,
                            C=C,
                            L=L,
                            action_list=env_h.action_list(), 
                            agent_h=agent_h, 
                            oracle_h=oracle_h, 
                            end_of_episode=(done or j == max_steps - 1), 
                            no_reward=no_reward, 
                            feedback_type = feedback_type, 
                            active_feedback_type=active_feedback_type,)
                               
                if done or j == max_steps - 1:             
                    update_Cest = ((i+1) % update_Cest_interval == 0)
                    agent_h.act(action, ob, rw, done, fb, C, update_Cest=update_Cest)
                    break
            
            totalRW_list.append(totRW)
            if i % 20 == 0:
                print(f"{k}, {i}: Ce: {agent_h.Ce} \t total reward: {np.mean(totalRW_list)}")
                totalRW_list = []
            
            # store result
            monitor.store('return', totRW)
            monitor.store('Cest', agent_h.Ce)
            if hasattr(agent_h, 'sum_of_right_feedback'):
                # store VI algorithm parameters
                monitor.store('alpha', agent_h.sum_of_right_feedback + agent_h.a)
                monitor.store('beta',  agent_h.sum_of_wrong_feedback + agent_h.b)
                
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
        
        # store the accumurated results
        monitor.store(done=True)

    # Save results
    fname = 'results/results_' + 'Env_' + str(env_size) + '_' + str(algID) + str(simInfo)
    monitor.saveData(fname)
        
    #fname = 'results/plot_' + str(algID) + str(simInfo)
    #mon.savePlot(fname)
    
    agent_h.save('learnedStates/pacman_' + str(algID))

if __name__ == '__main__':
    main()
