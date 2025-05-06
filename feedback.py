import numpy as np

# class for storing the trajectory for generating feedback
class Trajectory():
    def __init__(self, state=[], action=[], optimal_action=[], reward=[], done=[]):
        assert len(state) == len(action) == len(optimal_action) == len(reward) == len(done), \
            f"Length of state, action, optimal_action, reward, done must be the same: {len(state)}, {len(action)}, {len(optimal_action)}, {len(reward)}, {len(done)}"
        self.state = state
        self.action = action
        self.optimal_action = optimal_action
        self.reward = reward
        self.done = done
        return

    def reset(self):
        self.state = []
        self.action = []
        self.optimal_action = []
        self.reward = []
        self.done = []
        return

    def append(self, state=None, action=None, optimal_action=None, reward=None, done=None):
        self.state.append(state)
        self.action.append(action)
        self.optimal_action.append(optimal_action)
        self.reward.append(reward)
        self.done.append(done)
        
    def __str__(self):
        return f"state: {self.state}, action: {self.action}, reward: {self.reward}, done: {self.done}"
    
    def __len__(self):
        return len(self.state)

# class for storing the feedback
class Feedback():
    def __init__(self, state=[], good_actions=[], conf_good_actions=[], bad_actions=[], conf_bad_actions=[]):
        # good_actions = np.array(good_actions)
        # bad_actions = np.array(bad_actions)
        
        # check inputs
        # if len(good_actions.shape) == 1:
        #     good_actions = good_actions.reshape((1,-1))
        # if len(bad_actions.shape) == 1:
        #     bad_actions = bad_actions.reshape((1,-1))
        # if not hasattr(conf_good_actions, '__len__'):
        #     conf_good_actions = [conf_good_actions]
        # if not hasattr(conf_bad_actions, '__len__'):
        #     conf_bad_actions = [conf_bad_actions]
        
        self.state = state
        self.good_actions = good_actions
        self.conf_good_actions = conf_good_actions
        self.bad_actions = bad_actions
        self.conf_bad_actions = conf_bad_actions
        
# generate a single feedback        
def generate_single_feedback(state, action_list, C, optimal_actions, type='binary-feedback', action=None):
    # generate human feedback
    #   action_list = list of possible actions in the environment
    #   C = consistency level (probability of giving a right feedback)
    #   type = feedback type -- 'binary-feedback', 'soft-feedback', 'crisp-set', 'soft-set'
    #   optimal_actions = list of actions for the optimal actions
    #   action = action for giving a feedback
    
    if type=='binary-feedback':
        # right or wrong (original Adivce algorithm)
        if np.random.rand() < C:
            # right feedback
            if action in optimal_actions:
                ret = Feedback(state=state, good_actions=[action], conf_good_actions=1.0)
            else:
                ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
        else:
            # wrong feedback
            if action in optimal_actions:
                ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
            else:
                ret = Feedback(state=state, good_actions=[action], conf_good_actions=1.0)
    elif type=='soft-feedback':
        # right or wrong with the confidence level [0,1]
        sampled_C = np.random.beta(a=C*10, b=(10-C*10)) # sample C from beta distribution to keep the expectation to be the given C
        confidence = np.abs(sampled_C - 0.5) * 2.0
        if np.random.rand() < sampled_C:
            # right feedback
            if action in optimal_actions:
                ret = Feedback(state=state, good_actions=[action], conf_good_actions=confidence)
            else:
                ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=confidence)
        else:
            # wrong feedback
            if action in optimal_actions:
                ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=confidence)
            else:
                ret = Feedback(state=state, good_actions=[action], conf_good_actions=confidence)
    elif type=='crisp-set':
        # set of good and/or bad actions      
        num_feedback_actions = 2
        a_list = np.random.choice(action_list, num_feedback_actions, replace=False)
        good_actions, bad_actions = [], []
        for a in a_list:
            if np.random.rand() < C:
                # right feedback
                if a in optimal_actions:
                    good_actions.append(a)
                else:
                    bad_actions.append(a)
            else:
                # wrong feedback
                if a in optimal_actions:
                    bad_actions.append(a)
                else:
                    good_actions.append(a)
        conf_good_actions = 1.0 if len(good_actions) > 0 else []
        conf_bad_actions = 1.0 if len(bad_actions) > 0 else []
        ret = Feedback(state=state, good_actions=good_actions, conf_good_actions=conf_good_actions,
                                    bad_actions=bad_actions,   conf_bad_actions=conf_bad_actions)
    elif type=='soft-set':
        # set of good and/or bad actions      
        num_feedback_actions = 2
        a_list = np.random.choice(action_list, num_feedback_actions, replace=False)
        good_actions, bad_actions = [], []
        sampled_C = np.random.beta(a=C*10, b=(10-C*10)) # sample C from beta distribution to keep the expectation to be the given C
        confidence = np.abs(sampled_C - 0.5) * 2.0
        for a in a_list:
            if np.random.rand() < sampled_C:
                # right feedback
                if a in optimal_actions:
                    good_actions.append(a)
                else:
                    bad_actions.append(a)
            else:
                # wrong feedback
                if a in optimal_actions:
                    bad_actions.append(a)
                else:
                    good_actions.append(a)
        conf_good_actions = confidence if len(good_actions) > 0 else []
        conf_bad_actions = confidence if len(bad_actions) > 0 else []
        ret = Feedback(state=state, good_actions=good_actions, conf_good_actions=conf_good_actions,
                                    bad_actions=bad_actions,   conf_bad_actions=conf_bad_actions)
    
    # no information binary-feedbacks
    elif type == 'binary-random':
        # randomly pick right or wrong (original Adivce algorithm)
        if np.random.rand() < 1.0/len(action_list):
            ret = Feedback(state=state, good_actions=[action], conf_good_actions=1.0)
        else:
            ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
    elif type == 'binary-positive':
        # always positive (say right) (original Adivce algorithm)
        ret = Feedback(state=state, good_actions=[action], conf_good_actions=1.0)
        
    elif type == 'binary-negative':         
        # always negative (say wrong) (original Adivce algorithm)
        ret = Feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)

    return ret


# Generate various type of feedback with a specified Active feedback method
def generate_feedback(trajectory,
                      C,
                      L,
                      action_list, 
                      agent_h, 
                      oracle_h, 
                      end_of_episode=False, 
                      no_reward=False, 
                      feedback_type = 'binary-feedback', 
                      active_feedback_type=None,):
    
    if active_feedback_type is None:
        for trainerIdx in np.arange(len(fb)):
            if np.random.rand() < L[trainerIdx]:
                fb[trainerIdx] = [generate_single_feedback(
                                        trajectory.state[-1], 
                                        len(action_list), 
                                        C[trainerIdx], 
                                        [trajectory.optimal_action[-1]], 
                                        type=feedback_type, 
                                        action=trajectory.action[-1])] # Right feedback
            else:
                fb[trainerIdx] = [Feedback()] # no feedback
    elif 'random' in active_feedback_type:
        # active feedback (random) -- create feedback at the end of the episode
        fb = [[] for n in range(len(C))]
        if end_of_episode:
            for trainerIdx in np.arange(len(fb)):
                N_fb = len(trajectory) * L[trainerIdx]
                N_fb = int(N_fb) + 1 if np.random.rand() < (N_fb - int(N_fb)) else int(N_fb) # number of feedbacks
                idx = np.random.choice(len(trajectory), N_fb, replace=False) # pick N_fb items randomly
                for n in idx:
                    fb[trainerIdx].append(generate_single_feedback(
                                                trajectory.state[n],
                                                len(action_list), 
                                                C[trainerIdx], 
                                                [trajectory.optimal_action[n]], 
                                                type=feedback_type, 
                                                action=trajectory.action[n]))
    elif 'count' in active_feedback_type:
        # active feedback (count-based) -- create feedback at the end of the episode
        fb = [[] for n in range(len(C))] # reset feedbacks
        if end_of_episode:
            # get the number of visitations and feedbacks
            N = np.zeros((len(trajectory),))
            if 'last3' in active_feedback_type: # prioritise the last 3 items
                N[-np.minimum(3,len(N)):] -= 5
            for n, (s, a) in enumerate(zip(trajectory.state, trajectory.action)):
                if not no_reward:
                    N[n] += agent_h.Nsa[s, a]
                N[n] += agent_h.hp[:, s, a].sum() + agent_h.hm[:, s, a].sum()
            
            for trainerIdx in np.arange(len(fb)):
                N_fb = len(trajectory) * L[trainerIdx]
                N_fb = int(N_fb) + 1 if np.random.rand() < (N_fb - int(N_fb)) else int(N_fb) # number of feedbacks
                idx = np.argpartition(N+np.random.normal(0, 1.0, len(N)), N_fb)[:N_fb] # pick N_fb items with the smallest N
                # idx = np.argsort(N+np.random.normal(0, 1.0, len(N)))[:N_fb] # pick N_fb items with the smallest N
                for n in idx:
                    fb[trainerIdx].append(generate_single_feedback(
                                                trajectory.state[n],
                                                len(action_list), 
                                                C[trainerIdx], 
                                                [trajectory.optimal_action[n]], 
                                                type=feedback_type, 
                                                action=trajectory.action[n]))
    elif 'value' in active_feedback_type:
        # active feedback (count-based) -- create feedback at the end of the episode
        fb = [[] for n in range(len(C))] # reset feedbacks
        if end_of_episode:
            # get the number of visitations and feedbacks
            value_gap = np.zeros((len(trajectory),))
            for n, (s, a) in enumerate(zip(trajectory.state, trajectory.action)):
                N = agent_h.hp[:, s, a].sum() + agent_h.hm[:, s, a].sum() if no_reward else \
                    agent_h.hp[:, s, a].sum() + agent_h.hm[:, s, a].sum() + agent_h.Nsa[s, a]
                if 'last3' in active_feedback_type and n > (len(trajectory) - 4): # prioritise the last 3 items
                    N -= 5
                a_argmax = np.argmax(agent_h.Q[s, :])
                Q_max = 520 # environment dependent 
                value_gap[n] = (agent_h.Q[s, a_argmax] + (Q_max - agent_h.Q[s, a_argmax])/np.sqrt(np.maximum(N,1)) - agent_h.Q[s, a])

            for trainerIdx in np.arange(len(fb)):
                N_fb = len(trajectory) * L[trainerIdx]
                N_fb = int(N_fb) + 1 if np.random.rand() < (N_fb - int(N_fb)) else int(N_fb) # number of feedbacks
                idx = np.argpartition(-value_gap, N_fb)[:N_fb] # pick N_fb items with the largest value gap
                for n in idx:
                    fb[trainerIdx].append(generate_single_feedback(
                                                trajectory.state[n],
                                                len(action_list), 
                                                C[trainerIdx], 
                                                [trajectory.optimal_action[n]], 
                                                type=feedback_type, 
                                                action=trajectory.action[n]))
    
        
    elif active_feedback_type == 'ideal':
        # active feedback (ideal) -- create feedback at the end of the episode
        fb = [[] for n in range(len(C))]
        if end_of_episode:
            regret_on_trajectory = []
            for n, (s, a) in enumerate(zip(trajectory.state, trajectory.action)):
                regret_on_trajectory.append(oracle_h.Q[s, :].max() - oracle_h.Q[s, np.argmax(agent_h.Q[s, :])])
            for trainerIdx in np.arange(len(fb)):
                N_fb = len(trajectory) * L[trainerIdx]
                N_fb = int(N_fb) + 1 if np.random.rand() < (N_fb - int(N_fb)) else int(N_fb) # number of feedbacks
                idx = np.argpartition(-np.array(regret_on_trajectory) + 
                                        np.random.normal(0, 1.0, len(regret_on_trajectory)),
                                        N_fb)[:N_fb] # pick the item with the smallest regret
                for n in idx:
                    fb[trainerIdx].append(generate_single_feedback(
                                                trajectory.state[n],
                                                len(action_list), 
                                                C[trainerIdx], 
                                                [trajectory.optimal_action[n]], 
                                                type=feedback_type, 
                                                action=trajectory.action[n]))
    return fb

