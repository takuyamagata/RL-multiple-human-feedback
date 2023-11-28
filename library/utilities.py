import os
import numpy as onp
import jax
import jax.numpy as np
from itertools import combinations
from functools import partial
from config import parameters, environment, rl

"""
Episodic RL monitor
"""
class RLmon(object):
    
    def __init__(self, setup, numData=1):
        self.episode_count = setup['episode_count']
        self.numData = numData
        # prepare a buffer for averaging reward
        self.ave   = onp.zeros([numData, setup['episode_count']])
        self.aveSq = onp.zeros([numData, setup['episode_count']])
        self.raw   = onp.zeros([numData, setup['episode_count'], setup['trial_count']])
        return
    
    def store(self, episode_idx, trial_idx, d):
        # update averaged reward        
        self.ave[:,episode_idx]   = (self.ave[:,episode_idx]   * trial_idx + d) / (trial_idx + 1)
        self.aveSq[:,episode_idx] = (self.aveSq[:,episode_idx] * trial_idx + d**2) / (trial_idx + 1)
        self.raw[:,episode_idx, trial_idx] = d
        return
    
    def saveData(self, fname):
        stddev = np.sqrt( self.aveSq - self.ave**2 )
        np.savez(fname, ave=self.ave, std=stddev, raw=self.raw)        
        return


# Save results
def save_results(monitors,agent,algID,path):

    res_dir = ['monitors','learnedStates']

    for rd in res_dir:
        res_path = 'results/' + path + '/' + rd

        if not os.path.exists(res_path):
            os.makedirs(res_path)

    for name,monitor in monitors.items():
        monitor.saveData('results/'+ path + '/monitors/' + algID + '_' + name)

    agent.save('results/' + path + '/learnedStates/' + algID )


# ----------------------------------------------------------------------------
# Process data
# ----------------------------------------------------------------------------

def dict_search(my_dict, target_value):
    for key, value in my_dict.items():
        if np.array_equal(value,target_value):
            return key
    return None  # Return None if the value is not found


def moving_average(d, len):
    prePadLen = len//2
    posPadLen = len - prePadLen
    d_ = np.append(d[0:prePadLen], d)
    d_ = np.append(d_, d[-posPadLen:])
    cs = np.cumsum(d_)
    ma = (cs[len:] - cs[:-len]) / len
    return ma

@jax.jit
def mask_array(input_array):
    """
    Generate arrays containing every other element from the input array.

    Args:
    - input_array (array): The input array from which combinations are generated.

    Returns:
    - result_arrays (list of lists): A list of arrays, each of which omits one element from the input array.
    """
    return np.array([list(comb) for comb in combinations(input_array, len(input_array) - 1)])

# @jax.jit
def index_to_element(matrix_shape, index):
    """
    Converts an index for a matrix of any shape and dimensionality
    into the corresponding element in the flattened list.

    Args:
    - matrix_shape (tuple): The shape of the matrix as a tuple (rows, columns, depth, ...).
    - index (tuple): The index for the matrix as a tuple (row, column, depth, ...).

    Returns:
    - element: The corresponding element in the flattened list.
    """
    if len(matrix_shape) != len(index):
        raise ValueError("Matrix shape and index must have the same dimensionality.")

    if not all(0 <= i < dim for i, dim in zip(index, matrix_shape)):
        raise ValueError("Index is out of bounds for the given matrix shape.")

    flattened_index = 0
    multiplier = 1

    for i in range(len(matrix_shape) - 1, -1, -1):
        flattened_index += index[i] * multiplier
        multiplier *= matrix_shape[i]

    return int(flattened_index)


# Define the dictionary mapping input arrays to numbers
array_to_number = {
    (0, 1): 0,    # north
    (0, -1): 1,   # south
    (-1, 0): 2,   # west
    (1, 0): 3     # east
    }


def map_array_to_number(input_array):
    # Use the dictionary to map the input array to a number
    return array_to_number.get(tuple(input_array.tolist()))
# ----------------------------------------------------------------------------
# Log calculation functions
# ----------------------------------------------------------------------------

# calculate log(a+b) from log(a) and log(b)
# def logadd(a, b):
#     if a > b:
#         out = a + np.log( 1 + np.exp(b-a) )
#     elif a < b:
#         out = b + np.log( 1 + np.exp(a-b) )
#     else:
#         if np.abs(a) == np.inf:
#             out = a
#         else:
#             out = a + np.log( 1 + np.exp(b-a) )
            
#     return out

def logadd(a, b):
    max_ab = np.maximum(a, b)
    min_ab = np.minimum(a, b)
    out = max_ab + np.log1p(np.exp(min_ab - max_ab))
    return out    


# calculate log( sum(a) ) from log(a)
def logsum(a):
    m = np.max(a)
    out = m + np.log( np.sum( np.exp(a-m) ) )
    return out
    
# normalise log-probability p
def lognorm(p):
    m = np.max(p)
    out = p - (m + np.log( np.sum( np.exp(p-m) ) ) )
    return out
        
# Replace character 
def replaceChar(self, st, c, idx):
    return st[0:idx] + c + st[idx+1:]


def newPos(currPos, dir):
    newPos = currPos + dir
    
    if (all(np.greater_equal(np.array([environment['size']['X'],environment['size']['Y']]),newPos)) and 
        all(np.greater_equal(newPos,np.array([0,0])))) and not np.any(np.all(environment['obstacles'] == newPos, axis=1)):
        return  newPos
    else:
        return currPos


def randMove(key):
    return np.array(list(environment['actions'])[jax.random.choice(key, np.arange(rl['nActions']))])
    
    
jit_rand = jax.jit(randMove)

# @jax.jit
def find_array_index(array_to_find, higher_dimension_array):
    """
    Find the index of `array_to_find` within `higher_dimension_array`.
    
    Args:
    array_to_find: The array to search for.
    higher_dimension_array: The higher-dimensional array to search in.
    
    Returns:
    The index of `array_to_find` within `higher_dimension_array`.
    Returns None if not found.
    """
    # Convert the arrays to NumPy arrays for efficient comparison
    assert isinstance(array_to_find, np.ndarray), "array_to_find must be a NumPy array"
    assert isinstance(higher_dimension_array, np.ndarray), "higher_dimension_array must be a NumPy array"

    # Use NumPy's argwhere to find the index
    index = np.argwhere(np.all(higher_dimension_array == array_to_find, axis=1))

    # Check if the array was found
    if len(index) > 0:
        return index[0]
    else:
        return None    

@jax.jit
def gpPosIdx(pacman,ghost):
    return pacman[0] + pacman[1]*(environment['size']['X']+1), ghost[0] + ghost[1]*(environment['size']['X']+1)

@jax.jit
def st2ob(pacman_pos,ghost_pos,ghost_dir,pellets_valid):
        pPosIdx = pacman_pos[0] + pacman_pos[1]*(environment['size']['X']+1)
        gPosIdx = ghost_pos[0] + ghost_pos[1]*(environment['size']['X']+1)
        gDirIdx = map_array_to_number(ghost_dir) 
        peltIdx = int(np.sum((np.arange(0,len(environment['pellets']))+1)*pellets_valid))
        
        return index_to_element(rl['stateShape'],(pPosIdx,gPosIdx,gDirIdx,peltIdx))



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
        
        
def generate_feedback(key,state, nActions, C, right_actions, type='binary-feedback', action=None):
    # generate human feedback
    #   nActions = number of actions in the environment
    #   C = consistency level (probability of giving a right feedback)
    #   type = feedback type -- 'binary-feedback', 'soft-feedback', 'crisp-set', 'soft-set'
    #   right_actions = list of actions for the optimal actions
    #   action = action for giving a feedback
    key,subkey = jax.random.split(key)
    if type=='binary-feedback':
        # right or wrong (original Adivce algorithm)
        if jax.random.uniform(subkey) < C:
            # right feedback
            if find_array_index(action,right_actions):
                ret = feedback(state=state, good_actions=[action], conf_good_actions=1.0)
            else:
                ret = feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
        else:
            # wrong feedback
            if np.any(np.all(right_actions, action)):
                #TODO: UPDATE THIS IF STATEMENT FOR ALL OTHER METHODS
                ret = feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
            else:
                ret = feedback(state=state, good_actions=[action], conf_good_actions=1.0)
    elif type=='soft-feedback':
        # right or wrong with the confidence level [0,1]
        key,subkey = jax.random.split(key)
        sampled_C = jax.random.beta(subkey,a=C*10, b=(10-C*10)) # sample C from beta distribution to keep the expectation to be the given C
        confidence = np.abs(sampled_C - 0.5) * 2.0
        key,subkey = jax.random.split(key)
        if jax.random.uniform(subkey) < sampled_C:
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
        a_list = jax.random.choice(key,environment['actions'],shape=num_feedback_actions,replace=False)
        # a_list = np.zeros((num_feedback_actions))

        # for i in range(num_feedback_actions):

        #     key,subkey = jax.random.split(key)
        #     a_list = a_list.at[i].set(jit_rand(subkey))

        # a_list = jit_rand(nActions, num_feedback_actions, replace=False)
        good_actions, bad_actions = [], []
        for a in a_list:
            key,subkey = jax.random.split(key)
            if jax.random.uniform(key) < C:
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
        a_list = jax.random.choice(key,environment['actions'],shape=num_feedback_actions,replace=False)
        good_actions, bad_actions = [], []
        sampled_C = jax.random.beta(a=C*10, b=(10-C*10)) # sample C from beta distribution to keep the expectation to be the given C
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
    
    # no information binary-feedbacks
    elif type == 'binary-random':
        
        # randomly pick right or wrong (original Adivce algorithm)
        if jax.random.uniform(key) < 1.0/nActions:
            ret = feedback(state=state, good_actions=[action], conf_good_actions=1.0)
        else:
            ret = feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)
    elif type == 'binary-positive':
        # always positive (say right) (original Adivce algorithm)
        ret = feedback(state=state, good_actions=[action], conf_good_actions=1.0)
        
    elif type == 'binary-negative':         
        # always negative (say wrong) (original Adivce algorithm)
        ret = feedback(state=state, bad_actions=[action], conf_bad_actions=1.0)

    return ret

def collect_feedback(feedback_list,hp, hm):
    # collect feedback and update self.hp and self.hm. 
    for n, fb in enumerate(feedback_list):
        # if fb.good_actions.shape[1] > 0:
        for m in range(fb.good_actions.shape[0]): # support multiple set of good/bad actions with different confidence level
            for a in fb.good_actions[m]:
                hp = hp.at[n, fb.state, a].set(hp[n, fb.state, a] + fb.conf_good_actions[m])
        # if fb.bad_actions.shape[1] > 0:
        for m in range(fb.bad_actions.shape[0]):
            for a in fb.bad_actions[m]:
                hm = hm.at[n, fb.state, a].set(hm[n, fb.state, a] + fb.conf_bad_actions[m])
    return hp, hm

@jax.jit
def calcCe(ln_P_Q1,d,hp,hm,Ce):
    # E-step (compute posterior of O)
        #  type 2 (one optimal action)
        # for each state action pair where some feedback exists
        # update log probs 1 at s,a index with:
        #    normalised action probabilities 
        # +  sum of all positive trainer feedback * consistency 
        # -  sum of all positive trainer feedback * (1-consistency) 
    Ce_m=Ce[:,None,None]
    ln_P1 = (ln_P_Q1 + np.sum(d*np.log(Ce_m)) - np.sum(d*np.log(1-Ce_m)))
    
    ln_P0 = (np.sum(ln_P_Q1, axis=1, keepdims=True) - ln_P_Q1 + np.sum(d*np.log(Ce_m)) - np.sum(d*np.log(1-Ce_m)))
    
    ln_partition = np.vectorize(logadd)(ln_P0, ln_P1)
    ln_P0 = ln_P0 - ln_partition
    ln_P1 = ln_P1 - ln_partition
    # # todo: make sure to correct ln_partition in other algorithms
    
    P1, P0 = np.exp(ln_P1), np.exp(ln_P0)
    
    Ce = np.sum(P1 * hp + P0 * hm, axis=(1,2),keepdims=True)/ np.sum(hp  + hm,axis=(1,2),keepdims=True)
    
    return np.clip(Ce, 0.001, 0.999)  