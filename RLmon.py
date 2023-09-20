# -*- coding: utf-8 -*-
"""
 RL monitor class
 Monitor RL results 

 Note:  
"""
import numpy as np

"""
Episodic RL monitor
"""
class RLmon(object):
    
    def __init__(self, trial_count, episode_count, numData=1):
        self.episode_count = episode_count
        self.numData = numData
        # prepare a buffer for averaging reward
        self.ave   = np.zeros([numData, episode_count])
        self.aveSq = np.zeros([numData, episode_count])
        self.raw   = np.zeros([numData, episode_count, trial_count])
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