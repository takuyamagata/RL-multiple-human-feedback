# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:43:24 2018

@author: taku.yamagata

Agent class
"""
import json
import numpy as onp
import jax.numpy as np
import jax


from functools import partial

from scipy.special import psi
import pickle

import library.learning_functions as lf
import library.utilities as ut

from config import environment, rl




class agent():

    def __init__(self, algorithm, a=1.0, b=1.0):
        self.reset(algorithm, a, b)
        return
        
        
    def reset(self, algorithm, a, b):
        
        self.Q        = np.zeros([rl['nStates'], rl['nActions']])    # initialise Q value function (s,a) 
        self.prev_obs = None         # previous observation
        self.prev_act = None         # previous action
        
        self.hp = self.hm = np.zeros([rl['nTrainer'], rl['nStates'], rl['nActions']])           # delta for human feedback
        
        self.Ce = np.clip(np.ones(rl['nTrainer']) * 0.5, 0.001, 0.999)

        self.alg = algorithm
        
        # Variational Inference parameters
        self.a = a # prior parameter for C
        self.b = b # prior parameter for C
        return
    
    
    def act(self,obs, fb, rw, done, update_Cest = None):
        
        if self.prev_obs is not None:
            self.hp,self.hm = ut.collect_feedback(fb,self.hp,self.hm)
            pr, self.Q, self.hp, self.hm ,ce  = self.alg(self.Q, self.hp, self.hm, self.Ce, self.prev_obs, self.prev_act, obs, rw, done, update_Cest = update_Cest)
            
            if update_Cest:
                self.Ce = ce
        else:
            pr = 0

        # pr =np.argmax(self.Q[obs,:])
        action = np.array(list(environment['actions'])[pr])
        self.prev_obs = obs
        self.prev_act = ut.map_array_to_number(action)
        return action
    

    def Cest_em(self):
        self.Ce = lf.Cest_em(self.Q,self.d,self.hp,self.hm,self.Ce)


    def estimateC(self, hp, hm, Ce, ave_absQ, ave_nFBs, type):
        l_pr = self.Q[self.prev_obs,:]/self.tempConst
        l_pr = ut.lognorm(l_pr)
        pr = np.exp(l_pr)
        
        p1q = pr[self.prev_act]
        p0q = np.sum(pr[np.arange(len(pr)) != self.prev_act])        
        p1 = p1q
        p0 = p0q        
        C = 0.5
        
        d = hp - hm
        for n in range(50):
            # M-step
            Cnxt = (p1 * hp[self.prev_obs,self.prev_act] + p0 * hm[self.prev_obs,self.prev_act]) / (hp[self.prev_obs,self.prev_act] + hm[self.prev_obs,self.prev_act])
            Cnxt = np.round(Cnxt * 100) / 100
            if C == Cnxt:
                break
            else:
                if Cnxt == 1.0:
                    C = 1.0 - np.finfo(np.float32).resolution
                elif Cnxt == 0.0:
                    C = np.finfo(np.float32).resolution
                else:
                    C = Cnxt
                
            # E-step
            if type == 1:
                # type1 (general case)
                l_p1 = np.log(p1q) + d[self.prev_obs,self.prev_act] * np.log(C)
                l_p0 = np.log(p0q) + d[self.prev_obs,self.prev_act] * np.log(1.0-C)
            else:
                # type2 (only one optimal action)
                l_p1 = np.log(p1q) + d[self.prev_obs,self.prev_act] * np.log(C) \
                                   + np.sum(d[self.prev_obs,np.arange(self.nActions)!=self.prev_act]) * np.log(1-C)
                l_p0 = -np.inf
                for i in np.arange(self.nActions):
                    if i != self.prev_act:
                        l_p0 = ut.logadd(l_p0,  \
                                            np.log(pr[i])   \
                                                + d[self.prev_obs, i] * np.log(C)    \
                                                + np.sum(d[self.prev_obs,np.arange(self.nActions)!=i]) * np.log(1-C) )
            l_p1_ = l_p1 - ut.logadd(l_p0, l_p1)
            l_p0_ = l_p0 - ut.logadd(l_p0, l_p1)
            p1 = np.exp(l_p1_)
            p0 = np.exp(l_p0_)              
            
        nFBs = np.sum(hp[self.prev_obs,:]) + \
                        np.sum(hm[self.prev_obs,:])
        absQ = np.sum(np.abs(self.Q[self.prev_obs,:]))
            
        # average C over (s,a)
        lr = nFBs*absQ / (ave_nFBs*ave_absQ) / 16
        lr = np.min([lr, 1])
        #lr = 1/32 # FIXED LR #
        Ce = Ce + (C - Ce) * lr

        ave_nFBs = ave_nFBs + (nFBs - ave_nFBs) * lr
        ave_absQ = ave_absQ + (absQ - ave_absQ) * lr
        
        return {'Ce':Ce, 'ave_nFBs':ave_nFBs, 'ave_absQ':ave_absQ}

    def save(self, fname):
        with open(fname + '.pkl', 'wb') as fid:
            pickle.dump([self.Q, self.hp,self.hm,self.Ce], fid)
        return
    
    def load(self, fname):
        with open(fname, 'rb') as fid:
            self.Q, self.hp,self.hm,self.Ce = pickle.load(fid)
        return
