# -*- coding: utf-8 -*-
"""
Created on Mon Nov. 27 2023

@author: taku.yamagata

PbRL Agent class (Preference-based RL Agent)
"""
import numpy as np
import torch
from scipy.special import psi
import pickle

import mylib

class agent():

    def __init__(self, algID, nStates, nActions, nTrainers, lr=1e-2, beta_fixed=None, sgd_all=True, fixed_beta_value=1.0, fixed_trainer_id=0):
        self.algID    = algID
        self.nStates  = nStates
        self.nActions = nActions
        self.nTrainers = nTrainers
        self.lr       = lr           # learning rate for SGD
        self.beta_fixed = beta_fixed # Fixed beta - without beta (rationality) estimation
        self.sgd_all = sgd_all       # SGD flag, True: Apply SGD for utility (and rationality)
        self.fixed_beta_value = torch.tensor(fixed_beta_value, requires_grad=False)
        self.fixed_trainer_id = fixed_trainer_id
        
        self.reset()
        return
        
        
    def reset(self):
        self.gamma    = 0.9  # discount factor
        self.alpha    = 0.05 # learning rate
        self.eps      = 0.1  # e-greedy policy parametre (probability to select exploration action)
        
        self.Q        = np.zeros([self.nStates, self.nActions]) # value function - Q
        self.prev_obs = None                                    # previous observation
        self.prev_act = None                                    # previous action
        self.valid_state_flags = np.zeros(self.nStates)         # set one where received feedback
        
        self.w        = torch.zeros((self.nTrainers, self.nStates, self.nActions, self.nActions), requires_grad=False)
                                    # preference feedback table : w[k, s, a1, a2] = #feedback
                                    #       k: trainer index
                                    #       s: state
                                    #       a1, a2: actions (a1 is better than a2)
        self.u = torch.zeros((self.nStates, self.nActions), requires_grad=self.sgd_all)
                                    # utility table
        
        self.tempConst = 1.5         # Temperature constant for Bolzmann exploration policy
        if self.beta_fixed is None:
            self.beta = torch.ones(self.nTrainers, requires_grad=True)
        else:
            self.beta = torch.tensor(self.beta_fixed)
        if (self.beta_fixed is None) and (self.fixed_trainer_id is not None):
            with torch.no_grad():
                self.beta[self.fixed_trainer_id] = self.fixed_beta_value
            
        if self.sgd_all:
            self.optim = torch.optim.Adam([self.u, self.beta], lr=self.lr, weight_decay=1e-4)
        else:
            self.optim = torch.optim.Adam([self.beta], lr=self.lr, weight_decay=1e-4)
        return
    
    
    def act(self, obs, rw, done, fb, update_utility=False):

        action = 0 # default action.
        
        # state-action wise Expectation Maximisation based C estimation agent
        if self.algID == 'tabQL_pb_base':
            action = self.tabQL_pb_base(obs, rw, done, fb, update_utility)
        
        # Q-learning (no feedback)
        elif self.algID == 'tabQLgreedy':
            action = self.tabQLgreedy(obs, rw)
        
        else:
            raise ValueError(f'{self.algID}: Invalid algorithm option ID')
        
        # store the current observation/action for the next call
        self.prev_obs = obs
        self.prev_act = action

        return action
    
    
    def _collect_feedback(self, feedback_list):
        assert len(feedback_list) == self.nTrainers
        
        # collect feedback and update self.w 
        for k, fb in enumerate(feedback_list):
            if len(fb) > 0:
                if len(fb[1]) == 1:
                    # in the case of getting the best action as feedback
                    self.w[k, fb[0], fb[1][0], np.arange(self.nActions)!=fb[1][0]] += 1            
                else:
                    self.w[k, fb[0], fb[1][0], fb[1][1]] += 1 
                # update valid state flag
                self.valid_state_flags[fb[0]] = 1
                                
    # Tabular one step Temporal Difference
    def tabQLgreedy(self, obs, rw):
       
        # initialise Q
        if self.Q is None:
            self.Q = np.zeros([self.nStates, self.nActions]) # initialise Q function (s,a)
            
        curr_state_idx = obs
        
        # check if this is the first time...
        if self.prev_obs is not None:
            # one step TD algorithm
            prev_state_idx = self.prev_obs
            prev_action_idx = self.prev_act
            
            td_err = (rw + self.gamma * max(self.Q[curr_state_idx,:])) - self.Q[prev_state_idx, prev_action_idx] 
            self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
            
        # Greedy policy (always select the best action)
        action = np.argmax(self.Q[curr_state_idx,:])

        return action

    def tabQL_pb_base(self, obs, rw, done, fb, update_utility):
        """
        Tabular Q-learning with preference-based policy shaping -- no beta (rationality) estimation.
        """
        
        curr_state_idx = obs

        # Boltzmann exploration policy
        l_pr = torch.tensor(self.Q[curr_state_idx,:]/self.tempConst)
        
        # policy shaping
        l_pr += self.u[curr_state_idx,:] - torch.sum(self.u[curr_state_idx,:])
        
        m = torch.max(l_pr)
        logsum = m + torch.log( torch.sum( torch.exp(l_pr-m) ) )          
        l_pr = l_pr - logsum # mylib.logsum(l_pr)
    
        pr = torch.exp(l_pr)
        
        # decide action based on pr[] probability distribution
        # action = np.min( np.where( np.random.rand() < np.cumsum(pr) ) )
        action = torch.argmax(pr) # greedy policy       

        
        # check if this is the first time...
        if self.prev_obs is not None:
            # one step TD algorithm
            prev_state_idx = self.prev_obs
            prev_action_idx = self.prev_act
            
            max_a_idx = torch.argmax(pr)
            if done:
                td_err = (rw) - self.Q[prev_state_idx, prev_action_idx]             
            else:
                td_err = (rw + self.gamma * self.Q[curr_state_idx, max_a_idx]) - self.Q[prev_state_idx, prev_action_idx] 
            self.Q[prev_state_idx, prev_action_idx] += self.alpha * td_err
        
            # Human feedback updates
            self._collect_feedback(fb)
        
        if update_utility:
            # Update utility estimations
            if not self.sgd_all:
                max_iterations = 1
                for n in range(max_iterations):    
                    u_prev = self.u.detach().numpy().copy()
                    with torch.no_grad():
                        for s in np.arange(self.nStates)[self.valid_state_flags==1]: # go though only on the valid states (to speed up simulation)
                            for a1 in range(self.nActions): # cordinate-accent dimension
                                # Zermelo's algorithm (beta=1.0 fixed)                       
                                # self.u[s,a1] = (1 + np.sum(self.w[:,s, a1,:])) / \
                                #                (1/(self.u[s,a1]+1) + np.sum((self.w[:, s, a1, :] + (self.w[:, s, :, a1]) / (self.u[s,a1] + self.u[s,:]))))
                                
                                # Newman's algorithm (beta=1.0 fixed)
                                # self.u[s,a1] = (1/(self.u[s,a1]+1) + np.sum(self.w[:,s, a1,:]*self.u[s,:] / (self.u[s,a1]+self.u[s,:]))) / \
                                #                (1/(self.u[s,a1]+1) + np.sum(self.w[:, s, :, a1] / (self.u[s,a1] + self.u[s,:])))

                                    # num1, num2, den2 = 0, 0, 0
                                    # for k in range(self.nTrainers):
                                    #     b = self.beta[k]
                                    #     num1 += b / (self.u[s,a1]**b + 1) + np.sum(b*self.w[k,s,a1,:]*self.u[s,:]**b/(self.u[s,a1]**b + self.u[s,:]**b))
                                    #     num2 += b * self.u[s,a1]**b        * (1/(self.u[s,a1]**b+1) + np.sum(self.w[k,s,:,a1]/(self.u[s,a1]**b + self.u[s,:]**b)))
                                    #     den2 += b**2 * self.u[s,a1]**(b-1) * (1/(self.u[s,a1]**b+1) + np.sum(self.w[k,s,:,a1]/(self.u[s,a1]**b + self.u[s,:]**b)))
                                    # if np.isnan((num1 - num2) / (-den2)):
                                    #     print('nan detected')                            
                                    # self.u[s,a1] += 0.5 * (num1 - num2) / (-den2)

                                    # update utility
                                    num1, den1 = 0, 0
                                    for k in range(self.nTrainers):
                                        b = self.beta[k]
                                        # num1 += b / (self.u[s,a1]**b + 1) + np.sum(b*self.w[k,s,a1,:]*self.u[s,:]**b/(self.u[s,a1]**b + self.u[s,:]**b))
                                        # den1 += b * self.u[s,a1]**(b-1) * (1/(self.u[s,a1]**b+1) + np.sum(self.w[k,s,:,a1]/(self.u[s,a1]**b + self.u[s,:]**b)))
                                        num1 += b / (torch.exp(self.u[s,a1]*b) + 1) + \
                                                torch.sum(b*self.w[k,s,a1,:]*torch.exp(self.u[s,:]*b)/(torch.exp(self.u[s,a1]*b) + torch.exp(self.u[s,:]*b)))
                                        den1 += b * torch.exp(self.u[s,a1]*(b-1)) * (1/(torch.exp(self.u[s,a1]*b)+1) + \
                                                torch.sum(self.w[k,s,:,a1]/(torch.exp(self.u[s,a1]*b) + torch.exp(self.u[s,:]*b))))
                                        if np.isnan(num1.detach().numpy()/den1.detach().numpy()):
                                            print('nan!')
                                    self.u[s,a1] = torch.log(num1) - torch.log(den1)
                    
                    if np.max(np.abs(u_prev-self.u.detach().numpy())) < 0.001:
                        break

            # update rationality (beta)
            if (self.beta_fixed is None) or (self.sgd_all):
                max_iterations = 10
                for n in range(max_iterations):                                
                    self.optim.zero_grad()
                    loss = 0
                    
                    for k in range(self.nTrainers):
                        b = self.fixed_beta_value if (k == self.fixed_trainer_id) else self.beta[k]
                        for s in np.arange(self.nStates)[self.valid_state_flags==1]: # go though only on the valid states (to speed up simulation)
                            for a1 in range(self.nActions): # cordinate-accent dimension           
                                loss -= torch.sum(self.w[k,s,a1,:] * (b * self.u[s,a1] - \
                                                                     torch.log(torch.exp(b*self.u[s,a1]) + torch.exp(b*self.u[s,:]))))   
                
                    loss = loss / torch.sum(self.w)
                    loss.backward()
                    self.optim.step()
            if (self.beta_fixed is None) and (self.fixed_trainer_id is not None):
                with torch.no_grad():
                    self.beta[self.fixed_trainer_id] = self.fixed_beta_value
                        
                
        return action

    def save(self, fname):
        with open(fname + '.pkl', 'wb') as fid:
            pickle.dump([self.Q, self.w, self.u, self.beta], fid)
        return
    
    def load(self, fname):
        with open(fname, 'rb') as fid:
            Q, w, u, beta = pickle.load(fid)
            self.Q = Q
            self.w = w
            self.u = u
            self.beta = beta
        return