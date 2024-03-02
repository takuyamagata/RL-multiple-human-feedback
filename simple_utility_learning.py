"""
Simple ranking task test script - 
    Learning the utility and rationalisation parameter jointly

Taku Yamagata (29/11/2023)
"""

import numpy as np
import torch
from torch.autograd.functional import hessian
import os
import matplotlib.pyplot as plt

class preference_generator:
    def __init__(self, util, beta,):
        self.u = util
        self.beta = beta
        self.M = len(util) # number of objects
        self.K = len(beta) # number of trainers (humans)
        
    def gen_pref(self, num, k):
        """_summary_
            Generate preference based on Bradley-Terry model
        Args:
            num: number of preferences to generate
            k  : trainer index Defaults to None: for all trainers.
        """
        assert k < self.K
        
        # pick pairs of objects uniformly randomly
        pairs = [np.random.choice(self.M, 2, replace=False) for n in range(num)]
        
        # put the prefer object first
        for n, pair in enumerate(pairs):
            pairs[n] = self._BT_model([self.u[pair[0]], self.u[pair[1]]], self.beta[k], obj=pair)
            
        return np.array(pairs)
            
    def _BT_model(self, utils, beta, obj=[0,1]):
        Pi = np.exp(np.array(utils) * np.array(beta))
        if Pi[0] / np.sum(Pi) > np.random.rand():
            order_index = [obj[0], obj[1]]
        else:
            order_index = [obj[1], obj[0]]
        return order_index
    
    
class learn_utility_and_rationality:
    def __init__(self, M, K, sgd_sample_size=100, lr=1e-3, sgd_all=True, nIterations=1, nSGDsteps=1, beta_fixed = False):
        # M: number of objects to rank
        # K: number of trainers
        # lr: learning rate for SGD
        # sgd_all: True - learns the utitlity and rationality with SGD
        #          False - Learns the utilith with MM algorithm, the rationality with SGD
        # nIterations: number of iterations for a single update() call
        # nSGDsteps: number of SGD steps per interations (nIterations*nSGDsteps SGD update per update() call)
        self.M = M
        self.K = K
        self.nIterations = nIterations
        self.nSGDsteps = nSGDsteps
        self.sgd_all = sgd_all
        self.sgd_sample_size = sgd_sample_size
        self.u = torch.zeros(M, requires_grad=sgd_all)
        self.beta_fixed = beta_fixed
        self.pairs_list = None  # preference feedback list (used for SGD)
        self.w = torch.zeros(K,M,M) # preference feedback counter
        if hasattr(beta_fixed, '__len__'):
            self.beta = torch.tensor(beta_fixed)
            if sgd_all:
                self.optim = torch.optim.Adam([self.u], lr=lr)
            else:
                self.optim = None
        else:    
            self.beta = torch.ones(K, requires_grad=True)
            if sgd_all:
                self.optim = torch.optim.Adam([self.u, self.beta], lr=lr)
            else:
                self.optim = torch.optim.Adam([self.beta], lr=lr)
            
        return
        
    def _update_pref_counter(self, pairs_list):
        # update preference counter based on a new preference feedback
        for k in range(self.K):
            for n in range(len(pairs_list[k])):
                self.w[k, pairs_list[k][n,0], pairs_list[k][n,1]] += 1
        return

    def _append_pairs_list(self, pairs_list):
        # append pairs_list to self.pairs_list
        if self.pairs_list is None:
            self.pairs_list = pairs_list
        else:
            for k in range(self.K):
                np.vstack((self.pairs_list[k], pairs_list[k]))
            
    def add_pairs_list(self, pairs_list):
        # update internal preference counter and preference list
        self._update_pref_counter(pairs_list)
        self._append_pairs_list(pairs_list)
        
    def _sample_pairs_list(self, num_samples):
        # sample pairs_list
        sampled_pairs_list = [ 
                              self.pairs_list[k][np.random.choice(
                                                            self.pairs_list[k].shape[0], 
                                                            np.minimum(num_samples, self.pairs_list[k].shape[0]),
                                                            replace=False),:] 
                              for k in range(self.K)
                             ]
        return sampled_pairs_list
    
    def _loss_func(self, pairs, k):
        # compute loss for pairs & trainer index k
        if k == 0:
            # fixed 0-th trainer's beta=1.0
            loss = torch.sum( torch.log(1.0 + torch.exp(1.0 * (self.u[pairs[:,1]]-self.u[pairs[:,0]]))) ) 
            # loss = - torch.sum( 1.0*self.u[pairs[:,0]] - \
            #                     torch.log(torch.exp(1.0*self.u[pairs[:,0]]) + torch.exp(1.0*self.u[pairs[:,1]])) )
        else:
            loss = torch.sum( torch.log(1.0 + torch.exp(self.beta[k] * (self.u[pairs[:,1]]-self.u[pairs[:,0]]))) ) 
            # loss = - torch.sum( self.beta[k] * self.u[pairs[:,0]] - \
            #                     torch.log(torch.exp(self.beta[k]*self.u[pairs[:,0]]) + torch.exp(self.beta[k]*self.u[pairs[:,1]])) ) 
        return loss
    
    def calc_loss(self, pairs_all):
        loss = 0
        for k in range(self.K):
            loss += self._loss_func(pairs_all[k,:,:], k)
        return loss                
                
    def update(self):                      
        for _ in range(self.nIterations):
            if not self.sgd_all:
                with torch.no_grad():
                    # update utility with MM algorithm (Newman's algorithm)
                    for m in range(self.M):
                        b_base = 1.0 #torch.min(self.beta)
                        num, den = 0, 0
                        for k in range(self.K):
                            b = self.beta[k]
                            num += b/(torch.exp(b*self.u[m]) + 1) + \
                                    torch.sum(b*self.w[k,m,:]*torch.exp(b*self.u)/(torch.exp(b*self.u[m]) + torch.exp(b*self.u)))
                            den += b*torch.exp((b-1)*self.u[m]) * (1/(torch.exp(b*self.u[m]) + 1) + \
                                    torch.sum(b*torch.exp((b-1)*self.u[m])*self.w[k,:,m]/(torch.exp(b*self.u[m]) + torch.exp(b*self.u) )))
                        self.u[m] = torch.log(num/den)
                        #     den += b*torch.exp((b-1)*self.u[m]) * (1/(torch.exp(b*self.u[m]) + 1) + \
                        #             torch.sum(b*torch.exp((b-1)*self.u[m])*self.w[k,:,m]/(torch.exp(b*self.u[m]) + torch.exp(b*self.u) ))/torch.exp(self.u[m]*(b-b_base)))
                        # self.u[m] = torch.log(num/den)/b_base
                        
            if self.optim is not None:
                sampled_pairs_list = self._sample_pairs_list(self.sgd_sample_size) # sample preference feedbacks for SGD
                for n in range(self.nSGDsteps):    
                    self.optim.zero_grad()
                    loss = self.calc_loss(torch.tensor(np.array(sampled_pairs_list)))
                    loss.backward()
                    self.optim.step()
            

#########################################################################
# Main (top level) function
#########################################################################
def main(
        util = np.log([1.0, 1.1, 1.2, 0.8]),
        beta = np.array([1.0, 1.5, 0.3]),
        beta_fixed = False,
        lr = 2e-3,
        sgd_all = True,
        max_time_steps = 2000,
        num_feedback_at_start = 0,     # number of preferences at the start of learning 
        num_feedback_per_timestep = 0, # number of preferences per iteration per trainer
        sgd_sample_size = 1000,
        num_iterations_per_timestep = 1,
        num_SGD_steps_per_iteration = 1,
):
    M = len(util) # number of objects to rank
    K = len(beta) # number of human trainers
    
    gen = preference_generator(util, beta)
    agent = learn_utility_and_rationality(M, 
                                          K, 
                                          lr=lr, 
                                          sgd_all=sgd_all, 
                                          nIterations=num_iterations_per_timestep, 
                                          nSGDsteps=num_SGD_steps_per_iteration,
                                          sgd_sample_size = sgd_sample_size,
                                          beta_fixed = beta_fixed,)
    
    # perpare logging list
    u_list = []
    b_list = []
    
    if num_feedback_at_start > 0:
        pairs_list = [gen.gen_pref(num_feedback_at_start, k) for k in range(len(beta))]
        agent.add_pairs_list(pairs_list)
    
    for n in range(max_time_steps):
        if num_feedback_per_timestep > 0:
            pairs_list = [gen.gen_pref(num_feedback_per_timestep, k) for k in range(len(beta))]
            agent._update_pref_counter(pairs_list)
            agent.pairs_list = pairs_list # replace
        
        agent.update()

        u_list.append(agent.u.detach().numpy().copy())
        b_list.append(agent.beta.detach().numpy().copy())        
        
    return u_list, b_list

#########################################################################
if __name__ == '__main__':
    
    util = np.log(np.exp([1.0, 1.1, 1.2, 0.7]))
    beta = np.array([1.0, .5, 2.0])
    beta_fixed = False # beta
    lr = 2e-3
    sgd_all = True # True: SGD only, False: SGD + MM algorithm
    max_time_steps = 10000
    num_feedback_at_start = 100
    num_feedback_per_timestep = 0
    sgd_sample_size = 25
    
    u_list, b_list, = main(
                        util = util,
                        beta = beta,
                        lr = lr,
                        sgd_all = sgd_all,
                        max_time_steps = max_time_steps,
                        num_feedback_at_start = num_feedback_at_start,
                        num_feedback_per_timestep = num_feedback_per_timestep,
                        sgd_sample_size = sgd_sample_size,
                        beta_fixed = beta_fixed,
                        )
    
    ################
    # Plot results
    ################
    target_util_mean = np.mean(np.array(util))
    for n in range(len(u_list)):
        u_list[n] = u_list[n] - np.mean(u_list[n]) + target_util_mean
    
    fig,ax = plt.subplots(1,2,figsize=(12,4))
    # ax[0].plot(np.exp(u_list))
    ax[0].plot(u_list)
    ax[0].set_title('Estimated Utilities')
    ax[0].set_xlabel('learning time steps')
    ax[1].plot(b_list)
    ax[1].set_title('Estimated Rationality')
    ax[1].set_xlabel('learning time steps')
    x = [0, len(b_list)]
    y_util = [[u, u] for u in util]
    y_beta = [[b, b] for b in beta]
    ax[0].set_prop_cycle(None)
    # ax[0].plot(x, np.exp(y_util).T, linestyle=':')
    ax[0].plot(x, np.array(y_util).T, linestyle=':')
    ax[1].set_prop_cycle(None)
    ax[1].plot(x, np.array(y_beta).T, linestyle=':')
    
    fname = f"pb_test-SGD_{sgd_all}-start_fb_{num_feedback_at_start}-per_timestep_fb_{num_feedback_per_timestep}.png"
    
    fig.savefig(fname)
    