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
    def __init__(self, M, K, lr=1e-3, sgd_all=False, nIterations=1, nSGDsteps=1):
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
        self.u = torch.zeros(M, requires_grad=sgd_all)
        self.beta = torch.ones(K, requires_grad=True)
        if sgd_all:
            self.optim = torch.optim.Adam([self.u, self.beta], lr=lr)
        else:
            self.optim = torch.optim.Adam([self.beta], lr=lr)
            self.w = torch.zeros(K,M,M) # preference feedback counter
        return
        
    def _update_pref_counter(self, pairs_list):
        # update preference counter based on a new preference feedback
        for k in range(self.K):
            for n in range(len(pairs_list[k])):
                self.w[k, pairs_list[k][n,0], pairs_list[k][n,1]] += 1
        return
    
    def _loss_func(self, pairs, k):
        # compute loss for pairs & trainer index k
        if False: #k == 0:
            # fixed 0-th trainer's beta=1.0
            loss = - torch.sum( 1.0*self.u[pairs[:,0]] - \
                                torch.log(torch.exp(1.0*self.u[pairs[:,0]]) + torch.exp(1.0*self.u[pairs[:,1]])) )
        else:
            loss = - torch.sum( self.beta[k] * self.u[pairs[:,0]] - \
                                torch.log(torch.exp(self.beta[k]*self.u[pairs[:,0]]) + torch.exp(self.beta[k]*self.u[pairs[:,1]])) ) 
        return loss
    
    def calc_loss(self, pairs_all):
        loss = 0
        for k in range(self.K):
            loss += self._loss_func(pairs_all[k,:,:], k)
        return loss                
                
    
    def update(self, pairs_list):
        assert len(pairs_list) == self.K
        
        for _ in range(self.nIterations):
            if not self.sgd_all:
                with torch.no_grad():
                    # update preference counter
                    self._update_pref_counter(pairs_list)
                    # update utility with MM algorithm (Newman's algorithm)
                    for m in range(self.M):
                        num, den = 0, 0
                        for k in range(self.K):
                            b = self.beta[k]
                            num += b/(torch.exp(b*self.u[m]) + 1) + \
                                    torch.sum(b*self.w[k,m,:]*torch.exp(b*self.u)/(torch.exp(b*self.u[m]) + torch.exp(b*self.u)))
                            den += b*torch.exp((b-1)*self.u[m]) * (1/(torch.exp(b*self.u[m]) + 1) + \
                                    torch.sum(self.w[k,:,m]/(torch.exp(b*self.u[m]) + torch.exp(b*self.u) )))
                        self.u[m] = torch.log(num/den)
        
            for n in range(self.nSGDsteps):    
                self.optim.zero_grad()
                loss = self.calc_loss(torch.tensor(np.array(pairs_list)))
                loss.backward()
                self.optim.step()
            

#########################################################################
# Main (top level) function
#########################################################################
def main(
        util = np.log([1.0, 1.1, 1.2, 0.8]),
        beta = np.array([1.0, 1.5, 0.3]),
        lr = 2e-3,
        sgd_all = False,
        max_time_steps = 1000,
        num_feedback_per_timestep = 100, # number of preferences per iteration per trainer
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
                                          nSGDsteps=num_SGD_steps_per_iteration)
    
    # perpare logging list
    u_list = []
    b_list = []
    rmse_u = []
    rmse_b = []
    
    for n in range(max_time_steps):
        pairs_list = [gen.gen_pref(num_feedback_per_timestep, k) for k in range(len(beta))]
        agent.update(pairs_list)
        
        u_list.append(agent.u.detach().numpy().copy())
        b_list.append(agent.beta.detach().numpy().copy())        
        rmse_u.append(np.sqrt( np.mean( (util-agent.u.detach().numpy())**2 ) ))
        rmse_b.append(np.sqrt( np.mean( (beta-agent.beta.detach().numpy())**2 ) ))
        
    return u_list, b_list, rmse_u, rmse_b

#########################################################################
if __name__ == '__main__':
    
    util = np.log([1.0, 1.1, 1.2, 0.8])
    beta = np.array([1.5, 1.0, 0.3])
    lr = 2e-3
    sgd_all = False # True: SGD only, False: SGD + MM algorithm
    max_time_steps = 1000
    
    u_list, b_list, rmse_u, rmse_b = main(
                                        util = util,
                                        beta = beta,
                                        lr = lr,
                                        sgd_all = sgd_all,
                                        max_time_steps = max_time_steps,
                                        )
    
    ################
    # Plot results
    ################
    fig,ax = plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(np.exp(u_list))
    ax[0].set_title('Estimated Utilities')
    ax[0].set_xlabel('learning time steps')
    ax[1].plot(b_list)
    ax[1].set_title('Estimated Rationality')
    ax[1].set_xlabel('learning time steps')
    x = [0, len(b_list)]
    y_util = [[u, u] for u in util]
    y_beta = [[b, b] for b in beta]
    ax[0].set_prop_cycle(None)
    ax[0].plot(x, np.exp(y_util).T, linestyle=':')
    ax[1].set_prop_cycle(None)
    ax[1].plot(x, np.array(y_beta).T, linestyle=':')
    
    fig.savefig('pb_test_1.png')
    