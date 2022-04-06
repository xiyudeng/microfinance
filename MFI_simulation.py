# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 13:27:40 2022

@author: Tian
"""


############################ importing libraries ##############################

import numpy as np
import matplotlib.pyplot as plt
# import multiprocessing as mp


########################### function to generate s ############################

def random_apc_info(n_apcs, ninfo, nempty):
    
    # generate s
    n_bins = 100
    s1_prcnt = np.append(np.linspace(0, 1/n_bins, n_bins),0.5)
    s1_num = np.floor(s1_prcnt*n_apcs)
    s1_lb = np.linspace(0, 4, n_bins+1)
    s = 4*np.random.rand(n_apcs,ninfo)
    for i in range(1,n_bins+1):
        s[int(s1_num[0:i-1].sum()+1):int(s1_num[1:i-1].sum()+s1_num[i]),:] = \
            s1_lb[i-1] + (1/n_bins)*\
            s[int(s1_num[1:i-1].sum()+1):int(s1_num[1:i-1].sum()+s1_num[i]),:]
    s = np.random.permutation(s)
    
    # calculate return probability
    p = np.mean(s,1)/4
    
    # assign empty entry
    emty_idx = np.random.choice(s.size,nempty,replace=False)
    s_shape = s.shape
    s = s.reshape([s.size,1])
    s[emty_idx] = np.nan
    s = s.reshape(s_shape)
    
    # return s and p
    return s,p


###### function to calcualte the derevative of the probability function #######

def partial_pi_partial_Q(s,Q,Aid,ninfo,N,L_form,k,realmin,realmax):
    
    # initiation
    del_pi = np.zeros((N, 2*ninfo))
    
    # calculate exp(Q) for Q:Nxninfo
    exp_Q = np.exp((k*Q))
    exp_Q[exp_Q==0] = realmin
    exp_Q[np.isinf(exp_Q)] = realmax
    
    exp_plus = exp_Q + 1
    exp_sqr = exp_plus**2
    exp_sqr[exp_sqr==0] = realmin
    exp_sqr[np.isinf(exp_sqr)] = realmax
    
    # pick L_form
    if L_form=='A':
        par_del = exp_Q/exp_sqr
    elif L_form=='B':
        par_del = k/exp_Q; par_del[par_del>k] = k
    elif L_form=='C':
        par_del = 2*exp_Q/exp_sqr
    
    # calculate del_pi
    del_pi[Aid,0:ninfo] = s[Aid,:]*par_del[Aid,:]
    del_pi[Aid,ninfo:] = par_del[Aid,:];
    # del_pi[nAid,0:ninfo] = -1*s[nAid,:]*par_del[nAid,:];
    # del_pi[nAid,ninfo:] = -1*par_del[nAid,:];
    del_pi[del_pi==0] = realmin; del_pi[np.isnan(del_pi)] = realmin;
    del_pi[np.isposinf(del_pi)] = realmax
    del_pi[np.isneginf(del_pi)] = -realmax;
    
    # return
    return del_pi
    

##################### gradient ascent optimization ############################

def gradient_ascent(N,s,p,ninfo,return_variable,\
                    k,L_form,c,e,\
                    phi_now,eps_now,alpha,R_proposed_cum,Nt_sum):
    
    # set floating numbers
    realmin = np.finfo('d').tiny
    realmax = np.finfo('d').max
    
    # calculate Q
    s_phi = s*phi_now
    Q = s_phi + eps_now; Q[np.isnan(Q)] = 0
    Q[np.isposinf(Q)] = realmax
    Q[np.isneginf(Q)] = -realmax
    
    # policy pi
    exp_Q = np.exp((k*Q).mean(axis=1))
    exp_Q[exp_Q==0] = realmin
    exp_Q[np.isinf(exp_Q)] = realmax
    
    # pick L_form
    if L_form=='A':
        pie = exp_Q/(1+exp_Q)
    elif L_form=='B':
        pie = 1-(1/exp_Q)
        # pie[pie<0] = 0
    elif L_form=='C':
        pie = (2*exp_Q/(1+exp_Q))-1
    pie[pie==0] = realmin
    pie = pie.reshape([pie.size,1])
    
    # make decision
    decision_variable = np.random.rand(N,1)
    A = (decision_variable < pie)
    
    # calculate rewards
    R = np.zeros((N,1))
    R[np.logical_and(A==1,return_variable < p)] = c+e
    R[np.logical_and(A==1,return_variable >= p)] = -1+e
    
    # calculate del pi
    Aid = np.where(A == 1)[0] # index of the accepted applicants
    # nAid = np.where(A == 0)[0] # index of the rejected applicants
    del_pi = partial_pi_partial_Q(s,Q,Aid,ninfo,N,L_form,k,realmin,realmax)
    
    # decide Rbar
    Rbar = R_proposed_cum.sum()/Nt_sum
    
    # calculate F
    deltaR = R-Rbar
    F = (deltaR*del_pi/pie).mean(axis=0)
    F[np.isposinf(F)] = realmax
    F[np.isneginf(F)] = -realmax
    
    # update parameters
    phi_now = phi_now + alpha*F[0:ninfo]
    phi_now[np.isposinf(phi_now)] = realmax
    phi_now[np.isneginf(phi_now)] = -realmax
    eps_now = eps_now + alpha*F[ninfo:]
    eps_now[np.isposinf(eps_now)] = realmax
    eps_now[np.isneginf(eps_now)] = -realmax
    
    # summing rewards
    R_sum = R.sum()
    
    # acceptance and default numbers
    numA = A.sum()
    default_num = (R == (-1+e)).sum()
    
    # return the results
    return phi_now,eps_now,R_sum,numA,default_num



############################ function compare algorithms ######################

def MFI_simulation(case_option,MS_parameters,gradient_parameters,rndm):
    
    # decide randomness
    if not rndm:
        np.random.seed(1)
        
    
    ## case option ############################################################
    
    # reward rule
    c = case_option['c']
    e = case_option['e']
    
    # information parameters
    ninfo = case_option['ninfo']
    nempty_prcnt = case_option['nempty']
    
    # number of period
    t = case_option['t']
    
    
    ## multi-start optimization parameters ####################################
    
    nPartcl = MS_parameters['nPartcl']          # population size
    nkeep = MS_parameters['nkeep']              # number to keep
    MSO_itr = MS_parameters['itr']              # multi-start iterations
    
    
    ## gradient optimization parameters #######################################
    
    # form of pi
    L_form = gradient_parameters['L_form']
    k = gradient_parameters['k']
    
    # step size constant
    DG = gradient_parameters['DG']
    
    
    ## initialization #########################################################
    
    Nt = np.zeros((t,1)) # list to store number of applicants
    
    # control parameters
    phi = -10 + 20*np.random.rand(ninfo,nPartcl)
    phis = np.zeros((t,ninfo))
    eps = -10 + 20*np.random.rand(ninfo,nPartcl)
    eps_arr = np.zeros((t,ninfo))
    
    # perceptron parameters
    P = np.ones((ninfo,1))
    w = 0
    
    # initiate average cumulative rewards
    R_proposed_cum = np.zeros((t,1))
    R_prfct_cum = np.zeros((t,1))
    R_P_cum = np.zeros((t,1))
    R_all_cum = np.zeros((t,1))
    
    # initiate acceptance ratio
    global_max_numA = np.zeros((t,1))
    numA_P = np.zeros((t,1))
    numA_prfct = np.zeros((t,1))
    ratioAs = np.zeros((t,1))
    ratioAS_P = np.zeros((t,1))
    ratioAs_prfct = np.zeros((t,1))
    ratioAs_all = np.ones((t,1))
    
    # initiate default probability
    global_max_default_num = np.zeros((t,1))
    default_prob = np.zeros((t,1))
    default_num_P = np.zeros((t,1))
    default_prob_P = np.zeros((t,1))
    default_num_prfct = np.zeros((t,1))
    default_prob_prfct = np.zeros((t,1))
    default_num_all = np.zeros((t,1))
    default_prob_all = np.zeros((t,1))
    
    
    ## iterations #############################################################
    
    for t_idx in range(0,t):
        
        # generate number of applications and info
        N = np.random.randint(10000, 20000 + 1)
        Nt[t_idx] = N; Nt_sum = Nt.sum()
        nempty = np.ceil(nempty_prcnt*N*ninfo).astype(int)
        
        # generate applicants information
        [s,p] = random_apc_info(N, ninfo, nempty)
        p = p.reshape([N,1])
        
        # return probability
        return_variable = np.random.rand(N,1)
        
        
        ## approve all ########################################################
        
        # calculate rewards
        R_all = np.zeros((N,1))
        R_all[return_variable < p] = c+e
        R_all[return_variable >= p] = -1+e
        
        # calculate default probability
        default_num_all[t_idx] = (R_all == (-1+e)).sum()
        default_prob_all[t_idx] = (default_num_all).sum()/Nt_sum
        
        
        ## perfect decision ###################################################
        
        # making decision
        A_prfct = (p >= 0.95)
        
        # calculate rewards
        R_prfct = np.zeros((N,1))
        R_prfct[np.logical_and(A_prfct==1,return_variable < p)] = c+e
        R_prfct[np.logical_and(A_prfct==1,return_variable >= p)] = -1+e
        
        # calculate acceptance probability
        numA_prfct[t_idx] = (R_prfct>0).sum()
        ratioAs_prfct[t_idx] = (numA_prfct).sum()/Nt_sum
        
        # calculate default probability
        default_num_prfct[t_idx] = (R_prfct == (-1+e)).sum()
        default_prob_prfct[t_idx] = (default_num_prfct).sum()/numA_prfct.sum()
        
        
        ## perceptron #########################################################
        
        # calculate decision value
        sP = np.copy(s); sP[np.isnan(s)] = 0
        OP = (sP*P.T+w).sum(axis=1)
        
        # making decision
        A_P = np.array((OP > 0)).reshape([N,1])
        
        # calculate rewards
        R_P = np.zeros((N,1));
        R_P[np.where(\
             np.logical_and(A_P == 1,return_variable < p))[0]] = c+e;
        R_P[np.where(\
             np.logical_and(A_P == 1,return_variable >= p))[0]] = -1+e;
        
        # calculate acceptance probability
        numA_P[t_idx] = A_P.sum()
        ratioAS_P[t_idx] = numA_P.sum()/Nt_sum
        
        # calculate default probability
        default_num_P[t_idx] = (R_P == (-1+e)).sum()
        default_prob_P[t_idx] = default_num_P.sum()/numA_P.sum()
        
        # updating parameters
        neg_idx = np.where(np.logical_and(A_P == 1,return_variable >= p))[0]
        pos_idx = np.where(np.logical_and(A_P == 0,return_variable < p))[0]
        P = P - sP[neg_idx,:].sum(axis=0).reshape([P.size,1])
        w = w - neg_idx.size
        P = P + sP[pos_idx,:].sum(axis=0).reshape([P.size,1])
        w = w + pos_idx.size
        
        
        ## proposed ###########################################################
        
        # updating step size
        alpha = DG/np.sqrt(t_idx+1)
        
        if t_idx <= MSO_itr: # perform multi-start optimization
            
            # initiate variables
            R_sum = np.zeros((1,nPartcl))
            numA = np.zeros((1,nPartcl))
            default_num = np.zeros((1,nPartcl))
            
            # gradient update for each particle in parallel
            #with mp.Pool(processes=8) as pool:
            #    pool.starmap()
            
            # gradient update for each particle
            for p_idx in range(nPartcl):
                
                # current particle
                phi_now = phi[:,p_idx]
                eps_now = eps[:,p_idx]
                
                # call gradient ascent
                phi[:,p_idx],eps[:,p_idx],\
                 R_sum[:,p_idx],numA[:,p_idx],default_num[:,p_idx] = \
                                gradient_ascent(N,s,p,ninfo,return_variable,\
                                k,L_form,c,e,\
                                phi_now,eps_now,alpha,R_proposed_cum,Nt_sum)
                                    
                                    
            
            # pick the maximum
            global_max_R = R_sum.max()
            global_max_idx = np.where(R_sum == global_max_R)[1]
            global_max_phi = phi[:,global_max_idx][:,0]
            global_max_eps = eps[:,global_max_idx][:,0]
            
            # calculate acceptance probability
            global_max_numA[t_idx] = numA[:,global_max_idx[0]]
            ratioAs[t_idx] = global_max_numA.sum()/Nt_sum
            
            # calculate default probability
            global_max_default_num[t_idx] = default_num[:,global_max_idx[0]]
            default_prob[t_idx] = global_max_default_num.sum()/\
                global_max_numA.sum()
            
            # update particles
            srt_idx = R_sum.argsort()[:,::-1]
            phi[:,0:nkeep] = phi[:,srt_idx[:,0:nkeep]].reshape([ninfo,nkeep])
            phi[:,nkeep:] = -10 + 20*np.random.rand(ninfo,nPartcl-nkeep)
            eps[:,0:nkeep] = eps[:,srt_idx[:,0:nkeep]].reshape([ninfo,nkeep])
            eps[:,nkeep:] = -10 + 20*np.random.rand(ninfo,nPartcl-nkeep)
            
        else: # perform single gradient ascent
            
            # continue the max parameters
            phi_now = global_max_phi
            eps_now = global_max_eps
            
            # call gradient ascent
            global_max_phi,global_max_eps,global_max_R,\
             global_max_numA[t_idx],global_max_default_num[t_idx] = \
                            gradient_ascent(N,s,p,ninfo,return_variable,\
                            k,L_form,c,e,\
                            phi_now,eps_now,alpha,R_proposed_cum,Nt_sum)
                                
            # calculate acceptance probability
            ratioAs[t_idx] = global_max_numA.sum()/Nt_sum
            
            # calculate default probability
            default_prob[t_idx] = global_max_default_num.sum()/\
                global_max_numA.sum()
        
        # store maximum
        phis[t_idx,:] = global_max_phi
        eps_arr[t_idx,:] = global_max_eps
        
        
        ## calculate cumulative rewards over time #############################
        
        if t_idx == 0:
            R_proposed_cum[t_idx] = global_max_R
            R_prfct_cum[t_idx] = R_prfct.sum()
            R_P_cum[t_idx] = R_P.sum()
            R_all_cum[t_idx] = R_all.sum()
        else:
            R_proposed_cum[t_idx] = \
                (global_max_R.sum()+(R_proposed_cum[t_idx-1]*(t_idx-1)))/t_idx
            R_prfct_cum[t_idx] = \
                (R_prfct.sum()+(R_prfct_cum[t_idx-1]*(t_idx-1)))/t_idx
            R_P_cum[t_idx] = \
                (R_P.sum()+(R_P_cum[t_idx-1]*(t_idx-1)))/t_idx
            R_all_cum[t_idx] = \
                (R_all.sum()+(R_all_cum[t_idx-1]*(t_idx-1)))/t_idx
        
        
        
    ## compile output #########################################################
    
    # compile cumulative rewards
    R_cum = {'proposed' : R_proposed_cum, \
             'all' : R_all_cum, \
             'perfect' : R_prfct_cum, \
             'perceptron' : R_P_cum}
    
    # compile acceptance ratio
    acceptance = {'proposed' : ratioAs, \
                  'all' : ratioAs_all, \
                  'perfect' : ratioAs_prfct, \
                  'perceptron' : ratioAS_P}
    
    # compile default probability
    default = {'proposed' : default_prob, \
               'all' : default_prob_all, \
               'perfect' : default_prob_prfct, \
               'perceptron' : default_prob_P}
    
    
    ## return results #########################################################
    return R_cum,acceptance,default
    


################################ main code ####################################
if __name__ == '__main__':
    
    # case option
    case_option = {'c' : 0.25, \
                   'e' : 0, \
                   'ninfo' : 100, \
                   't' : 1000, \
                   'nempty' : 0}
    
    # parameters for multi-start optimization
    MS_parameters = {'nPartcl' : 10, \
                     'nkeep' : 5, \
                     'itr' : 50}
    
    # parameters for gradient ascent optimization
    gradient_parameters = {'L_form' : 'B', \
                           'k' : 1, \
                           'DG' : 100}
    
    [R_cum,acceptance,default] = \
            MFI_simulation(case_option,MS_parameters,gradient_parameters,0);
    
    
    # plt.hist(np.resize(s, (n_apcs*ninfo,1)))
    # plt.hist(p)
    
    plt.plot(R_cum['perceptron'],label='perceptron',\
             linewidth=2,color=(0.4940,0.1840,0.5560))
    plt.plot(R_cum['perfect'],label='perfect scenario',\
             linewidth=2,color='k')
    plt.plot(R_cum['proposed'],label='case '+gradient_parameters['L_form'],\
             linewidth=2,color='r')
    
    plt.legend(loc=4)
    plt.xlabel('lending period')
    plt.ylabel('average cumulative utility')

