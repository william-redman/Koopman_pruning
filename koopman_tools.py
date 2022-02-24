#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:09:45 2022

@author: wtredman
"""
import torch
import numpy as np
import scipy 

def W2XY(W):
    # Turns the whole weight history matrix to two matrices X and Y, which are 
    # offset by one time-step. Just a convienence. 
    X = W[:, :-1]
    Y = W[:, 1:]
    return X, Y

def ExactDMD(W): 
    # As implemented by Tu et al. 2014. "On Dynamic Mode Decomposition". This 
    # function only returns the first Koopman mode (i.e. that corresponding to
    # \lambda = 1). If you want to compute all Koopman modes, uncomment out 
    # the section labelled 'All Koopman Modes'. Do be warned, it is not 
    # optimized, so it can take some time. 
    
    X, Y = W2XY(W)
    X = np.asmatrix(X)
    Y = np.asmatrix(Y)
    
    U, s, Vh = scipy.linalg.svd(X, full_matrices=False) 
    S = np.diag(s)
    U = np.asmatrix(U)
    S = np.asmatrix(S)
    Vh = np.asmatrix(Vh)

    A_tilde = U.H * Y * Vh.H * np.linalg.inv(S)

    lam, w = np.linalg.eig(A_tilde)
    fixed_pt_id = np.argmin(np.abs(lam - 1.0))
    w = np.asmatrix(w)
    
    YViS = Y * Vh.H * np.linalg.inv(S)
    
    # Koopman mode corresponding to \lambda = 1. Used for Koopman magnitude
    # pruning. 
    phi = (1./lam[fixed_pt_id]) *  YViS * w[:, fixed_pt_id]
    
    # --------  All Koopman Modes ----------------------------------------#
    # Uncomment if you want to compute all the Koopman modes
    #all_phi = np.zeros(YViS.shape)
    #for ii in range(0, YViS.shape[1]):
    #   print(ii)
    #   all_phi[:, ii] = np.squeeze((1./lam[ii]) *  YViS * w[:, ii])
    #all_phi = np.array(all_phi)
    #all_phi = np.float32(all_phi)
    
    return phi

def map_fixed_pt_importances(fixed_pt, importances):
    # Maps the found Koopman mode (in the case of Koopman magnitude pruning, 
    # the fixed point) to the ShrinkBench variable called importances.     
    new_importances = importances
    counter = 0
    
    for key in importances.keys():        
        n_weights = np.prod(importances[key]['weight'].shape)
        new_importances[key]['weight'] = np.reshape(fixed_pt[counter:(counter + n_weights)], importances[key]['weight'].shape)
        counter = counter + n_weights
        
        try: 
            n_bias = importances[key]['bias'].shape[0]
            new_importances[key]['bias'] = np.reshape(fixed_pt[counter:(counter + n_bias)], importances[key]['bias'].shape)
            counter = counter + n_bias
        except:
            print('no bias')
    print(counter)        
    return new_importances
        
