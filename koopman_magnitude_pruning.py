#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:26:24 2022

@author: wtredman
"""
import numpy as np
import os.path

from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin)
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    activation_importance)

from ..koopman.koopman_tools import ExactDMD, map_fixed_pt_importances


class KoopGlobalMagWeight(VisionPruning):
        
    def model_masks(self):
        W = self.model.W
        
        path = '/Users/wtredman/shrinkbench/pretrained/MNIST/Early training/Seed 1/Epoch 20'
        
        if not os.path.exists(path + '/fixed_pt.npy'):
            fixed_pt = ExactDMD(W)
            fixed_pt = np.array(fixed_pt)
            fixed_pt = np.expand_dims(fixed_pt, 1)
            fixed_pt = np.float32(fixed_pt)
            print('Fixed pt computed')
            np.save(path + '/fixed_pt.npy', fixed_pt)
        else:
            fixed_pt = np.load(path + '/fixed_pt.npy')    
            print('Fixed pt loaded')
        
        abs_fixed_pt = np.abs(fixed_pt)
        threshold = fraction_threshold(abs_fixed_pt, self.fraction)
        importances = map_fixed_pt_importances(abs_fixed_pt, self.params())
        masks = importance_masks(importances, threshold)
        
        return masks