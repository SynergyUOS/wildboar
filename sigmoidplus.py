# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 21:06:40 2022

@author: univSEOULGIS
"""
import numpy as np

def sigmoidplus(x, Ax1, Ay2, cProp):
    return (1 / (1 + np.exp( (x - Ax1) * cProp))) * Ay2

#%% test

