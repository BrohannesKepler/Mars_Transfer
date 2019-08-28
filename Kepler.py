#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 21:39:07 2019

@author: haider
"""

import numpy as np



def Solve(M, e):
    
    "Solve Kepler's Equation using Newton's Method."

    ratio = 1
    nmax = 1000
    X = 0
    count = 0

    while abs(ratio) > 1E-8 and count <= nmax:
        
        fX = X - e * np.sin(X) - np.deg2rad(M)
        fXdot = 1 - e * np.cos(X)

        ratio = fX/fXdot
        
        X = X - ratio
        count = count + 1
        
    #Now, compute the true-anomaly:
    
    theta = 2 * np.arctan(np.sqrt((1 + e)/(1 - e)) * np.tan(X/2))

    theta = np.rad2deg(theta)
    
    if theta < 0:
        theta = theta + 360
        
    return theta



