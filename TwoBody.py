#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:59:26 2019

@author: haider
"""

import numpy as np

def f_earth(r, t):
    mu = 398600
    "Differential equations of motion under a central force field linearised\
    in first order form to be integrated using odeint."
    RR = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    #RR = np.linalg.norm(r)
    dRExdt = r[3]
    dREydt = r[4]
    dREzdt = r[5]
    dVExdt = (-mu/(RR)**3) * r[0]
    dVEydt = (-mu/(RR)**3) * r[1]
    dVEzdt = (-mu/(RR)**3) * r[2]
    
    
    return np.array([dRExdt, dREydt, dREzdt, dVExdt, dVEydt, dVEzdt])

def f_sun(r, t):
    mu = 1.327E11
    "Differential equations of motion under a central force field linearised\
    in first order form to be integrated using odeint."
    RR = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    #RR = np.linalg.norm(r)
    dRExdt = r[3]
    dREydt = r[4]
    dREzdt = r[5]
    dVExdt = (-mu/(RR)**3) * r[0]
    dVEydt = (-mu/(RR)**3) * r[1]
    dVEzdt = (-mu/(RR)**3) * r[2]
    
    
    return np.array([dRExdt, dREydt, dREzdt, dVExdt, dVEydt, dVEzdt])