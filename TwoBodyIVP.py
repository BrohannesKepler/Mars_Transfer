#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:59:26 2019

@author: haider
"""

import numpy as np

def f_earth(t, r):
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

def f_sun(t, r):
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

def f_capt(t, r):
    """Simulate two bodies moving under a central force field in order to determine SOI 
    computations."""
    mu = 1.327E11
    
    #Magnitude of positions of planet and S/C:
    RR_P = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    RR_SC = np.sqrt(r[6]**2+r[7]**2+r[8]**2)
    
    #Linearised planetary equation of motion:
    dRPxdt = r[3]
    dRPydt = r[4]
    dRPzdt = r[5]
    dVPxdt = (-mu/(RR_P)**3) * r[0]
    dVPydt = (-mu/(RR_P)**3) * r[1]
    dVPzdt = (-mu/(RR_P)**3) * r[2]
    
    #Linearised SC equation of motion:
    dRSCxdt = r[9]
    dRSCydt = r[10]
    dRSCzdt = r[11]
    dVSCxdt = (-mu/(RR_SC)**3) * r[6]
    dVSCydt = (-mu/(RR_SC)**3) * r[7]
    dVSCzdt = (-mu/(RR_SC)**3) * r[8]
    
    return np.array([dRPxdt, dRPydt, dRPzdt, dVPxdt, dVPydt, dVPzdt, dRSCxdt, dRSCydt, dRSCzdt, dVSCxdt, dVSCydt, dVSCzdt])
    
    
def events(t, r):
    #Events function to determine when the SOI is crossed in order to patch conics
    R_SOI_E = 0.924E6

    R = (r[0]**2 + r[1]**2 + r[2]**2)**0.5
    return R - R_SOI_E

def events_mars(t, r):
    #Mars SOI events function
    R_SOI_Mars = 0.576E6

    RP = (r[0]**2 + r[1]**2 + r[2]**2)**0.5
    RSC = (r[6]**2 + r[7]**2 + r[8]**2)**0.5
    return ((RSC-RP) - R_SOI_Mars)

def state_soi(t,r):
    return r[0]