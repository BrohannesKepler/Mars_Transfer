#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 21:36:36 2019

@author: haider
"""

import numpy as np

def Eclip2Plan(pos, vel):
    "Convert state from mean ecliptic frame to the planetocentric"
    alpha_m = np.deg2rad(0)
    dec_m = np.deg2rad(0)
    
    #Define the Martian equatorial tilt:
    eps_m = np.deg2rad(23.5)
    
    rot_1 = np.array([[1,0,0],[0,np.cos(eps_m),np.sin(eps_m)],[0,-np.sin(eps_m),np.cos(eps_m)]])
    #
    pol = np.array([np.cos(alpha_m)*np.cos(dec_m), np.sin(alpha_m)*np.cos(dec_m), np.sin(dec_m)])
    
    #North pole unit vector in the mean ecliptic:
    n_p = np.matmul(rot_1, pol)
    
    crn = np.cross(n_p,pol)
    
    omvec = crn/np.linalg.norm(crn)
    
    vvec = np.cross(n_p,omvec)
    
    #Rotation matrix to convert from mean ecliptic to planetocentric:
    rotmat = np.array([[omvec[0], omvec[1], omvec[2]], [vvec[0], vvec[1], vvec[2]], [n_p[0], n_p[1], n_p[2]]])

    V_inf_plan = np.matmul(rotmat, vel)
    
    RAinf = np.arctan2(V_inf_plan[1], V_inf_plan[0])
    Decinf = np.arcsin(V_inf_plan[2]/np.linalg.norm(V_inf_plan))
    
    #Initial position in the planetocentric frame:
    
    r_init = np.matmul(rotmat, pos)
    v_init = np.matmul(rotmat, vel)

    return r_init, v_init

def Plan2Eclip(pos, vel):
    "Convert position and velocity from planetocentric to mean ecliptic"
    
    alpha_m = np.deg2rad(0)
    dec_m = np.deg2rad(0)
    
    #Define the Martian equatorial tilt:
    eps_m = np.deg2rad(23.5)
    
    rot_1 = np.array([[1,0,0],[0,np.cos(eps_m),np.sin(eps_m)],[0,-np.sin(eps_m),np.cos(eps_m)]])
    #
    #pol = np.array([np.cos(alpha_m)*np.cos(dec_m), np.sin(alpha_m)*np.cos(dec_m), np.sin(dec_m)])
    pol = np.array([0,0,1])
    #North pole unit vector in the mean ecliptic:
    n_p = np.matmul(rot_1, pol)
    #n_p = np.array([0,0.397,0.91775])
    crn = np.cross(n_p,pol)
    
    omvec = crn/np.linalg.norm(crn)
    
    vvec = np.cross(n_p,omvec)
    
    #Rotation matrix to convert from mean ecliptic to planetocentric:
    rotmat = np.array([[omvec[0], omvec[1], omvec[2]], [vvec[0], vvec[1], vvec[2]], [n_p[0], n_p[1], n_p[2]]])

    
    r_init = np.matmul(np.linalg.inv(rotmat), pos)
    v_init = np.matmul(np.linalg.inv(rotmat), vel)

    return r_init, v_init
    
    
    

def Hyp_Dep(V_inf, hyp_inc, rp, mu):
    "Battin's method for computing hyperbolic injection for an Earth departure"
    
    hyp_inc = np.deg2rad(hyp_inc)
    vinf = np.linalg.norm(V_inf)
    
    e = 1 + rp*vinf**2/mu
    
    h_dep = mu * (e**2 - 1)**0.5/vinf
    
    a_dep = (h_dep**2/mu) * 1/(e**2 - 1)
    
    #velocity at perigee:
    
    vp = (vinf**2 + 2*mu/rp)**0.5
    
    #True anomaly at infinity:
    
    theta_inf = np.arccos(-1/e)

    # Apply Battin's Algorithm to determine the RAAN and w of the injection 
    # point of the Hyperbola. - Battin Astronautical Guidance
    
    #Define the ecliptic coordinates:
    
    neq = np.array([0, 0.397, 0.9178])
    npol = np.array([0, 0, 1])
    
    crn = np.cross(neq,npol)
    
    omvec = crn/np.linalg.norm(crn)
    
    vvec = np.cross(neq,omvec)
    
    #Calculate the V_inf vector in equoatorial coorinates and find the RA/dec:
    
    Vinfeq = np.array([np.dot(V_inf, omvec), np.dot(V_inf, vvec), np.dot(V_inf, neq)])
    unit_Vinfeq = Vinfeq/np.linalg.norm(Vinfeq)
    
    rotmat = np.array([[omvec[0], omvec[1], omvec[2]], [vvec[0], vvec[1], vvec[2]], [neq[0], neq[1], neq[2]]])

    Vinfeq2 = np.matmul(rotmat, V_inf)
    unit_Vinfeq2 = Vinfeq2/np.linalg.norm(Vinfeq2)
    
    RAinf = np.arctan2(Vinfeq2[1],Vinfeq2[0])
    Decinf = np.arcsin(Vinfeq2[2]/np.linalg.norm(Vinfeq2))
    
    if hyp_inc < Decinf:
        hyp_inc = Decinf
    
    bbetinf = np.arccos(np.tan(Decinf)/np.tan(hyp_inc))
    alf =  np.pi - bbetinf + RAinf
    h_un = np.array([np.cos(alf)*np.sin(hyp_inc), np.sin(alf)*np.sin(hyp_inc), np.cos(hyp_inc)])
    
    nu_inf  = theta_inf
    r_p = rp*(np.cos(nu_inf)*unit_Vinfeq2 + np.sin(nu_inf)*np.cross(unit_Vinfeq2, h_un))
    
    unit_rp = r_p/np.linalg.norm(r_p)
    
    D = np.sqrt(mu/(rp*(1 + np.cos(nu_inf))) + vinf**2/4)
    
    VHyp = (D + 0.5*vinf)*unit_Vinfeq2 + (D - 0.5*vinf)*unit_rp
    
    vcirc = np.sqrt(mu/rp)
    vhyp = np.sqrt(vcirc**2+vinf**2)
    dv = vhyp - vcirc
    
    DV = dv * np.cross(h_un, r_p)/rp
    
    return (r_p, VHyp, DV)


def Hyp_Arr(V_inf, hyp_inc, rp, mu):
    
    "Similar to departure procedure but for the arrival conic given an Rp."
    
    hyp_inc = np.deg2rad(hyp_inc)
    vinf = np.linalg.norm(V_inf)
    
    e = 1 + rp*vinf**2/mu
    
    h_dep = mu * (e**2 - 1)**0.5/vinf
    
    a_dep = (h_dep**2/mu) * 1/(e**2 - 1)
    
    #velocity at perigee:
    
    vp = (vinf**2 + 2*mu/rp)**0.5
    
    #True anomaly at infinity:
    
    theta_inf = np.arccos(-1/e)
    beta = np.arccos(1/e)
    
    neq = np.array([0, 0.397, 0.9178])
    npol = np.array([0, 0, 1])
    
    crn = np.cross(neq,npol)
    
    omvec = crn/np.linalg.norm(crn)
    
    vvec = np.cross(neq,omvec)
    
    rotmat = np.array([[omvec[0], omvec[1], omvec[2]], [vvec[0], vvec[1], vvec[2]], [neq[0], neq[1], neq[2]]])

    Vinfeq2 = np.matmul(rotmat, V_inf)
    unit_Vinfeq2 = Vinfeq2/np.linalg.norm(Vinfeq2)
    
    RAinf = np.arctan2(Vinfeq2[1],Vinfeq2[0])
    Decinf = np.arcsin(Vinfeq2[2]/np.linalg.norm(Vinfeq2))
    
    bbetinf = np.arccos(np.tan(Decinf)/np.tan(hyp_inc))
    alf =  np.pi - bbetinf + RAinf
    h_un = np.array([np.cos(alf)*np.sin(hyp_inc), np.sin(alf)*np.sin(hyp_inc), np.cos(hyp_inc)])
    
    r_p = rp*(np.cos(beta)*unit_Vinfeq2 + np.sin(beta)*np.cross(unit_Vinfeq2, h_un))
    
    unit_rp = r_p/np.linalg.norm(r_p)
    
    D = np.sqrt(mu/(rp*(1 + np.cos(theta_inf))) + vinf**2/4)
    
    VHyp = (D + 0.5*vinf)*unit_Vinfeq2 - (D - 0.5*vinf)*unit_rp
    
    return (r_p, VHyp)

