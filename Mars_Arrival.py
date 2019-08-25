#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:08:28 2019

@author: haider
"""

import numpy as np
import Transf
from scipy.integrate import solve_ivp

def f_mars(t, r):
    "2BP equation of motion about Mars in a Mars planetocentric frame."
    
    mu = 4.282E4
    RR = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    dRExdt = r[3]
    dREydt = r[4]
    dREzdt = r[5]
    dVExdt = (-mu/(RR)**3) * r[0]
    dVEydt = (-mu/(RR)**3) * r[1]
    dVEzdt = (-mu/(RR)**3) * r[2]
    
    
    return np.array([dRExdt, dREydt, dREzdt, dVExdt, dVEydt, dVEzdt])

def events(t, r):
    #Events function to determine when the SOI is crossed in order to patch conics
    R_SOI_E = 0.576E6

    R = (r[0]**2 + r[1]**2 + r[2]**2)**0.5
    return R - R_SOI_E


def Eclip2Plan(pos, vel):
    
    "Second method of conversion: From mean ecliptic to ICRF to body inertial"
    #First convert the ecliptic vectors into the ICRF by rotating about the 
    #obliquity of the ecliptic w.r.t Mars (25deg)
    
    #Define obliquity of ecliptic, and RA/Dec of planets spin axis
    
    ep = np.deg2rad(25)
    a = (np.pi/2 + np.deg2rad(317.269))
    d = np.pi/2 - np.deg2rad(54.433)
    
    #Define rotation from ecliptic to ICRF:
    
    RotICRF = np.array([[1,0,0],[0,np.cos(ep),np.sin(ep)],[0,-np.sin(ep),np.cos(ep)]])
    
    R_ICRF = np.matmul(RotICRF, pos)
    V_ICRF = np.matmul(RotICRF, vel)
    
    RotX = np.array([[1,0,0],[0, np.cos(d), np.sin(d)],[0, -np.sin(d), np.cos(d)]])
    RotZ = np.array([[np.cos(a),np.sin(a),0], [-np.sin(a), np.cos(a),0], [0,0,1]])
    
    #Construct the rotation matrix sequence for ICRF to BI:
    RotBI = np.matmul(RotX, RotZ)
    
    R_BI = np.matmul(RotBI, R_ICRF)
    V_BI = np.matmul(RotBI, V_ICRF)
    
    return R_BI, V_BI

def Arrival(pos, vel, rp):
    """Calculate orbital elements of the arrival hyperbola given Vinf (in ecliptic)
    and the desired perigee radius. Then compute hyperbolic velocity at perigee and
    propagate this to determine the arrival conic"""
    rmars = 3389.5

    mu = 4.282E4
    vinf = np.linalg.norm(vel)
    
    #Compute semi-major axis, eccentricity, and beta angle (between rp and asymptote)
    a = -mu/vinf
    e = 1 - rp/a
    beta = np.arccos(1/e)

    #Transform Vinf from ecliptic coordinates to body-inertial:
    
    R_BI, V_BI = Eclip2Plan(pos, vel)
    
    #Compute first rotation angle phi1:
    
    phi1 = np.arctan(V_BI[1]/V_BI[0])
    
    #Rotate to x'' system using angle phi1 about z axis of BI frame, followed 
    #by angle phi2 about y' vector:
    
    phi2 = np.arctan((V_BI[0]*np.cos([phi1]) + V_BI[1]*np.sin(phi1))/V_BI[2])
    phi2 = phi2[0]

    rot1 = np.array([[np.cos(phi1),np.sin(phi1),0], [-np.sin(phi1),np.cos(phi1),0], [0,0,1]])
    
    rot2 = np.array([[np.cos(phi2),0,-np.sin(phi2)], [0,1,0], [np.sin(phi2),0,np.cos(phi2)]])
    
    #Define the rotation vector from x -> x'', as well as the inverse of x'' -> x
    
    RotMat = np.matmul(rot2, rot1)
    RotMat_inv = np.linalg.inv(RotMat)
    
    #Now find the position of perigee in x'' and the orbit elements:
    #Define psi = 3pi/2 which corresponds to the minimum inclination trajectory
    
    psi = np.pi * 1.5
    
    R_p1 = rp * np.array([np.sin(beta)*np.cos(psi), np.sin(beta)*np.sin(psi), np.cos(beta)])
    
    R_p = np.matmul(RotMat_inv, R_p1)

    #Now use this position to compute the remaining orbital elements:
    
    #Eccentricity vector:
    ecc = e * R_p/np.linalg.norm(R_p)

    #Specific orbital momentum vector:
    h_mag = np.sqrt(a * mu * (1-e**2))
    
    unit_h = np.cross(R_p, V_BI)/np.linalg.norm(np.cross(R_p, V_BI))
    
    h = h_mag * unit_h
    
    i = np.rad2deg(np.arccos(h[2]/h_mag))
    
    #RAAN:
    K = np.array([0, 0, 1])
    N = np.cross(K, h)
    N_mag = np.linalg.norm(N)

    Om = np.rad2deg(np.arccos(N[0]/N_mag))

    if N[1] < 0:

        Om = 360 - Om
    
    #Argument of perigee:
    w = np.rad2deg(np.arccos(np.dot(N, ecc)/(N_mag*e)))

    if ecc[2] < 0:

        w = 360 - w
    
    State = np.array([h_mag, a, e, Om, i, w])
    #Decinf = print(np.arcsin(V_BI[2]/np.linalg.norm(V_BI))*180/3.141)
    
    #Get the state at the perigee of hyperbola and propagate this
    
    RPTEST, VPTEST = Transf.Orb2State(State[0], State[2], State[4], State[3], State[5], 0, mu)

    init_state = np.reshape(np.array([RPTEST, VPTEST]), 6)
    
    TOF = 20000
    t_array = np.linspace(0, TOF, 5000)
    
    STATE = solve_ivp(f_mars, (0, TOF), init_state, method = 'RK23', t_eval=t_array)
    
    Pos_Hyp = STATE.y
    
    #Propagate the circular orbit at the desired altitude and compute dVs
    
    v_circ = np.sqrt(mu/rp)
    vhyp = np.linalg.norm(VPTEST)

    dV_Circ = vhyp - v_circ

    #Get orbital state vector of circular orbit using RAAN and w of hyperbola
    h_leo = (rp) * np.sqrt(mu/(rp))
    LEO_STATE = Transf.Orb2State(h_leo, 0, i, State[3], State[5], 0, mu)

    Leo_init_state = np.reshape(np.array([LEO_STATE[0], LEO_STATE[1]]), 6)
    t_leo = 2*np.pi * np.sqrt(rp**3/mu)
    
    t_array_leo = np.linspace(0, t_leo, 3000)
    
    STATE_LEO = solve_ivp(f_mars, (0, t_leo), Leo_init_state, method='RK23', t_eval=t_array_leo)

    Pos_LEO  = STATE_LEO.y
    
    print("""
          """)
    print("===================== ARRIVAL CONDITIONS =======================")
    print('')
    print('Arrival Eccentricity:            ', e)
    print('Arrival Inclination:             ', i)
    print('Arrival Orbit Altitude [km]:     ', rp-rmars)
    print('Injection deltaV [km/s]:         ', dV_Circ)
    
    
    
    
    #return State, R_p, V_BI
    return Pos_Hyp, R_BI, Pos_LEO, V_BI


    
