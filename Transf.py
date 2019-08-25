# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:25:58 2018

@author: Haider
"""

import numpy as n

""" Transformations between state vector in ECI to Keplerian orbital elements\
    and vice-versa.

    ALL DISTANCE/VELOCITY UNITS IN KM AND KM/S

    MU IN KM3/S^2

    ANGULAR VALUES INPUT/OUTPUT IN DEGREES

"""

#mu = 398600


def State2Orb(R0, V0, mu):

    #Get absolute values of the position and velocity
    r = n.linalg.norm(R0)
    v = n.linalg.norm(V0)

    #Compute radial velocity
    vr = n.dot(R0, V0)/r

    #Compute specific orbital momentum 
    h = n.cross(R0, V0)
    h_mag = n.linalg.norm(h)

    #Inclination
    i = n.rad2deg(n.arccos(h[2]/h_mag))

    #RAAN:
    K = n.array([0, 0, 1])
    N = n.cross(K, h)
    N_mag = n.linalg.norm(N)

    Om = n.rad2deg(n.arccos(N[0]/N_mag))

    if N[1] < 0:

        Om = 360 - Om

    #Eccentricity
    ecc = n.cross(V0, h) - (mu * R0/r)
    ecc = 1/mu * ecc
    e_mag = n.linalg.norm(ecc)

    #Argument of perigee:
    w = n.rad2deg(n.arccos(n.dot(N, ecc)/(N_mag*e_mag)))

    if ecc[2] < 0:

        w = 360 - w

    #True anomaly:
    theta = n.rad2deg(n.arccos(n.dot(ecc, R0)/(e_mag*r)))

    if vr < 0:

        theta = 360 - theta

    #Semi-major axis:
    rp = h_mag**2/mu * 1/(1 + e_mag)
    ra = h_mag**2/mu * 1/(1 - e_mag)

    a = 0.5 * (ra + rp)
    
    State = n.array([h_mag, a, e_mag, Om, i, w, theta])
    print("\n")
    print("Orbital Elements:\n")
    print(h_mag)
    print(a)
    print(e_mag)
    print(Om)
    print(i)
    print(w)
    print(theta)
    print("\n")
    
    return State


def Orb2State(h, e, i, Om, w, theta, mu):
    
    # Transform the Keplerian orbital elements to a Cartesian state vector of 
    # position in km and velocity in km/s.
    
    #Convert all angles to radians:
    
    i = n.deg2rad(i)
    Om = n.deg2rad(Om)
    w = n.deg2rad(w)
    theta = n.deg2rad(theta)    
    
    #Calculate position and velocity in the perifocal frame (R_per, V_per):
    
    R_per = h**2/mu * 1/(1 + e*(n.cos(theta))) * n.array([n.cos(theta), n.sin(theta), 0])
    V_per = mu/h * n.array([-n.sin(theta), e + n.cos(theta), 0])
    
    #Rotation matrices that form the transformation matrix:
    
    R_w = n.array([[n.cos(w),n.sin(w),0], [-n.sin(w), n.cos(w),0], [0,0,1]])
    R_i = n.array([[1, 0, 0], [0, n.cos(i), n.sin(i) ], [0, -n.sin(i), n.cos(i)]])
    R_Om = n.array([[n.cos(Om),n.sin(Om),0], [-n.sin(Om), n.cos(Om),0], [0,0,1]])

    # Euler angle sequence:
    
    RwRi = n.matmul(R_w, R_i)
    QXx = n.matmul(RwRi, R_Om)

    # From perifocal to geocentric is the transpose of QXx

    QxX = n.transpose(QXx)

    R_State = n.matmul(QxX, R_per)
    V_State = n.matmul(QxX, V_per)
    
    #print("\n")
    #print("State vectors:\n")
    #print(R_State)
    #print(V_State)
    #print("\n")
    
    return R_State, V_State