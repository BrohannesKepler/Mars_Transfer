#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 19:59:36 2019

@author: haider
"""


import numpy as np
import Julian
import Transf
import Kepler


def main(planet_id, hr, day, month, year):
    mu = 1.327E11
    AU2KM = 1.49597770700E8
    
    """
    
    From planetary ephemeris, get the state vector in the heliocentric frame
    of reference.
     
    Ephemeris:
        
        0. Semi-major axis [AU]
        1. Eccentricity
        2. Inclination [deg]
        3. Longitude of ascending node [deg] 
        4. Londitude of pericenter [deg]
        5. Mean Longitude [deg]
        
        Plus accompanying rates in second file [same index]
        NB ANGULAR RATES ARE ARC SECONDS [1/3600 DEG]
    """

    
    Elems = np.genfromtxt("Planetary_Elements.csv", delimiter = ",")
    Elems_Rates = np.genfromtxt("Planetary_Elements_Rates.csv", delimiter = ",")
    
    #Get Julian Day number @ hour, and calculate time: 
    JD = Julian.Julian(year, month, day, hr, 0, 0)
    
    T0 = (JD - 2451545)/36525
    
    #Isolate elements and rates for planet:
    Elem = Elems[planet_id, :]
    Elem_R = Elems_Rates[planet_id, :]
    
    #Compute the elements value at the Julian Day number
    
    
    a = Elem[0] + Elem_R[0]*T0  
    e = Elem[1] + Elem_R[1]*T0
    i = Elem[2] + Elem_R[2]*T0 * 1/3600
    Om = Elem[3] + Elem_R[3]*T0 * 1/3600
    wdash = Elem[4] + Elem_R[4]*T0 * 1/3600
    L = Elem[5] + Elem_R[5]*T0 * 1/3600
        
    L = L % 360
    
    
    #Get Angular momentum:
    
    
    
    h = np.sqrt(a*mu*AU2KM*(1-e**2))
    
    #Get argument of perigee, mean anomaly, and call Kepler routine to find
    #true anomaly:
    
    
    w = wdash - Om
    if w < 0:
        w = 360 + w
    
    M = L - wdash
    if M < 0:
        M = 360 + M
    
    
    TA = Kepler.Solve(M, e)
    
    
    Elements = Transf.Orb2State(h, e, i, Om, w, TA, mu)
    
    #print("Julian Day: ", JD)
    #print("Orbital momentum: ", h)
    #print("Argument of Perigee: ", w)
    #print("Mean Anomaly: ", M)
    
    return Elements
    


main(3, 12, 27, 8, 2003)