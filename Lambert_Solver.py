# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 13:22:16 2018

@author: Haider
"""
import numpy as n
import Stumpff
import Transf
from mpl_toolkits import mplot3d



"Lambert solver given two position vectors and a time of flight between them.\
uses Newton's method to iteratively solve for a solution."

"ASSUMES A PROGRADE ORBITAL TRAJECTORY"


def lambert(r1, r2, dT, mu):

    global absr1, absr2, A
    nmax = 1000
    count = 0
    ratio = 1
    z = 0

    ratio = 1

    absr1 = n.linalg.norm(r1)
    absr2 = n.linalg.norm(r2)

    if n.cross(r1, r2)[2] >= 0:

        dtheta = n.arccos((n.dot(r1, r2))/(absr1*absr2))

    elif n.cross(r1, r2)[2] < 0:

        dtheta = 2*n.pi - n.arccos(n.dot(r1, r2)/(absr1*absr2))

    A = n.sin(dtheta) * n.sqrt((absr1*absr2)/(1-n.cos(dtheta)))

#Solve for the z value using Newton's method:

    while abs(ratio) > 1e-8 and count <= nmax:

        ratio = Fz(z, dT,mu)/Fzdash(z)

        z = z - ratio

        count = count+1

    #Update Stumpff functions given the computed value of z:
    S = Stumpff.S(z)
    C = Stumpff.C(z)

    y = absr1 + absr2 + A * ((z*S - 1)/n.sqrt(C))

    #Determine the lagrange coefficients:

    f = 1 - y/absr1
    g = A * n.sqrt(y/mu)

    fdot = n.sqrt(mu/(absr1*absr2)**2) * n.sqrt(y/C) * (z*S - 1)
    gdot = 1 - y/absr2
    #Now, calculate the new velocity vectors at positions r1 and r2

    v1 = 1/g * (r2 - f*r1)
    v2 = 1/g * (gdot*r2 - r1)

    #Finally, given the position and velocity at one of these points, orbital
    #elements can be calculated:
    State = Transf.State2Orb(r1, v1, mu)
    State2 = Transf.State2Orb(r2, v2, mu)


    print("Lambert's Problem, computed elements: \n")
    print("Momentum:            ", State[0], " km3/s2")
    print("Semi-major axis:     ", State[1], " km")
    print("Eccentricity:        ", State[2])
    print("RAAN:                ", State[3], " deg")
    print("Inclination:         ", State[4], " deg")
    print("Argument of Perigee: ", State[5], " deg")
    print("True Anomaly:        ", State[6], " deg")
    return State, State2


def Fz(z,dT,mu):
    S = Stumpff.S(z)
    C = Stumpff.C(z)
    y = absr1 + absr2 + A * ((z*S-1)/n.sqrt(C))
    F = (y/C)**1.5 * S + A*n.sqrt(y) - n.sqrt(mu)*dT

    return F


def Fzdash(z):
    S = Stumpff.S(z)
    C = Stumpff.C(z)

    y = absr1 + absr2 + A * ((z*S-1)/n.sqrt(C))

    if z == 0:

        F = n.sqrt(2)/40 * y**1.5 + A/8 * (n.sqrt(y) + A*n.sqrt(1/(2*y)))

    else:
        F = (y/C)**1.5 * (1/(2*z) * (C - (1.5*S/C)) + 0.75 * S**2/C) +\
                A/8 * (3*(S/C) * n.sqrt(y) + A*n.sqrt(C/z))

    return F
