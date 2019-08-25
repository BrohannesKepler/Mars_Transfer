#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 23:16:32 2019

@author: haider
"""
import numpy as np
from scipy.integrate import odeint, solve_ivp
import Ephemeris
import Transf
import Lambert_Solver
import Hyper
from TwoBody import f_sun
from TwoBodyIVP import f_earth, f_capt, events, events_mars
import Mars_Arrival
import matplotlib.pyplot as plt

def input_vars():
    
    global mu_s, mu_e, mu_m, rmars, r_SOI_e, Re
    

    mu_s = 1.327E11
    mu_e = 398600
    mu_m = 4.2828E4
    rmars = 3389.5
    Re = 6378
    r_SOI_e = 0.924E6
    
    print("""
          EARTH-MARS TRANSFER TRAJECTORY SOLVER
          
          ENTER DEPARTURE DATE FOR EARTH AND ARRIVAL DATE AT MARS, AS WELL AS
          ARRIVAL PARKING ORBIT ALTITUDE AT MARS.
          
          THIS PROGRAM WILL THEN CONSTRUCT THE HYPERBOLIC DEPARTURE AND ARRIVAL
          TRAJECTORIES AS WELL AS THE DIRECT TRANSFER TRAJECTORY SOLVED USING 
          LAMBERT'S PROBLEM.
          
          THE DELTA-V REQUIRED FOR THE MISSION WILL ALSO BE COMPUTED.



          """)

    print("Enter departure date: ")
    dep_date = input()
    print("Enter arrival date: ")
    arr_date = input()
    print("Enter the arrival parking orbit altitude in km: ")
    altorb = input()
    return dep_date, arr_date, altorb

def GetStates(dep_date, arr_date):
    
    "Compute the ephemeris at the departure and arrival dates, then use the \
     Lambert Solver to find the trajectory which satisfies both these positions\
     and the time of flight between them."
     
     #Get planetary states:
    
    P_States = np.zeros([4, 3])
    
    dep_d = int(dep_date[0:2])
    dep_m = int(dep_date[3:5])
    dep_y = int(dep_date[6:10])
    arr_d = int(arr_date[0:2])
    arr_m = int(arr_date[3:5])
    arr_y = int(arr_date[6:10])

    
    P_States[0,:] = Ephemeris.State(2, 12, dep_d, dep_m, dep_y)[0][0]
    P_States[1,:] = Ephemeris.State(3, 12, arr_d, arr_m, arr_y)[0][0]
    P_States[2,:] = Ephemeris.State(2, 12, dep_d, dep_m, dep_y)[0][1]
    P_States[3,:] = Ephemeris.State(3, 12, arr_d, arr_m, arr_y)[0][1]
    
    #Get planet 2's state at departure:
    P2_Init = np.zeros([2,3])
    
    P2_Init[0,:] = Ephemeris.State(3, 12, dep_d, dep_m, dep_y)[0][0]
    P2_Init[1,:] = Ephemeris.State(3, 12, dep_d, dep_m, dep_y)[0][1]
    
    #Calculate the time in seconds between departure and arrival, then call
    #the Lambert Solver to find the orbital state of the transfer trajectory:
    
    T_hours = Ephemeris.Julian(arr_y, arr_m, arr_d, 12, 0, 0) - Ephemeris.Julian(dep_y, dep_m, dep_d, 12, 0, 0)
    t = T_hours * 86400
    
    State, State2 = Lambert_Solver.lambert(P_States[0,:], P_States[1, :], t, mu_s)

    #Get initial cartesian state of transfer:
    
    Cart_State = Transf.Orb2State(State[0], State[2], State[4], State[3], State[5], State[6], mu_s)
    Cart_State_End = Transf.Orb2State(State2[0], State2[2], State2[4], State2[3], State2[5], State2[6], mu_s)
    
    #Compute hyperbolic excess velocities at departure and arrival:
    
    # v_inf_dep = V1 - V_Planet1
    # v_inf_arr = V2 - V_Planet2
    
    #Departure velocity vector:
    V_inf_dep = Cart_State[1] - P_States[2, :]
    V_inf_arr = Cart_State_End[1] - P_States[3, :]
    
    return P_States, P2_Init, t, Cart_State, V_inf_dep

def Propagate_Lambert(P_States, P2_Init, t, Cart_State):
    
    "Using the initial states of planet 1, the spacecraft, and the arrival\
     planet, integrate their respective trajectories assuming two-body\
     dynamics about the sun."
     
     #Get arrival planet position at departure:

    step = 5000
    T = np.linspace(0, t, step)
    
    #Configure the array of initial conditions for each body:
    TransInit = np.transpose(np.reshape(np.asarray(Cart_State), [6]))
    EarthInit = np.transpose(np.reshape(np.array([P_States[0,:], P_States[2,:]]), [6]))
    MarsInit = np.transpose(np.reshape(P2_Init, [6]))
    
    #Use odeint to propagate all three systems, store ina single 3D array
    
    STATES = np.zeros([step, 6, 3])
    inits = np.array([TransInit, EarthInit, MarsInit])
    
    for i in range(0, 3):
        
        STATES[:, :, i] = odeint(f_sun, inits[i],T)
        
    
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot(STATES[:,0,0], STATES[:,1,0], STATES[:,2,0], 'g', label = 'Transfer')
    ax.plot(STATES[:,0,1], STATES[:,1,1], STATES[:,2,1], 'b', label = 'Earth')
    ax.plot(STATES[:,0,2], STATES[:,1,2], STATES[:,2,2], 'r', label = 'Mars')
    ax.set_xlim3d(-3E8,3E8)
    ax.set_ylim3d(-3E8,3E8)
    ax.set_zlim3d(-1E8,1E8)
    ax.legend()
    plt.show()




def Compute_Departure(V_inf_dep):
    
    """ Compute the departure hyperbola as well as conditions at SOI exit for
        correct propagation."""
    #LEO calculations for a 300km parking orbit
    Re = 6378
    rp = Re + 300
    h_leo = (rp) * np.sqrt(mu_e/(rp))
    LEO_STATE = Transf.Orb2State(h_leo, 0.01, 0, 0, 0, 0, mu_e)
    R0 = LEO_STATE[0]
    V0 = LEO_STATE[1]
    vinf = np.linalg.norm(V_inf_dep) 
    
    #Hyperbola computations:
    
    e = 1 + rp*vinf*2/mu_e
    #print('Departure Eccentricity:      ', e)
    
    h_dep = mu_e * (e**2 - 1)**0.5/vinf
    #print('Departure specific momenum: ', h_dep)
    
    a_dep = (h_dep**2/mu_e) * 1/(e**2 - 1)
    
    R_P, VHHYP, DVV = Hyper.Hyp_Dep(V_inf_dep, 28.5, 6678, mu_e)

    Hyp_R = R_P
    Hyp_V = VHHYP
    unrp = R_P/np.linalg.norm(R_P)
    
    hyp_state = np.reshape(np.array([Hyp_R, Hyp_V]), 6)
    
    F = np.arccosh((1- (r_SOI_e/(-1*a_dep)))/e)
    M_hyp = e * np.sinh(F) - F
    t_SOI = (h_dep**3 * M_hyp) / (mu_e**2 * (e**2 - 1)**1.5)
    
    print("""
          """)
    print('===================== DEPARTURE CONDITIONS =======================')
    print('')
    print('Departure C3 [km^2/s^2]:         ', vinf**2)
    print('Departure Eccentricity:          ', e)
    print('Injection deltaV:                ', np.linalg.norm(Hyp_V) - np.linalg.norm(V0))    
    #=============================================================================
    #
    # SOLVE FOR TRAJECTORY FROM INITIAL CONDITIONS
    #
    
    
    tintend = t_SOI + 0.3*t_SOI
    tout = np.arange(0, tintend, 10)
    
    #outstate = odeint(f_earthodeint, hyp_state, np.linspace(0,tintend,1000))
    
    STATE = solve_ivp(f_earth, (0, tintend), hyp_state, method = 'RK45', t_eval = tout,  dense_output='True', events = (events))
    
    Res = STATE.y
    
    #Get position, velocity, and time at the sphere of influence
    
    soi_state = STATE.sol(STATE.t_events[0][0])
    soi_time = STATE.t_events[0][0]

    soi_pos = np.asarray(soi_state[0:3])
    soi_vel = np.asarray(soi_state[3:6])
    
    soi_ecl = Hyper.Plan2Eclip(soi_pos, soi_vel)
    soi_pos_ecl_e = soi_ecl[0]
    soi_vel_ecl_e = soi_ecl[1]
    
    x = Res[0, :]
    y = Res[1, :]
    z = Res[2, :]
    
    
    
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    Re = 6378
    phi = np.linspace(0, 2*np.pi, 20)
    zeta = np.linspace(0, np.pi, 20)
    xe = Re * np.outer(np.cos(phi), np.sin(zeta))
    ye = Re * np.outer(np.sin(phi), np.sin(zeta))
    ze = Re * np.outer(np.ones(np.size(phi)), np.cos(zeta))
    ax.plot(x, y, z, 'b', label = 'Departure')
   # ax.plot(xve*5*Re,yve*5*Re,zve*5*Re,'g', label = 'v_e Direction')
    #ax.plot(xeci*5*Re, [0,0],[0,0], 'k', label = 'ECI Frame')
    #ax.plot([0,0], yeci*5*Re,[0,0], 'k')
    #ax.plot([0,0],[0,0], zeci*5*Re, 'k')
    #ax.plot([0, neq[0]*6*Re], [0, neq[1]*6*Re], [0, neq[2]*6*Re],'y')
    #ax.plot([0, omvec[0]*6*Re], [0, omvec[1]*6*Re], [0, omvec[2]*6*Re],'y')
    #ax.plot([0, vvec[0]*6*Re], [0, vvec[1]*6*Re], [0, vvec[2]*6*Re],'y')
    #ax.plot([0, unit_Vinfeq[0]*10*Re], [0, unit_Vinfeq[1]*10*Re], [0, unit_Vinfeq[2]*10*Re],'r',label = 'Vinf direction ECI')
    ax.plot([0, unrp[0]*6*Re], [0, unrp[1]*6*Re], [0, unrp[2]*6*Re],'y',label = 'ir')
    
    #ax.plot([0, iinf[0]*7*Re], [0, iinf[1]*7*Re], [0, iinf[2]*7*Re],'p',label = 'i_inf')
    ax.set_xlim3d(-40000,40000)
    ax.set_ylim3d(-40000,40000)
    ax.plot_surface(xe,ye,ze)
    ax.set_zlim3d(-4.5E4, 4.5E4)
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.legend()
    
   
    return soi_pos_ecl_e, soi_vel_ecl_e, soi_time

def Compute_Arrival(dep_d, dep_m, dep_y, t, soi_pos_ecl_e, soi_vel_ecl_e, soi_time, altorb):
    
    """ Propagate from the SOI exit to the SOI arrival at Mars to obtain the 
        actual entry Vinfinity to use for arrival hyperbola determination."""
        
    jd_update = np.round(soi_time/86400)

    Earth_update_pos = Ephemeris.State(2, 12, dep_d+jd_update, dep_m, dep_y)[0][0]
    Earth_update_v = Ephemeris.State(2, 12, dep_d+jd_update, dep_m, dep_y)[0][1]   
    
    M_update_pos = Ephemeris.State(3, 12, dep_d+jd_update, dep_m, dep_y)[0][0]
    M_update_v = Ephemeris.State(3, 12, dep_d+jd_update, dep_m, dep_y)[0][1]
    
    E_init = np.reshape(np.array([Earth_update_pos, Earth_update_v]),6)
    M_init = np.reshape(np.array([M_update_pos, M_update_v]),6)
    
    inter_state = np.reshape(np.array([soi_pos_ecl_e+Earth_update_pos, soi_vel_ecl_e+Earth_update_v]), 6)
    
    inter_t = t - soi_time
    
    inits = np.array([M_init, inter_state])
    int_state = np.reshape(inits, 12)

    tout = np.arange(0, inter_t, 1000)
    
    STATE3 = solve_ivp(f_capt, (0, inter_t), int_state, method = 'RK23', t_eval = tout,  dense_output='True', events = events_mars)

    #Planet and SC states at SOI entrance:
    soi_state_mars = STATE3.sol(STATE3.t_events[0][0])
    
    soi_pos_mars = soi_state_mars[0:3]
    soi_vel_mars = soi_state_mars[3:6]
    soi_pos_scmars = soi_state_mars[6:9]
    soi_vel_scmars = soi_state_mars[9:12]
    
    soi_pos_arr = soi_pos_scmars - soi_pos_mars
    soi_vel_arr =  soi_vel_scmars - soi_vel_mars
    
    
    "Now find hyperbolic arrival characteristics"
    altorb = int(altorb)
    r_orb = rmars + altorb

    Pos_Arrival, R_BI, Pos_Arrival_Leo,  V_BI = Mars_Arrival.Arrival(soi_pos_arr, soi_vel_arr, r_orb)
    
    X_MH = Pos_Arrival[0, :]
    Y_MH = Pos_Arrival[1, :]
    Z_MH = Pos_Arrival[2, :]
    
    X_ME = Pos_Arrival_Leo[0, :]
    Y_ME = Pos_Arrival_Leo[1, :]
    Z_ME = Pos_Arrival_Leo[2, :]
    
    
    unit_R_BI = R_BI/np.linalg.norm(R_BI)
    unit_V_BI = V_BI/np.linalg.norm(V_BI)   
    
    
    
    phi = np.linspace(0, 2*np.pi, 20)
    zeta = np.linspace(0, np.pi, 20)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    xm_s = rmars * np.outer(np.cos(phi), np.sin(zeta))
    ym_s = rmars * np.outer(np.sin(phi), np.sin(zeta))
    zm_s = rmars * np.outer(np.ones(np.size(phi)), np.cos(zeta))
    #ax.plot(Xm1[0:100],Ym1[0:100],Zm1[0:100],'r', label = 'Mars')
    ax.plot(X_MH,Y_MH,Z_MH,'g', label = 'Arrival Hyperbola')
    ax.plot(X_ME,Y_ME,Z_ME,'b', label = 'Circular Orbit')
    ax.plot([0, unit_V_BI[0]*6*rmars], [0, unit_V_BI[1]*6*rmars], [0, unit_V_BI[2]*6*rmars],'k', label = 'V_Inf Direction')
    #ax.plot([0, unit_R_BI[0]*6*rmars], [0, unit_R_BI[1]*6*rmars], [0, unit_R_BI[2]*6*rmars])
    #ax.plot([0, unitvmars[0]*6*rmars], [0, unitvmars[1]*6*rmars], [0, unitvmars[2]*6*rmars],'r')
    #ax.plot([0, unitvmarseclip[0]*6*rmars], [0, unitvmarseclip[1]*6*rmars], [0, unitvmarseclip[2]*6*rmars],'y')
    ax.plot_surface(xm_s,ym_s,zm_s,color='r')
    ax.set_xlim3d(-30000,30000)
    ax.set_ylim3d(-30000,30000)
    ax.set_zlim3d(-3.5E4, 3.5E4)
    plt.legend()
    
    
    return soi_pos_arr, soi_vel_arr

def main():
    
    dep_date, arr_date, altorb = input_vars()
    P_States, P2_Init, t, Cart_State, V_inf_dep = GetStates(dep_date, arr_date)

    dep_d = int(dep_date[0:2])
    dep_m = int(dep_date[3:5])
    dep_y = int(dep_date[6:10])
    
    #altorb = 500
    Propagate_Lambert(P_States, P2_Init, t, Cart_State)
    soi_pos_ecl_e, soi_vel_ecl_e, soi_time = Compute_Departure(V_inf_dep)
    Compute_Arrival(dep_d, dep_m, dep_y, t, soi_pos_ecl_e, soi_vel_ecl_e, soi_time, altorb)
    
    
if __name__ == "__main__":
    #Run main function to execute script
    main()