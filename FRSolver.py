import numpy as np
import pandas as pd
from functions import *
from schemes import *
from plot import * 
from Kepler import *
from adaptive import *

def findh(rs, vs, h0 = np.inf):
    h = h0
    N = len(rs)
    for i in range(N):
        for j in range(i+1, N):
                r_mag = np.linalg.norm(rs[i] - rs[j])
                v_mag = np.linalg.norm(vs[i] - vs[j])
                h = min(h, r_mag / v_mag)
    return h

def maxDist(rs):
    d = -1
    N = len(rs)
    for i in range(N): 
        for j in range(i+1, N):
            d = max(d, np.linalg.norm(rs[i]-rs[j]))
    return d

def findR(v0s, E0, masses):
    ke = np.sum(KineticEnergy(v0s, masses))
    R = 5 / (2 * (ke - E0))
    return R

def distCalculator(rs):
    N = len(rs)
    Rs = []
    for i in range(N):
        for j in range(i + 1, N):
            Rs.append(np.linalg.norm(rs[i] - rs[j]))
    Rs = np.array(Rs)
    return Rs[:-1] / np.sum(Rs)

def fillGrid(path, gridsize = 1000):
    grid = np.zeros((gridsize, gridsize))

    for pos in path:

        grid_x = int(2 * (pos[0] - 1e-15) * gridsize)
        grid_y = int(2 * (pos[1] - 1e-15) * gridsize)

        grid[grid_x][grid_y] = 1

    return grid

def fullSolver(T, C, r0s, v0s, G, masses, hlim = 1e-3, Elim = 0.01, h0 = 1e-10, t0 = 0):
    
    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    h = h0
    
    E0 = TotalEnergy(r0s, v0s, G, masses)
    E0hat = E0 
    ## conditions to avoid dividing by zero 
    if E0hat == 0: E0hat = np.max(np.abs(TotalKE(v0s, masses) + np.abs(r0s, G, masses))) 
    if E0hat == 0: E0hat = 1

    t_vals = [t0]
    rs_traj = [r0s] 
    vs_traj = [v0s] 
    E_traj = [0]
    am_traj = [AngMomentum(rs, vs, masses)]
    times = 0 

    rs, vs = Optimised_FR_Step(rs, vs, h, G, masses) 
    
    # run scheme for requried number of steps 
    while t <= T:
        t1 = time.time()

        h = C * findh(rs, vs) # proposed timestep

        rs_bar, vs_bar = Optimised_FR_Step(rs, vs, h, G, masses)
        h_bar = C * findh(rs_bar, vs_bar) # proposed timestep

        h_new = (h + h_bar) / 2
        
        rs, vs = Optimised_FR_Step(rs, vs, h_new, G, masses)

        times += time.time() - t1

        E = TotalEnergy(rs, vs, G, masses) # Calculate Energy 
        relE = np.abs((E - E0) / E0hat)

        if h_new < hlim: 
            stability = 2
            break 
        if t+h_new > T: 
            stability =  1
            break 
        if relE > Elim:
            stability = 3
            break 
        if maxDist(rs) > 10:
            stability = 0
            break 
        
        t += h_new

        ## append values to trajectories 
        t_vals = t_vals + [t]
        rs_traj = rs_traj + [rs] 
        vs_traj = vs_traj + [vs]
        E_traj.append(relE)
        am_traj.append(AngMomentum(rs, vs, masses))

    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    vs_traj = np.array(vs_traj) 
    
    return (t_vals, rs_traj, vs_traj, E_traj, am_traj, times), stability 


def optimisedSolver(T, C, r0s, v0s, G, masses, hlim = 1e-3, Elim = 0.01, h0 = 1e-10, t0 = 0):
    
    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    h = h0
    
    E0 = TotalEnergy(r0s, v0s, G, masses)
    maxE = 0

    rs_traj = [r0s] 

    E0hat = E0 
    ## conditions to avoid dividing by zero 
    if E0hat == 0: E0hat = np.max(np.abs(TotalKE(v0s, masses) + np.abs(r0s, G, masses))) 
    if E0hat == 0: E0hat = 1

    rs, vs = Optimised_FR_Step(rs, vs, h, G, masses) 
    
    # run scheme for requried number of steps 
    while t <= T:
        h = C * findh(rs, vs) # proposed timestep

        rs_bar, vs_bar = Optimised_FR_Step(rs, vs, h, G, masses)
        h_bar = C * findh(rs_bar, vs_bar) # proposed timestep

        h_new = (h + h_bar) / 2
        
        rs, vs = Optimised_FR_Step(rs, vs, h_new, G, masses)

        E = TotalEnergy(rs, vs, G, masses) # Calculate Energy 
        relE = np.abs(np.sum(E) - E0) / np.abs(E0hat)

        if h_new < hlim: 
            stability = 2
            break 
        if t+h_new > T: 
            stability =  1
            break 
        if relE > Elim:
            stability = 3
            break 
        if maxDist(rs) > 10:
            stability = 0
            break 
        
        t += h_new

        rs_traj = rs_traj + [rs] 
        maxE = max(relE, maxE)

    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    
    return stability,  maxE, t

def shapeSolver(T, C, r0s, v0s, G, masses, defaultVar = 10000, hlim = 1e-3, Elim = 0.01, h0 = 1e-10, t0 = 0):
    
    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    h = h0
    variance = defaultVar
    
    E0 = TotalEnergy(r0s, v0s, G, masses)
    maxE = 0

    rs_traj = [r0s] 

    E0hat = E0 
    ## conditions to avoid dividing by zero 
    if E0hat == 0: E0hat = np.max(np.abs(TotalKE(v0s, masses) + np.abs(r0s, G, masses))) 
    if E0hat == 0: E0hat = 1

    rs, vs = Optimised_FR_Step(rs, vs, h, G, masses) 
    
    # run scheme for requried number of steps 
    while t <= T:
        h = C * findh(rs, vs) # proposed timestep

        rs_bar, vs_bar = Optimised_FR_Step(rs, vs, h, G, masses)
        h_bar = C * findh(rs_bar, vs_bar) # proposed timestep

        h_new = (h + h_bar) / 2
        
        rs, vs = Optimised_FR_Step(rs, vs, h_new, G, masses)

        E = TotalEnergy(rs, vs, G, masses) # Calculate Energy 
        relE = np.abs(np.sum(E) - E0) / np.abs(E0hat)

        if h_new < hlim: 
            stability = 2
            break 
        if t+h_new > T: 
            stability =  1
            path = [distCalculator(rs) for rs in rs_traj]
            grid = fillGrid(path, gridsize = 1000)
            variance = np.sum(grid)
            break 
        if relE > Elim:
            stability = 3
            break 
        if maxDist(rs) > 10:
            stability = 0
            break 
        
        t += h_new

        rs_traj = rs_traj + [rs] 
        maxE = max(relE, maxE)

    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    
    return stability, variance, maxE, t



def NStepsSolver(NSteps, C, r0s, v0s, G, masses, hlim = 1e-6, Elim = 0.01, h0 = 1e-10, t0 = 0):
    
    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    h = h0
    
    E0 = TotalEnergy(r0s, v0s, G, masses)
    E0hat = E0 
    ## conditions to avoid dividing by zero 
    if E0hat == 0: E0hat = np.max(np.abs(TotalKE(v0s, masses) + np.abs(r0s, G, masses))) 
    if E0hat == 0: E0hat = 1

    t_vals = [t0]
    rs_traj = [r0s] 
    vs_traj = [v0s] 
    E_traj = [0]
    am_traj = [AngMomentum(rs, vs, masses)]
    times = 0 

    totalSteps = 0

    rs, vs = Optimised_FR_Step(rs, vs, h, G, masses) 
    
    # run scheme for requried number of steps 
    while 1 < 2:
        t1 = time.time()

        h = C * findh(rs, vs) # proposed timestep

        rs_bar, vs_bar = Optimised_FR_Step(rs, vs, h, G, masses)
        h_bar = C * findh(rs_bar, vs_bar) # proposed timestep

        h_new = (h + h_bar) / 2
        
        rs, vs = Optimised_FR_Step(rs, vs, h_new, G, masses)

        times += time.time() - t1

        E = TotalEnergy(rs, vs, G, masses) # Calculate Energy 
        relE = np.abs((E - E0) / E0hat)

        # if h_new < hlim: 
        #     stability = 2
        #     break 
        if totalSteps > NSteps: 
            stability =  1
            break 
        # if relE > Elim:
        #     stability = 3
        #     break 
        # if maxDist(rs) > 10:
        #     stability = 0
        #     break 
        
        t += h_new

        ## append values to trajectories 
        t_vals = t_vals + [t]
        rs_traj = rs_traj + [rs] 
        vs_traj = vs_traj + [vs]
        E_traj.append(relE)
        am_traj.append(AngMomentum(rs, vs, masses))

        totalSteps += 1

    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    vs_traj = np.array(vs_traj) 
    
    return (t_vals, rs_traj, vs_traj, E_traj, am_traj, times), stability 

