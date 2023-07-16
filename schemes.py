import numpy as np
import scipy as sci
import scipy.integrate
from functions import *
import time

# ============================
#           Schemes
# ============================

def EulerStep(r0s, v0s, h, G, masses):
    '''
    One step of forward Euler method
    
    input: - r0s: starting positions of all particles
           - v0s: starting velocity of all particles
           - h: size of timestep
           
    output: - r1s: new position after one step 
            - v1s: new velocities after one step 
    '''
    r1s = r0s + h * dr_dt(v0s) 
    v1s = v0s + h * dv_dt(r0s, G, masses)
    
    return r1s, v1s

def EulerCromerStep(r0s, v0s, h, G, masses):
    '''
    One step of the symplectic Euler-Cromer method
    
    input: - r0s: starting positions of all particles
           - v0s: starting velocity of all particles
           - h: size of timestep
           
    output: - r1s: new position after one step 
            - v1s: new velocities after one step 
    '''
    r1s = r0s + h * dr_dt(v0s)
    v1s = v0s + h * dv_dt(r1s, G, masses)
    return r1s, v1s

def LeapfrogStep(r0s, v0s, h, G, masses):
    '''
    One step of the standard leapfrog method
    
    input: - r0s: starting positions of all particles
           - v0s: starting velocity of all particles
           - h: size of timestep
           
    output: - r1s: new position after one step 
            - v1s: new velocities after one step 
    '''
        
    vs_half = v0s + 0.5 * dv_dt(r0s, G, masses) * h 
    r1s = r0s + vs_half * h 
    v1s = vs_half + 0.5 * dv_dt(r1s, G, masses) * h
    return r1s, v1s

def ForestRuthStep(r0s, v0s, h, G, masses):
    theta = 1/ (2 - (2 ** (1/3)))
    
    r1s = r0s + theta * h * 0.5 * dr_dt(v0s)
    v1s = v0s + theta * h * dv_dt(r1s, G, masses)
    
    r2s = r1s + (1-theta)*h*0.5*dr_dt(v1s)
    v2s = v1s + (1-2*theta)*h*dv_dt(r2s, G, masses)
    
    r3s = r2s + (1-theta)*h*0.5*dr_dt(v2s)
    v3s = v2s + theta * h * dv_dt(r3s, G, masses)
    
    r4s = r0s + theta * h * 0.5 * dr_dt(v3s)
    v4s = v3s
    
    return r4s, v4s

def PEFRLStep(r0s, v0s, h, G, masses):
    p = + 0.1786178958448091E+00
    l = - 0.2123418310626054E+00
    c = - 0.6626458266981849E-01
    
    r1s = r0s + p * h * v0s
    v1s = v0s + (1 - 2 * l) * 0.5 * h * dv_dt(r1s, G, masses)
    r2s = r1s + c * h * v1s
    v2s = v1s + l * h * dv_dt(r2s, G, masses)
    r3s = r2s + (1 - 2*(c + p)) * h * v2s
    v3s = v2s + l * h * dv_dt(r3s, G, masses)
    r4s = r3s + c * h * v3s
    v4s = v3s + (1 - 2 * l) * 0.5 * h * dv_dt(r4s, G, masses)
    r5s = r4s + p * h * v4s
    v5s = v4s
    
    return r5s, v5s

def Optimised_FR_Step(r0s, v0s, h, G, masses):
    p = + 0.1786178958448091E+00
    l = - 0.2123418310626054E+00
    c = - 0.6626458266981849E-01

    a = (1 - 2 * l)
    
    r1s = r0s + p * h * v0s
    v1s = v0s + a * 0.5 * h * dv_dt(r1s, G, masses)
    r2s = r1s + c * h * v1s
    v2s = v1s + l * h * dv_dt(r2s, G, masses)
    r3s = r2s + (1 - 2*(c + p)) * h * v2s
    v3s = v2s + l * h * dv_dt(r3s, G, masses)
    r4s = r3s + c * h * v3s
    v4s = v3s + a * 0.5 * h * dv_dt(r4s, G, masses)
    r5s = r4s + p * h * v4s
    
    return r5s, v4s
    

def RK4Step(r0s, v0s, h, G, masses):
    '''
    One step of the standard fourth order Runge-Kutta method
    
    input: - r0s: starting positions of all particles
           - v0s: starting velocity of all particles
           - h: size of timestep
           
    output: - r1s: new position after one step 
            - v1s: new velocities after one step 
    '''
        
    # transform to one long vector 
    w0 = vec_to_w(r0s, v0s)

    k1 = h * all_derivatives(w0, 0, G, masses)
    k2 = h * all_derivatives(w0 + k1/2, 0, G, masses)
    k3 = h * all_derivatives(w0 + k2/2, 0, G, masses)
    k4 = h * all_derivatives(w0 + k3, 0, G, masses)
    
    w1 = w0 + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    r1s, v1s = w_to_vec(w1)
    
    return r1s, v1s
    
# ============================
#         Integrators
# ============================

def run_scheme(scheme, t0, T, h, r0s, v0s, G, masses):
    '''
    Evolution of the n-body problem using a numerical scheme.
    
    input: - scheme: numerical scheme to use
           - t0:     starting time
           - T:      time period 
           - h:      timestep
           - r0s:    starting position of each particle 
           - v0s:    starting velocity of each particle 
           - G:      gravitational constant
           - masses: mass of each particle      
           
    output: - t_vals:  list of time values
            - rs_traj: trajectory of positions of each particle 
            - vs_traj: trajectory of velocity of each particle 
            - E_traj: trajectory of energy of each particle 
            - am_traj: trajectory of angular momentum of each particle 
    '''
    
    # reposition centre of mass to origin with no momentum 
    rcom, vcom = CentreOfMass(r0s, v0s, masses)
    r0s -= rcom
    v0s -= vcom
    
    # Set the number of steps; it is best if h is an integral fraction of T
    Nsteps = int(T / h)
    
    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    
    # Initialize our saved trajectories to be blank 
    t_vals = [t0]
    rs_traj = [r0s] 
    vs_traj = [v0s] 
    E_traj = [TotalEnergy(r0s, v0s, G, masses)]
    am_traj = [AngMomentum(r0s, v0s, masses)]
    
    times = time.time() - time.time()
    
    # run scheme for requried number of steps 
    for _ in range(Nsteps):
        t1 = time.time()
        rs,vs = scheme(rs, vs, h, G, masses)  # Update step
        times += time.time() - t1
        E = TotalEnergy(rs, vs, G, masses)
        am = AngMomentum(rs, vs, masses) # Calculate angular momentum 
        
        ## append values to trajectories 
        t = t + h
        t_vals = t_vals + [t]
        rs_traj = rs_traj + [rs] 
        vs_traj = vs_traj + [vs]
        E_traj = E_traj + [E]
        am_traj = am_traj + [am]
        
    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    vs_traj = np.array(vs_traj) 
    E_traj = np.array(E_traj)
    am_traj = np.array(am_traj)
    
    # reposition centre of mass to origin with no momentum 
    rs_traj = np.array([rs + rcom for rs in rs_traj])
    vs_traj = np.array([vs + vcom for vs in vs_traj])
    
    return (t_vals, rs_traj, vs_traj, E_traj, am_traj, times)


def run_scipy(t0, T, h, r0s, v0s, G, masses):
    '''
    Integrate trajectories from initial conditions using scipy.
    
    input: - t0:     starting time
           - T:      time period 
           - h:      timestep
           - r0s:    starting position of each particle 
           - v0s:    starting velocity of each particle 
           - G:      gravitational constant
           - masses: mass of each particle     
           
    output: - t_vals:  list of time values
            - rs_traj: trajectory of positions of each particle 
            - vs_traj: trajectory of velocity of each particle 
            - ke_traj: trajectory of kinetic energy of each particle 
            - pe_traj: trajectory of potential energy of each particle 
            - am_traj: trajectory of angular momentum of each particle 
    '''
    
    # reposition centre of mass to origin with no momentum 
    rcom, vcom = CentreOfMass(r0s, v0s, masses)
    r0s -= rcom
    v0s -= vcom
    
    # format initial conditions for use in scipy integrator
    w0 = np.concatenate((r0s, v0s)).flatten()

    t_vals = np.linspace(t0, T-t0, int(T/h)) # t-vals to integrate over
    
    times = time.time()
    true_sols = sci.integrate.odeint(all_derivatives, w0, t_vals, args=(G,masses)) # integrate scheme using scipy 
    times = time.time() - times
    
    # Initialize our saved trajectories to be blank 
    rs_traj = [] 
    vs_traj = [] 
    E_traj = []
    am_traj = []
   
    
    # loop over all t-vals 
    for i, true_sol in enumerate(true_sols):
        true_sol = np.reshape(true_sol, (len(true_sol) // 3, 3)) # reshape solution to 3D
        true_r, true_v = np.split(true_sol, 2) # get true position and velocities 
        rs_traj.append(true_r) # position trajectory 
        vs_traj.append(true_v) # veclocity trajectory 
        E_traj.append(TotalEnergy(true_r, true_v, G, masses))
        am_traj.append(AngMomentum(true_r, true_v, masses)) # angular momentum trajectory 

    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    vs_traj = np.array(vs_traj) 
    E_traj = np.array(E_traj)
    am_traj = np.array(am_traj)
        
    # reposition centre of mass to origin with no momentum 
    rs_traj = np.array([rs + rcom for rs in rs_traj])
    vs_traj = np.array([vs + vcom for vs in vs_traj])
        
    return (t_vals, rs_traj, vs_traj, E_traj, am_traj, times)

