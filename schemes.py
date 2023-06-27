import numpy as np
import scipy as sci
import scipy.integrate
from functions import *

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

def Leapfrog4Step(r0s, v0s, h, G, masses):
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
    vs_int = vs_half + 0.5 * dv_dt(r1s, G, masses) * h
    rs_int = r0s + vs_int * h
    v1s = vs_half + 0.5 * dv_dt(rs_int, G, masses) * h
    return r1s, v1s

# ============================
#         Integrator
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
            - ke_traj: trajectory of kinetic energy of each particle 
            - pe_traj: trajectory of potential energy of each particle 
            - am_traj: trajectory of angular momentum of each particle 
    '''
    
    # Set the number of steps; it is best if h is an integral fraction of T
    Nsteps = int(T / h)
    
    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    
    # Initialize our saved trajectories to be blank 
    t_vals = []
    rs_traj = [] 
    vs_traj = [] 
    ke_traj = []
    pe_traj = []
    am_traj = []
    
    # run scheme for requried number of steps 
    for n in range(Nsteps):
        rs,vs = scheme(rs, vs, h, G, masses)  # Update step
        ke = KineticEnergy(vs, masses) # Calculate kinetic energy
        pe = PotentialEnergy(rs, G, masses) # Calculate potential energy
        am = AngMomentum(rs, vs, masses) # Calculate angular momentum 
        
        ## append values to trajectories 
        t = t + h
        t_vals = t_vals + [t]
        rs_traj = rs_traj + [rs] 
        vs_traj = vs_traj + [vs]
        ke_traj = ke_traj + [ke]
        pe_traj = pe_traj + [pe]
        am_traj = am_traj + [am]
        
    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    vs_traj = np.array(vs_traj) 
    ke_traj = np.array(ke_traj)
    pe_traj = np.array(pe_traj)
    am_traj = np.array(am_traj)
    
    return t_vals, rs_traj, vs_traj, ke_traj, pe_traj, am_traj

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
    
    # format initial conditions for use in scipy integrator
    w0 = np.concatenate((r0s, v0s)).flatten()

    t_vals = np.linspace(t0, T-t0, int(T/h)) # t-vals to integrate over
    true_sols = sci.integrate.odeint(all_derivatives, w0, t_vals, args=(G,masses)) # integrate scheme using scipy 

    # Initialize our saved trajectories to be blank 
    rs_traj = [] 
    vs_traj = [] 
    ke_traj = []
    pe_traj = []
    am_traj = []
    
    # loop over all t-vals 
    for i, true_sol in enumerate(true_sols):
        true_sol = np.reshape(true_sol, (len(true_sol) // 3, 3)) # reshape solution to 3D
        true_r, true_v = np.split(true_sol, 2) # get true position and velocities 
        rs_traj.append(true_r) # position trajectory 
        vs_traj.append(true_v) # veclocity trajectory 
        ke_traj.append(KineticEnergy(true_v, masses)) # kinetic energy trajectory 
        pe_traj.append(PotentialEnergy(true_r, G, masses)) # potential energy trajectory 
        am_traj.append(AngMomentum(true_r, true_v, masses)) # angular momentum trajectory 

    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    vs_traj = np.array(vs_traj) 
    ke_traj = np.array(ke_traj)
    pe_traj = np.array(pe_traj)
    am_traj = np.array(am_traj)
    
    return (t_vals, rs_traj, vs_traj, ke_traj, pe_traj, am_traj)
