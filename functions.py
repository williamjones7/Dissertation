import numpy as np

# ============================
#      Governing Equations
# ============================

def Force_i(rs, i, G, masses):
    '''
    Total force acting on particle i, due to mass of other particles.
    
    input: - rs:     position of each particle
           - i:      particle to find force on
           - G:      gravitational constant
           - masses: mass of each particle      
           
    output: - F_i: total force acting on particle i, as a 3D vector
    '''
    
    ri = rs[i] # get position of i-th mass 
    F_i = np.zeros(ri.shape) # create empty force vector for i-th mass 
    
    # loop over other masses
    for j, rj in enumerate(rs):
        
        if i != j:
            assert np.linalg.norm(rj - ri), 'Collision' # masses are at the same position 
            Fij = G * masses[i] * masses[j] * (rj - ri) / ((np.linalg.norm(rj - ri))** 3) # force of j-th mass on i-th mass
            F_i += Fij # sum forces
            
    return F_i

def Force(rs, G, masses):
    '''
    Finds force acting on each particle
    
    input: - rs:     position of each particle
           - G:      gravitational constant
           - masses: mass of each particle      
           
    output: - Fs: 3D forces acting on each particle
    '''
    
    N = len(rs)

    Fs = np.zeros_like(rs).astype('float64') # empty vector of forces 

    for i in range(N):
        for j in range(i+1, N):
            rij = rs[j] - rs[i]
            rij_mag = np.linalg.norm(rij)
            F = masses[i] * masses[j] * (rij) / (rij_mag ** 3)
            Fs[i] += F
            Fs[j] += - F
        
    return G * Fs

def dr_dt(vs):
    '''Returns the r-derivative of all particles'''
    return vs

def dv_dt(rs, G, masses):
    '''Returns the second r-derivative (acceleration) of all particles, from F = ma'''
    N = len(rs)

    Fs = np.zeros_like(rs).astype('float64') # empty vector of forces 

    for i in range(N):
        for j in range(i+1, N):
            rij = rs[j] - rs[i]
            F = (rij) / (np.linalg.norm(rij) ** 3)
            Fs[i] += F * masses[j]
            Fs[j] += - F * masses[i]
        
    return G * Fs

def w_to_vec(w):
    ''' Transforms flat 1D vector to position and velocity vectors'''
    W = np.reshape(w, (len(w) // 3, 3))
    rs, vs = np.split(W, 2) 
    return rs, vs

def vec_to_w(rs, vs):
    ''' Transforms positions and velocities to a flat 1D vector'''
    return np.concatenate((rs.flatten(), vs.flatten()))
    
def all_derivatives(w, t, G, masses):
    '''
    Finds derivative, in every cardinal direction, of position and velocity of each particle
    
    input: - w:      flattened list of positions and velocities of each mass
           - t:      time
           - G:      gravitational constant
           - masses: mass of each particle      
           
    output: flattened gradient of position and velocity
    '''
 
    rs, vs = w_to_vec(w) # separate positions and velocities
    
    drdt = dr_dt(vs) # find velocity of all particles
    dvdt = dv_dt(rs, G, masses) # find acceleration of all particles
    
    derivs = vec_to_w(drdt, dvdt)
    # derivs = np.concatenate((drdt,dvdt)).flatten() # reformat to be used by scipy integrate
    
    return derivs

# ============================
#         Reposition 
# ============================

def CentreOfMass(rs, vs, masses):
    '''
    Position and centre of mass of system 
    
    input: - rs:     position of each particle
           - vs:     velocity of each particle 
           - masses: mass of each particle      
           
    output: - rcom: position of centre of mass
            - vcom: velocity of centre of mass
    '''
    
    rcom = sum([rs[i] * masses[i] for i in range(len(masses))]) / np.sum(masses)
    vcom = sum([vs[i] * masses[i] for i in range(len(masses))]) / np.sum(masses)
    return rcom, vcom

def Centralise(rs_traj, i):
    '''
    Position particle i at the centre of the system 
    
    input: - rs: position of each particle
           - i:  particle to make centre  
           
    output: - rs: new position of each particle
    '''
    
    ri = np.copy(rs_traj[:,i,:])
    
    for j in range(rs_traj.shape[1]):
        rs_traj[:,j,:] -= ri
        
    return rs_traj

# ============================
#          Energy 
# ============================

def KE(vs, i, masses):
    '''
    Total kinetic energy of particle i. E_k = 1/2 * m * v^2
    
    input: - vs:     velocity of each particle
           - i:      particle to find force on
           - masses: mass of each particle      
           
    output: - ke: kinetic energy
    '''
        
    ke = 0.5 * masses[i] * np.linalg.norm(vs[i]) ** 2
    return ke

def KineticEnergy(vs, masses):
    '''
    Total kinetic energy of each particle. E_k = 1/2 * m * v^2
    
    input: - vs:     velocity of each particle
           - i:      particle to find force on
           - masses: mass of each particle      
           
    output: - kes: kinetic energies
    '''
        
    ke = 0.5 * masses.T @ np.array([np.linalg.norm(v) ** 2  for v in vs])
    return ke

def TotalKE(vs, masses):
    '''
    Total kinetic of system E_k = 1/2 * m * v^2
    
    input: - vs:     velocity of each particle
           - i:      particle to find force on
           - masses: mass of each particle      
           
    output: - kes: kinetic energies
    '''
    ke = 0.5 * masses.T @ np.array([np.linalg.norm(v) ** 2  for v in vs])
    return ke

def PE(rs, i, G, masses):
    '''
    Total potential energy of particle i. E_p = - ||F_i|| * ||r_i||
    
    input: - rs:     position of each particle
           - i:      particle to find force on
           - G:      gravitational constant
           - masses: mass of each particle      
           
    output: - pe: potential energy
    '''

    ri = rs[i]
    U = 0
    for j, rj in enumerate(rs):
        if i != j:
            # print(ri, rj, np.linalg.norm(rj - ri))
            Uij = masses[i] * masses[j] / np.linalg.norm(rj - ri)
            U += Uij 

    return - G * U

def PotentialEnergy(rs, G, masses):
    '''
    Total potential energy of each particle. E_p = - ||F_i|| * ||r_i||
    
    input: - rs:     position of each particle
           - G:      gravitational constant
           - masses: mass of each particle      
           
    output: - pe: potential energy
    '''

    N = len(rs)

    pes = [PE(rs, i, G, masses) for i in range(N)]
        
    return pes / 2

def TotalPE(rs, G, masses):
    '''
    Total potential energy of each particle. E_p = - ||F_i|| * ||r_i||
    
    input: - rs:     position of each particle
           - G:      gravitational constant
           - masses: mass of each particle      
           
    output: - pe: potential energy
    '''

    N = len(rs)

    U = 0

    for i in range(N):
        for j in range(i+1, N):
            U += masses[i] * masses[j] / np.linalg.norm(rs[j] - rs[i])
        
    return - G * U

def RelativeEnergy(E_traj):
    '''
    Relative change in energy of the system over time, as a percentage of initial energy
    
    input: - E_traj: trajectory of kenergies of each particle
           
    output: - dE:      trajectory of relative energy difference from initial energy
    '''
    Et = np.sum(E_traj, axis = 1)
    E0 = Et[0] # initial energy 
    E0hat = E0 # scaling coefficient 
    
    ## conditions to avoid dividing by zero 
    if E0hat == 0: E0hat = np.max(np.abs(E_traj[0])) 
    if E0hat == 0: E0hat = 1
    
    dE = np.abs(Et - E0) / np.abs(E0hat)
    
    return dE * 100

def Energies(rs, vs, G, masses):
    ke = KineticEnergy(vs, masses)
    pe = PotentialEnergy(rs, G, masses)
    return ke + pe

def TotalEnergy(rs, vs, G, masses):
    return TotalKE(vs, masses) + TotalPE(rs, G, masses)

# ============================
#      Angular Momentum
# ============================

def AM(rs, vs, i, masses):
    '''
    Angular momentum of particle i. l = r x v
    
    input: - rs:     position of each particle
           - vs:     velocity of each particle
           - i:      particle to find force on
           - masses: mass of each particle      
           
    output: - l: angular momentum
    '''
    l = np.cross(rs[i], masses[i]*vs[i])
    return l 

def AngMomentum(rs, vs, masses):
    '''
    Angular momentum of each particle. l = r x v
    
    input: - rs:     position of each particle
           - vs:     velocity of each particle
           - masses: mass of each particle      
           
    output: - L: angular momentums
    '''
        
    L = [AM(rs, vs, i, masses) for i in range(len(masses))]
    return np.array(L)

def RelativeAngMomentum(am_traj):
    '''
    Relative change in angular momentum of the system over time
    
    input: - am_traj: trajectory of angular momentum of each particle
           
    output: - dL:      trajectory of change in angular momentum from initial angular momentum
    '''
        
    Lt = np.sum(am_traj, axis = 1) # total angular momentum of the system
    L0 = Lt[0] # initial angular momentum
    L0hat = L0 # scaling coefficient 
    
    ## conditions to avoid dividing by zero 
    if L0hat == 0: L0hat = np.max(np.abs(am_traj))
    if L0hat == 0: L0hat = 1
        
    # scale schange in angular momentum
    dL = np.abs(Lt - L0) / np.abs(L0hat)
    return dL * 100
