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
    
    Fs = np.zeros_like(rs).astype('float64') # empty vector of forces 
    
    for i, r in enumerate(rs): # loop over all particles
        Fs[i] = Force_i(rs, i, G, masses) # calculate force acting on that particle
        
    return Fs

def dri_dt(vs, i): 
    '''Returns the r-derivative of particle i'''
    return vs[i]

def dvi_dt(rs, i, G, masses):
    '''Returns the second r-derivative (accelation) of particle i, from F = ma'''
    return Force_i(rs, i, G, masses) / masses[i]

def dr_dt(vs):
    '''Returns the r-derivative of all particles'''
    return vs

def dv_dt(rs, G, masses):
    '''Returns the second r-derivative (acceleration) of all particles, from F = ma'''
    Fs = Force(rs, G, masses)
    return np.divide(Fs, masses[:, np.newaxis])

def all_derivatives(w, t, G, masses):
    '''
    Finds derivative, in every cardinal direction, of position and velocity of each particle
    
    input: - w:      flattened list of positions and velocities of each mass
           - t:      time
           - G:      gravitational constant
           - masses: mass of each particle      
           
    output: flattened gradient of position and velocity
    '''
    
    W = np.reshape(w, (len(w) // 3, 3)) # store position and velocity in 3D
    rs, vs = np.split(W, 2) # separate positions and velocities
    
    drdt = dr_dt(vs) # find velocity of all particles
    dvdt = dv_dt(rs, G, masses) # find acceleration of all particles
    
    derivs = np.concatenate((drdt,dvdt)).flatten() # reformat to be used by scipy integrate
    
    return derivs

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
        
    kes = [KE(vs, i, masses) for i in range(len(masses))]
    return np.array(kes)

def PE(rs, i, G, masses):
    '''
    Total potential energy of particle i. E_p = - ||F_i|| * ||r_i||
    
    input: - rs:     position of each particle
           - i:      particle to find force on
           - G:      gravitational constant
           - masses: mass of each particle      
           
    output: - pe: potential energy
    '''
  
    force_i = Force_i(rs, i, G, masses)
    fmag = np.linalg.norm(force_i)
    rmag = np.linalg.norm(rs[i])
    return -fmag * rmag

def PotentialEnergy(rs, G, masses):
    '''
    Total potential energy of each particle. E_p = - ||F_i|| * ||r_i||
    
    input: - rs:     position of each particle
           - G:      gravitational constant
           - masses: mass of each particle      
           
    output: - pe: potential energy
    '''
        
    Fs = Force(rs, G, masses)
    Fmags = np.linalg.norm(Fs, axis = 1)
    rmags = np.linalg.norm(rs, axis = 1)
    pes = [-1 * Fmags[i] *  rmags[i] for i in range(len(Fmags))]
    return np.array(pes)

def EnergyDiff(ke_traj, pe_traj):
    '''
    Relative change in energy of the system over time
    
    input: - ke_traj: trajectory of kinetic energies of each particle
           - pe_traj: trajectory of postential energies of each particle
           
    output: - dE:      trajectory of energy difference from initial energy
            - ke_traj: scaled kinetic energy trajectory 
            - pe_traj: scaled potential energy trajectory 
    '''
    
    total_ke = np.sum(ke_traj, axis = 1) # total kinetic energy of system
    total_pe = np.sum(pe_traj, axis = 1) # total potential energy of system
    Et = total_ke + total_pe # total energy of system
    E0 = Et[0] # initial energy 
    E0hat = E0 # scaling coefficient 
    
    ## conditions to avoid dividing by zero 
    if E0hat == 0: E0hat = np.max(np.abs(kes[0]+pes[0])) 
    if E0hat == 0: E0hat = np.max(np.abs(kes[0]))
    if E0hat == 0: E0hat = np.max(np.abs(pes[0]))
    if E0hat == 0: E0hat = 1
        
    # scale change in energy accross total, kinetic and potential energies 
    dE = (Et - E0) / np.abs(E0hat)
    kes = (total_ke - total_ke[0]) / np.abs(E0hat)
    pes = (total_pe - total_pe[0]) / np.abs(E0hat)
    
    return dE, kes, pes


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

def AngMomentumDiff(am_traj):
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
    return dL