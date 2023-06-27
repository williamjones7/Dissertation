import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functions import * 

# ============================
#        Plot in 2D
# ============================

def plot2D(t_traj, rs_traj, vs_traj, ke_traj, pe_traj, am_traj, masses, scheme='',dark = True):
    '''
    Plot the orbits, energy and angular momentum of a system of N-bodies
    
    input: - t_vals:  list of time values
           - rs_traj: trajectory of positions of each particle 
           - vs_traj: trajectory of velocity of each particle 
           - ke_traj: trajectory of kinetic energy of each particle 
           - pe_traj: trajectory of potential energy of each particle 
           - am_traj: trajectory of angular momentum of each particle
    '''
    
    N = rs_traj.shape[1] # number of masses in the system
    colours = plt.cm.rainbow(np.linspace(0,1,N)) # colour according to how many masses
    
    colourface = 'black'
    if dark == True: 
        plt.style.use('dark_background')
        colourface = 'white'
        
    # create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))

    ### ORBITS ###
    
    # loop over all masses 
    for i in range(N):
        ri_traj = rs_traj[:,i,:] # get the i-th trajectory
        ax1.plot(ri_traj[:,0], ri_traj[:,1], color = colours[i],linestyle='-', zorder = 1, label=f'mass {i+1}') # plot the orbits
        ax1.scatter(ri_traj[0,0],ri_traj[0,1],color=colours[i],marker="o", facecolors='none', s=50, zorder = 2) # plot the start positions
        ax1.scatter(ri_traj[-1,0],ri_traj[-1,1],color=colours[i],marker="o",s=50, zorder = 2) # plot the final positions of the
    
    # find and plot the centre of mass
    rcoms = []
    for i in range(rs_traj.shape[0]):
        rcom, vcom = CentreOfMass(rs_traj[i], vs_traj[i], masses)
        rcoms = rcoms + [rcom]
    rcoms = np.array(rcoms)
    ax1.scatter(rcoms[:,0], rcoms[:,1], color = colourface, label = 'Centre of mass', marker = 'x', zorder = 3)
    
    ax1.set_title(f'Trajectories of {N} masses in $z=0$ using {scheme}')
    ax1.set_xlabel('x-coordinate')
    ax1.set_ylabel('y-coordinate')
    ax1.legend()
    
    ### ENERGY ###
    
    # get the relative change in energy 
    relative_e_traj, relative_ke_traj, relative_pe_traj = EnergyDiff(ke_traj, pe_traj)
    
    # ax2.plot(t_traj, relative_ke_traj, color = 'red', label = 'Kinetic Energy')
    # ax2.plot(t_traj, relative_pe_traj, color = 'blue', label = 'Potential Energy')
    ax2.plot(t_traj, relative_e_traj, color = colourface, linestyle = '--', label = 'Total Energy')
    
    ax2.set_title(f'Relative change in energy using {scheme}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Relative change in energy')
    ax2.legend()
    
    ### ANGULAR MOMENTUM ###
    
    am_traj = am_traj[:,:,-1]
    relative_am_traj = AngMomentumDiff(am_traj)
    
    ax3.plot(t_traj, relative_am_traj)
    ax3.set_title(f'Relative change in angular momentum using {scheme}', y = 1.08 )
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Relative change in angular momentum')
