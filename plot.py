import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functions import * 

# ============================
#        Plot in 2D
# ============================

def plot2D(trajectories, masses, scheme='', style = 'default', COM = True):
    '''
    Plot the orbits, energy and angular momentum of a system of N-bodies
    
    input: - t_vals:  list of time values
           - rs_traj: trajectory of positions of each particle 
           - vs_traj: trajectory of velocity of each particle 
           - ke_traj: trajectory of kinetic energy of each particle 
           - pe_traj: trajectory of potential energy of each particle 
           - am_traj: trajectory of angular momentum of each particle
    '''
    t_traj, rs_traj, vs_traj, E_traj, am_traj, time = trajectories
    N = rs_traj.shape[1] # number of masses in the system
    colours = plt.cm.rainbow(np.linspace(0,1,N)) # colour according to how many masses
    
    plt.style.use(style)
    
        
    # create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36,12))

    ### ORBITS ###
    
    # loop over all masses 
    for i in range(N):
        ri_traj = rs_traj[:,i,:] # get the i-th trajectory
        ax1.plot(ri_traj[:,0], ri_traj[:,1], color = colours[i], zorder = 1, label=f'mass {i+1}') # plot the orbits
        ax1.scatter(ri_traj[0,0],ri_traj[0,1],color=colours[i],marker="o", facecolors='none', s=50, zorder = 2) # plot the start positions
        ax1.scatter(ri_traj[-1,0],ri_traj[-1,1],color=colours[i],marker="o",s=50, zorder = 2) # plot the final positions of the
    
    if COM: 
        # find and plot the centre of mass
        rcoms = []
        for i in range(rs_traj.shape[0]):
            rcom, vcom = CentreOfMass(rs_traj[i], vs_traj[i], masses)
            rcoms = rcoms + [rcom]
        rcoms = np.array(rcoms)
        ax1.scatter(rcoms[:,0], rcoms[:,1], label = 'Centre of mass', marker = 'x', zorder = 3)
    
    ax1.set_aspect(aspect = 'equal')
    ax1.set_title(f'{scheme}, t = {np.round(time, 3)}', y = 1.05)
    ax1.set_xlabel('x-coordinate')
    ax1.set_ylabel('y-coordinate')
    ax1.legend()
    
    ### ENERGY ###
    
    # get the relative change in energy 
    relative_e_traj = RelativeEnergy(E_traj)
    
    # ax2.plot(t_traj, relative_ke_traj, color = 'red', label = 'Kinetic Energy')
    # ax2.plot(t_traj, relative_pe_traj, color = 'blue', label = 'Potential Energy')
    ax2.plot(t_traj, relative_e_traj, label = 'Total Energy')
    
    ax2.set_title(f'Relative energy error using {scheme}', y = 1.05)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Percentage error')
    # ax2.legend()
    
    ### ANGULAR MOMENTUM ###
    
    am_traj = am_traj[:,:,-1]
    relative_am_traj = RelativeAngMomentum(am_traj)
    
    ax3.plot(t_traj, relative_am_traj)
    ax3.set_title(f'Relative angular momentum error using {scheme}', y = 1.05 )
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Percentage error')

def PlotOrbits(trajectories, figsize = (7,7)):
    t_traj, rs_traj, vs_traj, E_traj, am_traj, time = trajectories
    N = rs_traj.shape[1] # number of masses in the system

    colours = plt.cm.rainbow(np.linspace(0,1,N)) # colour according to how many masses
    
    plt.style.use('default')

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # loop over all masses 
    for i in range(N):
        color = colours[i]
        ri_traj = rs_traj[:,i,:] # get the i-th trajectory
        vi_traj = vs_traj[:,i,:]
        ax.plot(ri_traj[:,0], ri_traj[:,1],  zorder = 1, label=f'mass {i+1}', color=color) # plot the orbits
        ax.scatter(ri_traj[0,0],ri_traj[0,1],marker="o",s=50, zorder = 2, color=color) # plot the start positions
        # ax.scatter(ri_traj[-1,0],ri_traj[-1,1],marker="o",s=50, zorder = 2, color=color) # plot the start positions


    # find and plot the centre of mass
    # rcoms = []
    # for i in range(rs_traj.shape[0]):
    #     rcom, vcom = CentreOfMass(rs_traj[i], vs_traj[i], masses)
    #     rcoms = rcoms + [rcom]
    # rcoms = np.array(rcoms)
    # ax.scatter(rcoms[:,0], rcoms[:,1], color = 'black', label = 'Centre of mass', marker = 'x', zorder = 3)

    ax.set_aspect(aspect = 'equal')
    # ax.text(0.05, 0.95, f'T = {T}, h = {h}, time = {np.round(time, 5)}', transform=ax.transAxes, 
    #         va='top', fontsize = 15)
    ax.set_xlabel('x-coordinate', fontsize = 15)
    ax.set_ylabel('y-coordinate', fontsize = 15)
    ax.legend(fontsize = 15)

    plt.tight_layout()

def PlotEnergy(trajectories):
    t_traj, rs_traj, vs_traj, E_traj, am_traj, time = trajectories

    ### ENERGY ###
    
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    
    relative_e_traj = RelativeEnergy(E_traj) 
    
    ax.plot(t_traj, relative_e_traj, label = 'Total Energy')
    
    ax.set_xlabel('Time', fontsize = 12)
    ax.set_ylabel('Relative Energy Error (%)', fontsize = 12)

    plt.tight_layout()

def PlotTotalEnergy(trajectories, figsize = (7,7)):
    t_traj, rs_traj, vs_traj, E_traj, am_traj, time = trajectories

    ### ENERGY ###
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.plot(t_traj, E_traj, label = 'Total Energy')
    
    ax.set_xlabel('Time', fontsize = 12)
    ax.set_ylabel('Relative Energy Error (%)', fontsize = 12)

    plt.tight_layout()

def PlotPath(path, figsize = (7,7)):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.plot(path[:,0], path[:,1], label = 'Total Energy', color = 'navy')
    
    ax.set_xlabel(r'$x1$', fontsize = 12)
    ax.set_ylabel(r'$x2$', fontsize = 12)

    ax.set_aspect('equal')
    plt.tight_layout()


def PlotAngularMomentum(trajectories):
    t_traj, rs_traj, vs_traj, E_traj, am_traj, time = trajectories

    ### ANGULAR MOMENTUM ###
    
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    
    relative_am_traj = RelativeAngMomentum(am_traj[:,-1]) 
    
    ax.plot(t_traj, relative_am_traj)
    
    ax.set_xlabel('Time', fontsize = 12)
    ax.set_ylabel('Relative Angular Momentum Error (%)', fontsize = 12)

    plt.tight_layout()

def PlotError(err_traj, trajectory):
    t_traj, rs_traj, vs_traj, E_traj, am_traj, time = trajectory

    ### ENERGY ###
    
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    
    ax.plot(t_traj, err_traj, label = 'Total Energy')
    
    ax.set_xlabel('Time', fontsize = 12)
    ax.set_ylabel('Absolute Error', fontsize = 12)

    plt.tight_layout()