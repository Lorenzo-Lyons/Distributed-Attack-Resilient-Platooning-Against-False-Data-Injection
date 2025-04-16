import numpy as np
import matplotlib.pyplot as plt
from classes_definintion import platooning_problem_parameters,Vehicle_model,set_scenario_parameters,generate_color_gradient,generate_color_1st_last_gray
from tqdm import tqdm
from scipy.linalg import sqrtm

from matplotlib import rc
font = {'family' : 'serif',
        #'serif': ['Times New Roman'],
        'size'   : 20}

rc('font', **font)





# in this script we simulate the vehicle longitudinal controller with CACC against the method in 
# "Constraining Attacker Capabilities Through Actuator Saturation"
# we will use the simplified version since we have only one constraint that is the collision avoidance with front vehicle
# All the comunication channels will be attacked and we will then show the reachable set obtained with this monte carlo roll-out for the 2 methods
# hopefully ours will be less conservative while still preserving safety
# also at some point emergency brake (see what happens cause they don't use this in the paper) (but maybe it's part of the input that is attacked?? check
#)

# ok they have a heterogenous platoon in the sense that each vehicle has different parameters
# but they all know about everybody else's parameters
# we could also do this but it would be a bit more complicated, but maybe worth it because they always ask about heterogenous platoon

# also another disadvantage is that you need to do everything in a centralized way. Ok you can do this off ine but if you get an additional vehicle 
# you need to re-run the optimization problem. But in our case also if the actuation bounds are different for the new vehicle hmhmh.

# also they don't consider additional state constraints (like speed limits), but only the indirect effect of the actuation being limited

# also they add artificial limits on the feed forward term, but don't consider that the linear controller also contributes to reaching the actuation limits
# this is actually quite a significant difference, because in our case we do consider the effect of both.



dt = 0.1 #[s]



# simulating for a platoon of 3 vehicles (so you can visualize the reachable set better)
# the dynamics of each vehicle are:
# x_i = x_i + v_i * dt
# v_i = v_i + u_i * dt

# we then express these dynamics with 
# d_tilde = x_i - x_i-1 + d
# v_tilde = v-v_d (v_d is the desired velocity of the platoon)

# we can then express the dynamics of the system with the new variables
# d_tilde_i = d_tilde_i + (v_tilde_i-v_tilde_i-1) * dt
# v_tilde_i = v_tilde_i + u_i * dt

# the controller can be expressed as
# u_i = - k * d_tilde_i -kh * v_tilde_i - c (v_tilde_i-v_tilde_i-1) 

# the controlled system is then for car number 1 (leader is car 0)

# d_tilde_1 = d_tilde_1 + v_tilde_1*dt - v_tilde_0*dt
# v_tilde_1 = v_tilde_1 - k * d_tilde_1*dt -kh * v_tilde_1*dt - c * v_tilde_1*dt + c * v_tilde_0


# the leader only has v_tilde as a state and just uses the relevant part in the controller, i.e.
# v_tilde_ 0 = v_tilde_0 - k * v_tilde_0 * dt - c * v_tilde_0*dt

# therefore the controlled system is

# x = [d1_tilde d2_tilde v0_tilde v1_tilde v2_tilde]^T
# x_k+1 = F x_k + G u_k


# load gains from previously saved values from "linear_controller_gains.py"
testing_on_the_real_robot = False # leave value to False to simulate on full scale vehicles
platooning_problem_parameters_obj = platooning_problem_parameters(testing_on_the_real_robot) 
v_d = platooning_problem_parameters_obj.v_d 
# vehicle maximum acceleration / braking
u_min = platooning_problem_parameters_obj.u_min  
u_max = platooning_problem_parameters_obj.u_max 
v_max = platooning_problem_parameters_obj.v_max 

# load gains from previously saved values from "linear_controller_gains.py"
params = np.load('saved_linear_controller_gains.npy')
k = params[0]
c = params[1]
h = params[2]
d = params[3]

# print out parameters
print('k = ',k)
print('c = ',c)
print('h = ',h)
print('d = ',d)

def build_state_transition_matrices(dt,k,c,h):
    F = np.array([[1    ,0   ,-dt     ,+dt          ,0            ],
                [0    ,1   ,0       ,-dt          ,+dt          ],
                [0    ,0   ,-k*h*dt ,-c*dt        ,0            ],
                [-k*dt,0   ,+c*dt   ,1-k*h*dt-c*dt,0            ],
                [0    ,-k*dt,0      ,+c*dt        ,1-k*h*dt-c*dt]])


    # an additional control action contribution comes from the feedforward term
    # and is simply added on top of the controlled system
    # u = [u1 u2 u3]^T
    G = np.array([[0,0,0],
                [0,0,0],
                [dt,0,0],
                [0,dt,0],
                [0,0,dt]])
    return F,G

F, G = build_state_transition_matrices(dt,k,c,h)


# note to self, to stack matrices later maybe just reorganize the states so they are 0 1 2 ecc so x = [v0 d1 v1 d2 v2] so you can build the matrix more easily



# define actuation bounds for the CACC controller, (this is according to their formulation)
u_max_cacc = 1 # [m/s^2] we will asuume all same bounds for all vehicles and symmetric in acceleration and braking


# evaluate the reachable set for the CACC controller with current bounds
# note that this is an ellips that encapsulates the reachable set of the system in a tight way
# reformulate as u**2 leq gamma
gamma = np.sqrt(u_max_cacc)



# solve the matrix inequality problem with CVXPY to determine tightest ellips of the reachable set
import cvxpy as cp
# Problem dimensions
n = F.shape[0]  # size of P
I = np.eye(G.shape[1])  # identity matrix of size n

# define values of parameter a
#a_vec = np.linspace(0.01, 0.99, 30) # we will grid search in this space
a_vec = np.linspace(0.95, 0.99, 10)




fig_d1d2, ax_d1d2 = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
ax_d1d2.set_xlabel('d1_tilde')
ax_d1d2.set_ylabel('d2_tilde')
ax_d1d2.set_title('d1_tilde vs d2_tilde')
ax_d1d2.legend()




def plot_ellipse(P,scale,label,ax):
    # P needs to be a 2 by 2 matrix
    # Create grid points on unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    circle = np.vstack((np.cos(theta), np.sin(theta)))  # shape (2, N)

    # Transform unit circle using the inverse square root of P_2d
    P_inv = np.linalg.inv(P)
    L = sqrtm(P_inv) # the dimensionality of the input bounds
    ellipse = scale * (L @ circle)

    ax.plot(ellipse[0, :], ellipse[1, :], label=label)
    ax.set_aspect('equal')
    ax.grid(True)




P_opt = np.zeros((n,n))
objective_value = np.infty
#R = np.diag([1/u_max_cacc**2,1/(u_max_cacc*2)**2,1/(u_max_cacc*3)**2])
R = np.diag([1/u_max_cacc**2,1/u_max_cacc**2,1/u_max_cacc**2])


for i in range(len(a_vec)): # tqdm(
    a = a_vec[i]

    # Define variable
    P = cp.Variable((n, n), symmetric=True)

    # Build F(P) as a CVXPY expression
    Q_expr = cp.bmat([
        [a * P - F.T @ P @ F,     -F.T @ P @ G],
        [-G.T @ P @ F, (1 - a) * R - G.T @ P @ G]
    ])

    # Constraints
    constraints = [
        P >> 0,    # P is positive definite
        (Q_expr + Q_expr.T) / 2 >> 0      # Q is positive semidefinite
    ]

    # Objective
    objective = cp.Minimize(-cp.log_det(P))

    # Problem definition and solving
    prob = cp.Problem(objective, constraints)
    #prob.solve(solver=cp.SCS, verbose=False)
    prob.solve(solver=cp.SCS, verbose=False) # , max_iters=1000

    print('')
    print('Problem status:', prob.status)
    print('a =', np.round(a,3), '  objective value:', np.round(prob.value,3))

    # Output the result
    if (prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE) and prob.value < objective_value: #update optimal value
        objective_value = prob.value
        # rescale the shape of the ellips using actual bound
        P_opt = P.value #* (1/gamma)

    # Extract the top-left 2x2 block of P
    P_2d = P.value[:2, :2]
    scale = np.sqrt(3) # G.shape[1]
    label = 'objective = ' + str(np.round(prob.value,2))


plot_ellipse(P_2d,scale,label,ax_d1d2)
ax_d1d2.legend()
























# define simulation time

# run simulation loop
t_sim = 50 # [s]
dt_sim = 0.01 # [s]
F_sim,G_sim = build_state_transition_matrices(dt_sim,k,c,h)

# define initial conditions
# [d1_tilde d2_tilde v0_tilde v1_tilde v2_tilde]^T
x_0 = np.array([0,0,0,0,0]) # all start too far away and going too slow 



# run simulation loop
sim_runs = 1
sim_steps = int(np.round(t_sim/dt_sim))








# Example setup if not already defined
# t_sim, dt, F, G should be defined prior to this point
# Assuming x_history already computed as per your code

colors = ['dodgerblue', 'orange', 'green', 'red', 'purple']
# Time vector
time = np.linspace(0, t_sim, sim_steps)

# Create figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)


def plot_trajectories_timeseries(axs,x_history,colors,ellipse_val,scale):
    # First subplot: d1, d2
    axs[0].plot(time, x_history[:, 0], label='d1',color=colors[1])
    axs[0].plot(time, x_history[:, 1], label='d2',color=colors[2])
    axs[0].set_ylabel('Distance States')
    axs[0].set_title('Distance State Evolution')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot: v0, v1, v2
    axs[1].plot(time, x_history[:, 2], label='v0',color=colors[0])
    axs[1].plot(time, x_history[:, 3], label='v1',color=colors[1])
    axs[1].plot(time, x_history[:, 4], label='v2',color=colors[2])
    axs[1].set_ylabel('Velocity States')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_title('Velocity State Evolution')
    axs[1].legend()
    axs[1].grid(True)

    # plot ellipse val
    axs[2].plot(time, scale * np.ones(sim_steps), label='max_val',color='gray',linestyle='--')
    axs[2].plot(time, ellipse_val, label='Ellipse Value',color='k')









for _ in range(sim_runs):

    # pre-allocate state matrix
    x_history = np.zeros((sim_steps, len(x_0)))
    # assign initial condition
    x_history[0, :] = x_0
    u_history = np.zeros((sim_steps, 3))
    ellipse_val = np.zeros(sim_steps)

    for i in range(1,sim_steps):
        # define random control input
        #u_comm = np.random.uniform(low=-u_max_cacc, high=u_max_cacc, size=(3,1))
        u_comm = np.array([[u_max_cacc],[u_max_cacc],[u_max_cacc]])
        #u_comm = np.array([[u_max_cacc],[2 * u_max_cacc],[3 * u_max_cacc]])
        
        # update state
        x_history[i,:] = np.squeeze(F_sim @ np.expand_dims(x_history[i-1,:],1) + G_sim @ u_comm)
        # store control input
        u_history[i,:] = np.squeeze(u_comm)

        # evaluate - log determinant of P
        ellipse_val[i] = np.expand_dims(x_history[i-1,:],0) @ P_opt @ np.expand_dims(x_history[i-1,:],1)

    ax_d1d2.plot(x_history[:,0],x_history[:,1],label='d1 d2 trajectory',color='dodgerblue')
    plot_trajectories_timeseries(axs,x_history,colors,ellipse_val,scale)




# now plot simulation
#show the trajectories in the d1 d2 space
# fig_d1d2, ax_d1d2 = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
# ax_d1d2.plot(x_history[:,0],x_history[:,1],label='d1 d2 trajectory',color='dodgerblue')
# ax_d1d2.set_xlabel('d1_tilde')
# ax_d1d2.set_ylabel('d2_tilde')
# ax_d1d2.set_title('d1_tilde vs d2_tilde')
# ax_d1d2.legend()








plt.tight_layout()
plt.show()


