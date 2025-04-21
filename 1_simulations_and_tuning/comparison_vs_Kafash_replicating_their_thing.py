import numpy as np
import matplotlib.pyplot as plt
from classes_definintion import platooning_problem_parameters,Vehicle_model,set_scenario_parameters,generate_color_gradient,generate_color_1st_last_gray
from Kafash_functions import plot_ellipse,plot_trajectories_timeseries
from tqdm import tqdm
from scipy.linalg import sqrtm

from matplotlib import rc
font = {'family' : 'serif',
        #'serif': ['Times New Roman'],
        'size'   : 12}

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



# Platooning example
Dt = 0.5    # sampling period
beta = -0.1 # velocity loss caused by friction
d_opt = 1   # desired distance between vehicles
kp = 0.2    # proportional gain of the forward-and-reverse-looking PD control 
kd = 0.3    # derivative gain of the forward-and-reverse-looking PD control 


# simulating for a platoon of 3 vehicles (so you can visualize the reachable set better)
# the dynamics of each vehicle are:
# x_i = x_i + v_i * Dt
# v_i = v_i + u_i * Dt

# we then express these dynamics with 
# d_tilde = x_i - x_i-1 - d
# v_tilde = v-v_d (v_d is the desired velocity of the platoon)

# we can then express the dynamics of the system with the new variables
# d_tilde_i = d_tilde_i + (v_tilde_i-v_tilde_i-1) * Dt
# v_tilde_i = v_tilde_i + u_i * Dt

# the controller can be expressed as
# u_i = - k * d_tilde_i -kh * v_tilde_i - c (v_tilde_i-v_tilde_i-1) 

# the controlled system is then for car number 1 (leader is car 0)

# d_tilde_1 = d_tilde_1 + v_tilde_1*Dt - v_tilde_0*Dt
# v_tilde_1 = v_tilde_1 - k * d_tilde_1*Dt -kh * v_tilde_1*Dt - c * v_tilde_1*Dt + c * v_tilde_0


# the leader only has v_tilde as a state and just uses the relevant part in the controller, i.e.
# v_tilde_ 0 = v_tilde_0 - k * v_tilde_0 * Dt - c * v_tilde_0*Dt

# therefore the controlled system is

# x = [d1_tilde d2_tilde v0_tilde v1_tilde v2_tilde]^T
# x_k+1 = F x_k + G u_k


# load gains from previously saved values from "linear_controller_gains.py"

def build_state_transition_matrices(Dt,k,c,h):

    # Define matrix F
    F = np.array([
        [1,  0, -Dt,          Dt,           0],
        [0,  1,  0,          -Dt,          Dt],
        [kp, 0,  (1 + beta) - kd, kd,       0],
        [-kp, kp, kd, (1 + beta) - 2*kd, kd],
        [0, -kp, 0,           kd, (1 + beta) - kd]
    ])

    # Define matrix G
    G = np.vstack([
        np.zeros((2, 3)),
        Dt * np.eye(3)
    ])

    print("F =\n", F)
    print("G =\n", G)
    return F,G

F, G = build_state_transition_matrices(Dt,kp,kd,beta)


# Build matrix R with the actuation bounds for the CACC controller (assuming symmetric bound in acceleration and braking)
u_max_cacc = np.array([1.1,0.9,1.05]) # [m/s^2] we will asuume all same bounds for all vehicles and symmetric in acceleration and braking


# evaluate the reachable set for the CACC controller with current bounds
# note that this is an ellips that encapsulates the reachable set of the system in a tight way
# reformulate as u**2 leq gamma
gamma = np.sqrt(u_max_cacc)
R = np.diag([1/gamma[0],1/gamma[1],1/gamma[2]]) # this is the matrix that we will use to define the ellips
print('R = \n', R)








# solve the matrix inequality problem with CVXPY to determine tightest ellips of the reachable set
import cvxpy as cp
# Problem dimensions
n = F.shape[0]  # size of P
m = G.shape[1]  # size of u
I = np.eye(m)  # identity matrix of size n

# define values of parameter a
#a_vec = np.linspace(0.01, 0.99, 11)
#a_vec = np.linspace(0.8, 0.99, 11)
#a_vec = np.linspace(0.83, 0.88, 11)
a_vec = np.array([0.855]) # once we have found the optimal value we can just use this



fig_d1d2, ax_d1d2 = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
ax_d1d2.set_xlabel('d1_tilde')
ax_d1d2.set_ylabel('d2_tilde')
ax_d1d2.set_title('d1_tilde vs d2_tilde')
ax_d1d2.legend()









P_opt = np.zeros((n,n))
objective_value = np.infty


for i in range(len(a_vec)): # tqdm(
    a = a_vec[i]

    # Define variable
    P = cp.Variable((n, n), symmetric=True)

    # Build F(P) as a CVXPY expression
    Q_expr = cp.bmat([
        [a * P - F.T @ P @ F,     -F.T @ P @ G],
        [-G.T @ P @ F, (1 - a) * R - G.T @ P @ G]
    ])
    # np.transpose(
    # Constraints
    constraints = [
        P-0.0001 * np.eye(P.shape[0]) >> 0,    # P is positive definite
        Q_expr >> 0      # Q is positive semidefinite
    ]

    # Objective
    objective = cp.Minimize(-cp.log_det(P))

    # Problem definition and solving
    prob = cp.Problem(objective, constraints)
    #prob.solve(solver=cp.SCS, verbose=False)
    prob.solve(solver=cp.CLARABEL, verbose=False) # , max_iters=1000

    print('')
    print('Problem status:', prob.status)
    print('a =', np.round(a,3), '  objective value:', np.round(prob.value,3))

    # Output the result
    if (prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE) and prob.value < objective_value: #update optimal value
        objective_value = prob.value
        # rescale the shape of the ellips using actual bound
        P_opt = P.value #* (1/gamma)
        P_2d_opt = P_opt[:2, :2]
        label_opt = 'optimal a = ' + str(np.round(a,3)) + '  objective = ' + str(np.round(prob.value,3))

    # Extract the top-left 2x2 block of P
    P_2d = P.value[:2, :2]
    scale = np.sqrt(m)
    label = 'objective = ' + str(np.round(prob.value,2))
    # plot the ellips
    #color = 'gray'
    #plot_ellipse(P_2d,scale,label,ax_d1d2,color)


# plot the best ellips
plot_ellipse(P_2d_opt,m,ax_d1d2, edgecolor='skyblue', linewidth=2,facecolor='none',label = label_opt)
ax_d1d2.legend()
ax_d1d2.set_xlim([-6, 6])
ax_d1d2.set_ylim([-6, 6])















# define simulation time

# run simulation loop
t_sim = 50 # [s]
dt_sim = Dt #0.01 # [s]
# F_sim,G_sim = build_state_transition_matrices(dt_sim,k,c,h)
F_sim = F
G_sim = G

# define initial conditions
# [d1_tilde d2_tilde v0_tilde v1_tilde v2_tilde]^T
x_0 = np.array([0,0,0,0,0]) # all start too far away and going too slow 



# run simulation loop
sim_runs = 1
sim_steps = int(np.round(t_sim/dt_sim))








# Example setup if not already defined
# t_sim, Dt, F, G should be defined prior to this point
# Assuming x_history already computed as per your code

# Time vector
time = np.linspace(0, t_sim, sim_steps)

# Create figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)










add_label = True
for _ in range(sim_runs):

    # pre-allocate state matrix
    x_history = np.zeros((sim_steps, len(x_0)))
    # assign initial condition
    x_history[0, :] = x_0
    u_history = np.zeros((sim_steps, 3))
    ellipse_val = np.zeros(sim_steps)

    for i in range(1,sim_steps):
        # define control input from cacc
        # emergency brake while 2nd car accelerates
        u_cacc = np.array([[-u_max_cacc[0]],[+u_max_cacc[1]],[u_max_cacc[2]]])

        # # random input
        # u1= np.random.uniform(low=-u_max_cacc[0], high=u_max_cacc[0])
        # u2= np.random.uniform(low=-u_max_cacc[1], high=u_max_cacc[1])
        # u3= np.random.uniform(low=-u_max_cacc[2], high=u_max_cacc[2])
        # u_cacc = np.array([[u1],[u2],[u3]])

        
        # update state
        x_history[i,:] = np.squeeze(F_sim @ np.expand_dims(x_history[i-1,:],1) + G_sim @ u_cacc)
        # store control input
        u_history[i,:] = np.squeeze(u_cacc)

        # evaluate - log determinant of P
        ellipse_val[i] = np.expand_dims(x_history[i-1,:],0) @ P_opt @ np.expand_dims(x_history[i-1,:],1)

    if add_label:
        label_traj = 'trajectory attack-mitigation OFF'
    else:   
        label_traj = ''
    ax_d1d2.plot(x_history[:,0],x_history[:,1],label=label_traj,color='dodgerblue')
    ax_d1d2.legend()
    attack_mitigation = False
    plot_trajectories_timeseries(axs,time,x_history,ellipse_val,scale,add_label,attack_mitigation)
    add_label = False





# now add the knowledge of the dangerous regions
#  Dangerous regions: -x1>=d1* and -x2>=d2*
#  We need to rewrite them in the form ci'x = bi

# collision accours for large negative values of d1_tilde and d2_tilde. 
# d_tilde1 = x0-x1-d 
# so dangerous region is d1_tilde >= -1 (or some other positive number)

c1 = np.zeros((n, 1))
c1[0, 0] = -1  
c2 = np.zeros((n, 1))
c2[1, 0] = -1
# Define b1 and b2
b1 = 1
b2 = 1



# Optimization variables
Y = cp.Variable((n, n), PSD=True)
R_hat_diag = cp.Variable(m)  # diagonal entries
R_hat = cp.diag(R_hat_diag)

# using Laura's solution
#R_hat = np.diag([35.3950, 19.5614, 35.3950]) # this is the matrix that we will use to define the ellips



# Define constraints
constraints = [
    R_hat >> R,
    cp.quad_form(c1, Y) <= (b1 ** 2) / m,
    cp.quad_form(c2, Y) <= (b2 ** 2) / m,
    cp.bmat([
        [a * Y,             np.zeros((n, m)), Y @ F.T],
        [np.zeros((m, n)), (1 - a) * R_hat,   G.T],
        [F @ Y,             G,                Y]
    ]) >> 0
]

# Solve the SDP
prob = cp.Problem(cp.Minimize(cp.trace(R_hat)), constraints)
prob.solve(solver=cp.CLARABEL) # , max_iters=1000
print("Problem status:", prob.status)

# print("available solvers:")
# print(cp.installed_solvers())


# Results
print("R_hat =\n", R_hat.value)
print("inv(Y) =\n", np.linalg.inv(Y.value))
print("eig(Y) =\n", np.linalg.eigvals(Y.value))


# what is the 
Y_inv = np.linalg.inv(Y.value)

# plot the ellipse
P_2d_hat = Y.value[:2, :2]
P_2d_hat_inv = np.linalg.inv(P_2d_hat)
plot_ellipse(P_2d_hat_inv,m,ax_d1d2,edgecolor='coral', linewidth=2,facecolor='none',label = 'R_hat')





#plot the bounds
# draw vertcal red line for b1
ax_d1d2.axvline(x=-b1, color='red', linestyle='--', label='b1')
#draw horizontal red line for b2
ax_d1d2.axhline(y=-b2, color='red', linestyle='--', label='b2')

# add legend
ax_d1d2.legend()



# now simulate the system with the new bounds
# define new bounds
# extract diag entries of R_hat
R_hat_diag = np.diag(R_hat.value)
u_max_controlled = np.array([np.sqrt(1/R_hat_diag[0]),
                             np.sqrt(1/R_hat_diag[1]),
                             np.sqrt(1/R_hat_diag[2])])






add_label = True
for _ in range(sim_runs):

    # pre-allocate state matrix
    x_history = np.zeros((sim_steps, len(x_0)))
    # assign initial condition
    x_history[0, :] = x_0
    u_history = np.zeros((sim_steps, 3))
    ellipse_val = np.zeros(sim_steps)

    for i in range(1,sim_steps):
        # define control input from cacc
        # emergency brake while 2nd car accelerates
        u_cacc = np.array([[-u_max_controlled[0]],[+u_max_controlled[1]],[u_max_controlled[2]]])

        # # random input
        # u1= np.random.uniform(low=-u_max_cacc[0], high=u_max_cacc[0])
        # u2= np.random.uniform(low=-u_max_cacc[1], high=u_max_cacc[1])
        # u3= np.random.uniform(low=-u_max_cacc[2], high=u_max_cacc[2])
        # u_cacc = np.array([[u1],[u2],[u3]])

        
        # update state
        x_history[i,:] = np.squeeze(F_sim @ np.expand_dims(x_history[i-1,:],1) + G_sim @ u_cacc)
        # store control input
        u_history[i,:] = np.squeeze(u_cacc)

        # evaluate - log determinant of P
        ellipse_val[i] = np.expand_dims(x_history[i-1,:],0) @ Y_inv @ np.expand_dims(x_history[i-1,:],1)

    if add_label:
        label_traj = 'trajectory attack-mitigation ON'
    else:   
        label_traj = ''
    ax_d1d2.plot(x_history[:,0],x_history[:,1],label=label_traj,color='orangered')
    ax_d1d2.legend()
    attack_mitigation = True
    plot_trajectories_timeseries(axs,time,x_history,ellipse_val,scale,add_label,attack_mitigation)
    add_label = False




plt.tight_layout()
plt.show()


