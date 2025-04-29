import numpy as np
import matplotlib.pyplot as plt
from classes_definintion import platooning_problem_parameters,Vehicle_model,set_scenario_parameters,generate_color_gradient,generate_color_1st_last_gray
from Kafash_functions import plot_ellipse, plot_trajectories_timeseries,reachable_set_given_bounds, evaluate_new_bounds_with_constraints
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
u_min_original = platooning_problem_parameters_obj.u_min  
u_max_original = platooning_problem_parameters_obj.u_max
# chose min abs val because the method need symmetric bounds
u_max = np.max(np.abs([u_max_original,u_min_original])) # [m/s^2] we will asuume all same bounds for all vehicles and symmetric in acceleration and braking
#u_max = 1  # NOTE temp overrride


print('v_d = ',v_d)
print('v_max = ',platooning_problem_parameters_obj.v_max)
print('u_min_original = ',u_min_original)
print('u_max_original = ',u_max_original)
print('u_max = ',u_max)

# no max speed used
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



# reachable distance if car 1 is still and car 2 receives maximum acc input
print('reachable d if car 1 is still and car 2 receives maximum acc input: u_max/k = ',u_max/k)

def build_state_transition_matrices(dt,k,c,h):
    #     F = np.array([
    #     [1,  0, -Dt,          Dt,           0],
    #     [0,  1,  0,          -Dt,          Dt],
    #     [kp, 0,  (1 + beta) - kd, kd,       0],
    #     [-kp, kp, kd, (1 + beta) - 2*kd, kd],
    #     [0, -kp, 0,           kd, (1 + beta) - kd]
    # ])

    F = np.array([[1    ,0    ,-dt           ,+dt           ,0            ],
                  [0    ,1    ,0             ,-dt           ,+dt          ],
                  [0    ,0    ,1-k*h*dt-c*dt ,0             ,0            ],
                  [-k*dt,0    ,+c*dt         ,1-k*h*dt-c*dt ,0            ],
                  [0    ,-k*dt,0             ,+c*dt         ,1-k*h*dt-c*dt]])
    # beta = 0.1
    # F = np.array([[1    ,0    ,-dt           ,+dt           ,0            ],
    #               [0    ,1    ,0             ,-dt           ,+dt          ],
    #               [0    ,0    ,1-beta ,0             ,0            ],
    #               [-k*dt,0    ,+c*dt         ,1-beta-k*h*dt-c*dt ,0            ],
    #               [0    ,-k*dt,0             ,+c*dt         ,1-beta-k*h*dt-c*dt]])
    

    # an additional control action contribution comes from the feedforward term
    # and is simply added on top of the controlled system
    # u = [u1 u2 u3]^T
    # Define matrix G
    G = np.vstack([
        np.zeros((2, 3)),
        dt * np.eye(3)
    ])
    # check that F describes a stable system
    eigenvalues = np.linalg.eigvals(F)
    # print eigenvalues with 2 decimal points
    # check if they are all in the unit circle
    if np.all(np.abs(eigenvalues) < 1):
        print('Eigenvalues are inside the unit circle')
    eigenvalues = np.round(eigenvalues, 2)
    print('Eigenvalues of F:', eigenvalues)


    return F,G

F, G = build_state_transition_matrices(dt,k,c,h)









# note to self, to stack matrices later maybe just reorganize the states so they are 0 1 2 ecc so x = [v0 d1 v1 d2 v2] so you can build the matrix more easily






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






# Build matrix R with the actuation bounds for the CACC controller (assuming symmetric bound in acceleration and braking)
u_max_cacc = np.array([u_max,u_max,u_max]) # [m/s^2] we will asuume all same bounds for all vehicles and symmetric in acceleration and braking


# evaluate the reachable set for the CACC controller with current bounds
# note that this is an ellips that encapsulates the reachable set of the system in a tight way
# reformulate as u**2 leq gamma
gamma = u_max**2
R = np.diag([1/gamma,1/gamma,1/np.sqrt(0.01)]) # this is the matrix that we will use to define the ellips
print('R = \n', R)
print('1/gamma = ', 1/gamma)








# solve the matrix inequality problem with CVXPY to determine tightest ellips of the reachable set
import cvxpy as cp
# Problem dimensions
n = F.shape[0]  # size of P
m = G.shape[1]  # size of u
I = np.eye(m)  # identity matrix of size n

# define values of parameter a
a_vec = np.linspace(0.01, 0.99, 21)
#a_vec = np.linspace(0.6, 0.9, 11)
#a_vec = np.linspace(0.95, 0.99, 11)
#a_vec = np.array([0.2]) # once we have found the optimal value we can just use this



fig_d1d2, ax_d1d2 = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
ax_d1d2.set_xlabel('d1_tilde')
ax_d1d2.set_ylabel('d2_tilde')
ax_d1d2.set_title('d1_tilde vs d2_tilde')
ax_d1d2.legend()








P_opt = np.zeros((n,n))
objective_value = np.infty
a_opt = 0


for i in range(len(a_vec)): # tqdm(
    a = a_vec[i]


    P,prob = reachable_set_given_bounds(a,F,G,R)
    print('a =', np.round(a,3), '  objective value:', np.round(prob.value,3))

    # Output the result
    if (prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE) and prob.value < objective_value: #update optimal value
        objective_value = prob.value
        # rescale the shape of the ellips using actual bound
        #P_opt_unscaled = P.value 
        #P_opt = 1/gamma *  P_opt_unscaled
        P_opt = P
        label_opt = 'optimal a = ' + str(np.round(a,3)) + '  objective = ' + str(np.round(prob.value,3))
        a_opt = a



# print resutls
print('')
print('a_opt = ', np.round(a_opt,3))
print('objective value = ', np.round(objective_value,3))
print('P_opt = \n', P_opt)
# print out the eigenvalues of P_opt
eigenvalues = np.linalg.eigvals(P_opt)
print('eigenvalues of P_opt = ', eigenvalues)
print('')


# plot the best ellips
#plot_ellipse(P_opt_unscaled[:2,:2],m,ax_d1d2, edgecolor='k', linewidth=2,facecolor='none',label = 'unscaled ellipse')
plot_ellipse(P_opt[:2,:2],m,ax_d1d2, edgecolor='skyblue', linewidth=2,facecolor='none',label = label_opt)
ax_d1d2.legend()
ax_d1d2.set_xlim([-d*1.1, d*1.1])
ax_d1d2.set_ylim([-d*1.1, d*1.1])







# define simulation time

# run simulation loop
t_sim = 50 # [s]
dt_sim = dt #0.01 # [s]
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




# simulating an emergency brake scenario while attack is ongoing
t_brake_emergency = 250 # [s] time the attack begins


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
        u_cacc = np.array([[0],[+u_max],[u_max]])

        # just the system dynamics
        #u_cacc = np.array([[0],[0],[0]])

        # # random input
        # u1= np.random.uniform(low=-u_max_cacc[0], high=u_max_cacc[0])
        # u2= np.random.uniform(low=-u_max_cacc[1], high=u_max_cacc[1])
        # u3= np.random.uniform(low=-u_max_cacc[2], high=u_max_cacc[2])
        # u_cacc = np.array([[u1],[u2],[u3]])
        new_state_candidate = np.squeeze(F_sim @ np.expand_dims(x_history[i-1,:],1) + G_sim @ u_cacc)
        
        
        # # check for max velocity
        # if new_state_candidate[2] > v_max -v_d:
        #     new_state_candidate[2] = v_max -v_d
        # if new_state_candidate[3] > v_max -v_d:
        #     new_state_candidate[3] = v_max -v_d
        # if new_state_candidate[4] > v_max -v_d:
        #     new_state_candidate[4] = v_max -v_d
        
        # # update state
        # if i * dt_sim < t_brake_emergency:
        #     pass
        #     #x_history[i,:] = new_state_candidate #np.squeeze(F_sim @ np.expand_dims(x_history[i-1,:],1) + G_sim @ u_cacc)
        # else:
        #     #x_history[i,:] = np.squeeze(F_sim @ np.expand_dims(x_history[i-1,:],1) + G_sim @ u_cacc)
        #     # leader vehicle performs an emergency brake
        #     v_leader = x_history[i-1,2] - u_max*dt_sim
        #     if v_leader < -v_d: # you have reached 0 velocity
        #         v_leader = -v_d
            
        #     # update leader state
        #     new_state_candidate[2] = v_leader


        x_history[i,:] = new_state_candidate

        # evaluate ellipse value x^T P x
        ellipse_val[i] = np.expand_dims(x_history[i-1,:],0) @ P_opt @ np.expand_dims(x_history[i-1,:],1)

    if add_label:
        label_traj = 'trajectory attack-mitigation OFF'
    else:   
        label_traj = ''
    ax_d1d2.plot(x_history[:,0],x_history[:,1],label=label_traj,color='dodgerblue')
    ax_d1d2.legend()
    attack_mitigation = False
    scale = np.sqrt(m)
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

# add maximum velocity constraints
c3 = np.zeros((n, 1))
c3[2, 0] = -1
b3 = v_max - v_d
c4 = np.zeros((n, 1))
c4[3, 0] = -1
b4 = v_max - v_d
c5 = np.zeros((n, 1))
c5[4, 0] = -1
b5 = v_max - v_d



Y,R_hat = evaluate_new_bounds_with_constraints(a,F,G,R,c1,c2,b1,b2)


# Results
print("R_hat =\n", R_hat)
print("inv(Y) =\n", np.linalg.inv(Y))


# what is the 
Y_inv = np.linalg.inv(Y)
Y_inv_2d = np.linalg.inv(Y[:2, :2])
plot_ellipse(Y_inv[:2,:2],m,ax_d1d2,edgecolor='coral', linewidth=2,facecolor='none',label = 'R_hat inverting Y full')
plot_ellipse(Y_inv_2d,m,ax_d1d2,edgecolor='k', linewidth=2, facecolor='none',label = 'R_hat inverting Y[:2,:2] ',linestyle='--')



# Now try re-solving the original problem with the new bounds
P_hat_resolved,prob = reachable_set_given_bounds(a,F,G,R_hat)
plot_ellipse(P_hat_resolved[:2,:2],m,ax_d1d2,edgecolor='brown', linewidth=2,facecolor='none',linestyle = '--',label = 'R_hat different gamma_hats re-evaluated')


# now simulate the system with the new bounds
# define new bounds
# extract diag entries of R_hat
u_max_controlled = np.array([np.sqrt(1/R_hat[0,0]),
                             np.sqrt(1/R_hat[1,1]),
                             np.sqrt(1/R_hat[2,2])])





#plot the bounds
# draw vertcal red line for b1
ax_d1d2.axvline(x=b1, color='red', linestyle='--', label='b1')
#draw horizontal red line for b2
ax_d1d2.axhline(y=b2, color='red', linestyle='--', label='b2')

# add legend
ax_d1d2.legend()






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
        u_cacc = np.array([[-u_max_controlled[0]],[u_max_controlled[1]],[u_max_controlled[2]]])

        # # random input
        # u1= np.random.uniform(low=-u_max_cacc[0], high=u_max_cacc[0])
        # u2= np.random.uniform(low=-u_max_cacc[1], high=u_max_cacc[1])
        # u3= np.random.uniform(low=-u_max_cacc[2], high=u_max_cacc[2])
        # u_cacc = np.array([[u1],[u2],[u3]])

        
        # update state
        if i * dt_sim < t_brake_emergency:
            x_history[i,:] = np.squeeze(F_sim @ np.expand_dims(x_history[i-1,:],1) + G_sim @ u_cacc)
        else:
            x_history[i,:] = np.squeeze(F_sim @ np.expand_dims(x_history[i-1,:],1) + G_sim @ u_cacc)
            # leader vehicle performs an emergency brake
            v_leader = x_history[i-1,2] - u_max*dt_sim
            if v_leader < -v_d: # you have reached 0 velocity
                v_leader = -v_d
            
            # update leader state
            x_history[i,2] = v_leader




        # store control input
        u_history[i,:] = np.squeeze(u_cacc)

        # evaluate - log determinant of P
        ellipse_val[i] = np.expand_dims(x_history[i-1,:],0) @ P_hat_resolved @ np.expand_dims(x_history[i-1,:],1)

    if add_label:
        label_traj = 'trajectory attack-mitigation ON'
    else:   
        label_traj = ''
    ax_d1d2.plot(x_history[:,0],x_history[:,1],label=label_traj,color='orangered')
    ax_d1d2.legend()
    attack_mitigation = True
    scale = np.sqrt(m)
    plot_trajectories_timeseries(axs,time,x_history,ellipse_val,scale,add_label,attack_mitigation)
    add_label = False




plt.tight_layout()
plt.show()


