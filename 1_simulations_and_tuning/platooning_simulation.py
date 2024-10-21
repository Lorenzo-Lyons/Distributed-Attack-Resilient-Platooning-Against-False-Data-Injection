import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from classes_definintion import platooning_problem_parameters,Vehicle_model,set_scenario_parameters,generate_color_gradient
from tqdm import tqdm


# this figure shows the v_relative - p_relative v_absolute reachable set 

# load platooning parameters from class
platooning_problem_parameters_obj = platooning_problem_parameters(False) # leave value to False to simulate on full scale vehicles

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




#print out parameters
print('---------------------')
print('Problem parameters:')
print('v_d =',v_d, ' [m/s]')
print('v_max =',v_max, ' [m/s]')
print('u_max =', u_max,' [m/s^2]')
print('u_min =', u_min,' [m/s^2]')
print('intervehicle distance lin = ', d)

print('Controller gains:')
print('k =',k)
print('h =',h)
print('c =',c)
print('---------------------')






#-----------------------------
#----- set up simulation -----
#-----------------------------

simulation_time = 200  #[s]
dt_int = 0.1 #[s]
n_follower_vehicles = 2 # number of follower vehicles (so the leader is vehicle 0)


# Example usage
start_color = "#000000"
end_color = "#71b6cb" 

colors = generate_color_gradient(n_follower_vehicles + 1, start_color, end_color)














# chose scenario to simulate
# 1 = strady state behaviour linear only (staring from far away in the p_rel-v_rel plane)
# 2 = linear controller only, leader oscillates around equilibrium point
# 3 = linear + u_ff, leader oscillates around equilibrium point  u_ff = u_i
# 4 = linear + u_ff, leader oscillates around equilibrium point  u_ff = u_i +kh(v_(i+1)-vD)
# 5 = linear + u_ff, fake data injection attack sinusoidal wave leader-vehicle1
# 6 = linear + u_ff, fake data injection attack extremely high acceleration
# 7 = linear + u_ff, fake data injection attack extremely high acceleration and leader performs emergency brake
# 8 = MPC (like scenario 1)
# 9 = MPC (like scenario 3-4)



scenario = 9


# select scenario- i.e. leader behavior, using mpc, sliding mode, ecc
v_rel_follower_1,p_rel_1,v_rel_follower_others,p_rel_others,\
x0_leader,v0_leader,\
leader_acc_fun,use_ff,attack_function,\
use_MPC = set_scenario_parameters(scenario,d,v_d,c,k,h,v_max,u_min,u_max)




#instantiate vehicle classes
sim_steps = int(np.round(simulation_time/dt_int))

controller_parameters =[v_d,d,k,h,c] 
vehicle_parameters = [v_max,u_max,u_min]
vehicle_vec = []
vehicle_states = ()

for kk in range(n_follower_vehicles+1):
    #set up vehicle state recording
    vehicle_i_state = np.zeros((sim_steps,5))
    
    vehicle_number = kk
    leading_vehicle_number = kk-1

    if kk == 0: # set up leader
        v0 = v0_leader
        x0 = x0_leader

    elif kk==1:
        #set up initial state of first follower vehicle
        x0 = p_rel_1-d+vehicle_states[kk-1][0,4] 
        v0 = vehicle_states[kk-1][0,3]+v_rel_follower_1
    else:
        #set up initial state of first follower vehicle
        x0 = p_rel_others-d+vehicle_states[kk-1][0,4] 
        v0 = vehicle_states[kk-1][0,3]+v_rel_follower_others

    vehicle_vec = [*vehicle_vec, Vehicle_model(vehicle_number,x0,v0,leading_vehicle_number,controller_parameters,vehicle_parameters,use_MPC,dt_int)]
    vehicle_i_state[0,3] = v0
    vehicle_i_state[0,4] = x0
    vehicle_states = (*vehicle_states, vehicle_i_state)



# set up solver (only once because all vehicles will use the same MPC parameters)
if use_MPC:
    from classes_definintion import DMPC
    # default MPC parameters
    # self.dt_int = dt_int # Time step [s]
    MPC_N = 20  # Number of steps in the horizon
    Tf = dt_int * MPC_N  # Time horizon
    v_min = vehicle_vec[0].v_min

    DMPC_obj = DMPC()
    MPC_solver = DMPC_obj.setup_mpc_solver(Tf, MPC_N,u_max,u_min,v_max,v_min)

    # create reference for the leader to track
    v_leader_reference = np.zeros(sim_steps+MPC_N+1)
    x_leader_reference = np.zeros(sim_steps+MPC_N+1)
    # assign initial state
    v_leader_reference[0] = vehicle_vec[0].v
    x_leader_reference[0] = vehicle_vec[0].x

    for stage in range(0,sim_steps+MPC_N):
        u_leader_reference = leader_acc_fun(stage*dt_int)
        v_leader_reference[stage+1] = v_leader_reference[stage] + u_leader_reference * dt_int
        x_leader_reference[stage+1] = x_leader_reference[stage] + v_leader_reference[stage] * dt_int








# set up vectors to store the simulation results
#-leader acceleration
u_vector_leader=np.zeros(sim_steps)


#-total acceleration of all vehicles with saturation
u_total_followers = np.zeros((sim_steps,n_follower_vehicles))

#collect u_leader predictions for later plots
u_leader_predictions = ()



#run the simulation
for t in tqdm(range(sim_steps), desc ="Simulation progress"):

    #store current relative state
    for kk in range(n_follower_vehicles + 1):
        if kk > 0: # the current vehicle has a leading vehicle
            vehicle_states[kk][t,0] = vehicle_vec[kk].v - vehicle_vec[kk-1].v
            vehicle_states[kk][t,1] = vehicle_vec[kk].x - vehicle_vec[kk-1].x + vehicle_vec[kk].d
            vehicle_states[kk][t,2] = vehicle_vec[kk].v - v_d # to keep consisten with plot
            vehicle_states[kk][t,3] = vehicle_vec[kk].v
            vehicle_states[kk][t,4] = vehicle_vec[kk].x
        else: # leader has no vehicle in front of it
            vehicle_states[kk][t,0] = 0
            vehicle_states[kk][t,1] = 0
            vehicle_states[kk][t,2] = 0
            vehicle_states[kk][t,3] = vehicle_vec[kk].v
            vehicle_states[kk][t,4] = vehicle_vec[kk].x            


    # --- Evaluate control actions ---




    # -- leader --
    # store leader acceleration for plots
    if use_MPC == False:
        vehicle_vec[0].u = leader_acc_fun(t*dt_int)
        u_vector_leader[t] = vehicle_vec[0].u
    # produce leader open loop prediction if using MPC
    else: # use MPC

        # compuute reference trajectory and assumed trajectory
        x_ref_i = x_leader_reference[t:t+MPC_N+1] # take reference state
        x_open_loop_prev = vehicle_vec[0].x_open_loop if t > 0 else x_ref_i # assign previous iteration open loop trajectory
        v_open_loop_prev_N = vehicle_vec[0].v_open_loop_N if t > 0 else 0 # assign last stage velocity of previous iteration
        x_current = np.array([vehicle_vec[0].v, vehicle_vec[0].x])

        # set up solver for current iteration
        MPC_solver_t,x_assumed_open_loop_i = DMPC_obj.set_up_sovler_iteration(  MPC_solver,\
                                                                                MPC_N,\
                                                                                x_ref_i,\
                                                                                x_open_loop_prev,\
                                                                                v_open_loop_prev_N,\
                                                                                dt_int,\
                                                                                u_min,\
                                                                                u_max,\
                                                                                x_current)
        
        # solve the optimization problem
        # Solve MPC problem
        u_open_loop,v_open_loop,x_open_loop = DMPC_obj.solve_mpc(MPC_solver_t,MPC_N)

        # store assumed leader trajectory
        vehicle_vec[0].x_open_loop = x_open_loop
        vehicle_vec[0].v_open_loop_N = v_open_loop[-1]

        # assign control input to leader
        vehicle_vec[0].u = u_open_loop[0]
        u_vector_leader[t] = vehicle_vec[0].u
            



    # -- other vehicles --
    # for follower car use both MPC and linear controller
    # reset initial guess to 0 (otherwise you'll use the previous solution i.e. doing warmstart)
    #MPC_model_dynamic_constraint_obj.u.data = torch.zeros(N-1)

    for kk in range(1,n_follower_vehicles+1):
        # not using MPC
        if use_MPC == False:
            #using feed-forward action
            if use_ff:

                if attack_function == []:
                    # copy accelration from previous vehicle
                    u_ff = vehicle_vec[kk-1].u
                    if scenario == 4: # add external damping term compensation
                        u_ff = u_ff + k*h*(vehicle_vec[kk].v-v_d)
                else:
                    if kk ==1: #simulate an attack between leader and vehicle 1
                        u_ff = attack_function(t*dt_int,vehicle_vec[kk-1].u)
                    else:
                        u_ff = vehicle_vec[kk-1].u


                # apply constraints to u_ff
                alpha = 0.95 # lowering this number ensures a bit of margin before triggering the emergency brake manoeuvre 
                u_ff_max = k * (d * alpha + h*(vehicle_vec[kk].v - v_d)) 
                max_p_rel =  d - c/k*(vehicle_states[kk][t,0]) # vehicle_states[kk][t,0] = relative velocity

                if (vehicle_states[kk][t,1]) >= max_p_rel:
                    u_ff = 0
                elif u_ff > u_ff_max:
                    u_ff = u_ff_max

            else:
                u_ff = 0


            #compute linear controller action
            u_lin = -k*(vehicle_states[kk][t,1]) - c*(vehicle_states[kk][t,0]) - k*h*(vehicle_vec[kk].v-v_d)

            # compute total acceleration:
            u_no_sat = u_lin + u_ff

        else: # using MPC
            # compuute reference trajectory and assumed trajectory
            x_ref_i = vehicle_vec[kk-1].x_open_loop - vehicle_vec[kk].d # reference trajectory is the lead vehicle -d
            x_open_loop_prev = vehicle_vec[kk].x_open_loop if t > 0 else x_ref_i # assign previous iteration open loop trajectory
            v_open_loop_prev_N = vehicle_vec[kk].v_open_loop_N if t > 0 else 0 # assign last stage velocity of previous iteration
            x_current = np.array([vehicle_vec[kk].v, vehicle_vec[kk].x])

            # set up solver for current iteration
            MPC_solver_t,x_assumed_open_loop_i = DMPC_obj.set_up_sovler_iteration(  MPC_solver,\
                                                                                    MPC_N,\
                                                                                    x_ref_i,\
                                                                                    x_open_loop_prev,\
                                                                                    v_open_loop_prev_N,\
                                                                                    dt_int,\
                                                                                    u_min,\
                                                                                    u_max,\
                                                                                    x_current)
            
            # solve the optimization problem
            # Solve MPC problem
            u_open_loop,v_open_loop,x_open_loop = DMPC_obj.solve_mpc(MPC_solver_t,MPC_N)

            # store open loop trajectory
            #vehicle_vec[kk].x_assumed = x_assumed_open_loop_i
            vehicle_vec[kk].x_open_loop = x_open_loop
            vehicle_vec[kk].v_open_loop_N = v_open_loop[-1]

            u_no_sat = u_open_loop[0] # select action to apply



        
        # apply saturation limits 
        if u_no_sat > u_max:
            u = u_max
        elif u_no_sat < u_min:
            u = u_min
        else:
            u = u_no_sat

        # store result for later integration
        vehicle_vec[kk].u = u

        #store actual acceleration command for plots
        u_total_followers[t,kk-1] = u


    # integrate vehicles states
    for kk in range(n_follower_vehicles+1):
        vehicle_vec[kk].integrate_state(vehicle_vec[kk].u,dt_int)


    




# ----------------------------
# plotting simulation results
# ----------------------------

# plot trajectories of vehicles in the p_rel - v_rel - v_abs space
# determine vertices of the safe set
# determine axis limits
x_lim = [-v_max,v_max]
y_lim = [-1.2*(d),1.2*(d)]
z_lim = [-v_d,v_max-v_d]

O=[0,0,0]

SP_1 = [0,d,0-v_d]
SP_2 = [0,d,v_max-v_d]
SP_3 = [x_lim[1],d-c/k*x_lim[1],v_max-v_d]
SP_4 = [x_lim[1],d-c/k*x_lim[1],0-v_d]


CP_1 = [x_lim[0],d,0-v_d]
CP_2 = [x_lim[0],d,v_max-v_d]
CP_3 = [x_lim[1],d,v_max-v_d]
CP_4 = [x_lim[1],d,0-v_d]


#determining reachable set while braking
#While car in front is still braking:
#determine max duration of braking

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# vertices of a pyramid
v = np.array([O])

Collision_plane = np.array([[CP_1,CP_2,CP_3,CP_4]])
Sat_brake_plane = np.array([[SP_1,SP_2,SP_3,SP_4]])

ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
# generate list of sides' polygons of our pyramid

# plot collision plane and ulin saturation line
ax.add_collection3d(Poly3DCollection(Sat_brake_plane, facecolors='cyan', linewidths=1, edgecolors='gray', alpha=.25))
ax.add_collection3d(Poly3DCollection(Collision_plane, facecolors='darkorange', linewidths=1, edgecolors='gray', alpha=.25))


ax.set_xlabel(r'$\tilde{{v}}$' + '[m/s]')
ax.set_ylabel(r'$\tilde{{p}}$' +'[m]')
ax.set_zlabel(r'$v-v^D$' +'[m/s]')
ax.set_title(' State trajectories')


ax.axes.set_xlim3d(x_lim) 
ax.axes.set_ylim3d(y_lim) 
ax.axes.set_zlim3d(z_lim) 
# plot trajectories in the p_rel - v_rel - v_abs plane
for kk in range(1,n_follower_vehicles+1):
    # resample a subset of points not to overload the figure
    num_points = 200
    if len(vehicle_states[kk][:, 0]) > num_points:
        # Define the number of points you want to plot
        # Subsample the data to get equally spaced points
        indices = np.round(np.linspace(0, len(vehicle_states[kk]) - 1, num_points)).astype(int)
        subsampled_data = vehicle_states[kk][indices]
    else:
        subsampled_data = vehicle_states[kk]

    # Line plot
    ax.plot(subsampled_data[:, 0], subsampled_data[:, 1], subsampled_data[:, 2], alpha=0.7, color=colors[kk])
    ax.scatter3D(subsampled_data[:, 0], subsampled_data[:, 1], subsampled_data[:, 2],label='vehicle '+str(int(kk+1)), color=colors[kk-1])






# plot absolute velocity 
t_vec = np.array(range(sim_steps)) * dt_int
plt.figure()
for kk in range(n_follower_vehicles+1):
    plt.plot(t_vec,vehicle_states[kk][:,3],label='vehicle ' + str(int(kk)), color=colors[kk]) 

plt.legend()
plt.xlabel('time [s]')
plt.ylabel('v [m/s]')
plt.title('Absolute Velocity')


# plot relative velocity leader and follower 
t_vec = np.array(range(sim_steps)) * dt_int
plt.figure()
for kk in range(1,n_follower_vehicles+1):
    plt.plot(t_vec,vehicle_states[kk][:,0],label='vehicle ' + str(int(kk+1)), color=colors[kk])

plt.legend()
plt.xlabel('time [s]')
plt.ylabel('v [m/s]')
plt.title('Relative Velocity')



# plot relative position leader vs follower 
t_vec = np.array(range(sim_steps)) * dt_int
plt.figure()
plt.plot(t_vec,np.ones(len(t_vec))*-d,linestyle='--',color='gray',label='d')
for kk in range(1,n_follower_vehicles+1):
    plt.plot( t_vec,-(vehicle_states[kk-1][:,4] - vehicle_states[kk][:,4]),label='x_rel ' + str(int(kk+1)), color=colors[kk])

plt.legend()
plt.xlabel('time [s]')
plt.ylabel('distance [m]')
plt.title('Relative position')


#plot acceleration
plt.figure()
plt.plot(t_vec,u_min*np.ones(len(u_vector_leader)),linestyle='--',color='gray',label='u_min')
plt.plot(t_vec,u_max*np.ones(len(u_vector_leader)),linestyle='--',color='gray',label='u_max')
plt.plot(t_vec,u_vector_leader,label='u leader', color=colors[0])
for hh in range(n_follower_vehicles):
    plt.plot(t_vec,u_total_followers[:,hh],label='vehicle'+str(hh+2), color=colors[hh+1])
plt.ylabel('acceleration [m/s^2]')
plt.xlabel('time [s]')
plt.legend()
plt.title('Acceleration')





plt.show()



