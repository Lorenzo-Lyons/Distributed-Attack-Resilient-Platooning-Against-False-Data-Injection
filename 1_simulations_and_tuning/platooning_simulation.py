import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from classes_definintion import platooning_problem_parameters,Vehicle_model,set_scenario_parameters,generate_color_gradient,generate_color_1st_last_gray
from tqdm import tqdm




# choose scenario to simulate
# 1 = strady state behaviour linear only (staring from far away in the p_rel-v_rel plane)
# 2 = linear controller only, leader oscillates around equilibrium point
# 3 = linear + u_ff, leader oscillates around equilibrium point  u_ff = u_i
# 4 = linear + u_ff, leader oscillates around equilibrium point  u_ff = u_i +kh(v_(i+1)-vD)
# 5 = linear + u_ff, fake data injection attack sinusoidal wave leader-vehicle1
# 6 = linear + u_ff, fake data injection attack extremely high acceleration
# 7 = linear + u_ff, fake data injection attack extremely high acceleration and leader performs emergency brake
# 8 = MPC (like scenario 1)
# 9 = MPC (like scenario 3-4)
# 10 = MPC affected by FDI and leader brakes (like scenario 7)
# 11 = linear controller with emergency brake
# 12 = MPC with emergency brake (no attack)



# our method with sinusoidal reference
# scenario = 3
# dt_int = 0.1 #[s]

# DMPC with sinusoidal reference
# scenario = 9
# dt_int = 0.1 #[s]


# our method with FDI attack and emergency brake
# scenario = 7
# dt_int = 0.0001 #[s]

# # DMPC with FDI attack and emergency brake
# scenario = 10
# dt_int = 0.1 #[s]









scenario = 12
dt_int = 0.1 #[s]








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

simulation_time = 50  #[s]  200
n_follower_vehicles = 9 # number of follower vehicles (so the leader is vehicle 0)



# this is needed because of the discrete time implementation, in real life it would still represent the 
# extra safety margine needed to account for delays in detecting the emergency situation.
max_u_change = (u_max-u_min) # in real life you know there is some actuation dynamics so you can be less conservative
stopping_time = v_max/(-u_min)
extra_safety_margin = 0 #dt_int * max_u_change * stopping_time # note that this goes to 0 if dt_int --> 0


# Example usage
# start_color = "#000000"
# end_color = "#71b6cb" 
#colors = generate_color_gradient(n_follower_vehicles + 1, start_color, end_color)

leader_color = "#ea7317"
start_color = "#57b8ff"
end_color = "#3d348b"
colors = generate_color_1st_last_gray(n_follower_vehicles, start_color, end_color)
# append leader color in front
colors = [leader_color] + colors
















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
        #set up initial state of first follower vehicle (may be different from others in some scenarios)
        x0 = p_rel_1-d -extra_safety_margin+vehicle_states[kk-1][0,4] 
        v0 = vehicle_states[kk-1][0,3]+v_rel_follower_1
    else:
        #set up initial state of first follower vehicle
        x0 = p_rel_others-d-extra_safety_margin+vehicle_states[kk-1][0,4] 
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
    v_leader_reference = np.zeros((sim_steps,MPC_N+1))
    x_leader_reference = np.zeros((sim_steps,MPC_N+1))
    # assign initial state
    v_leader_reference[0,0] = vehicle_vec[0].v
    x_leader_reference[0,0] = vehicle_vec[0].x

    for t0 in range(0,sim_steps):
        for stage in range(0,MPC_N+1):
            u_leader_reference = leader_acc_fun(t0, stage*dt_int) # pass in both current time and open loop time
            v_ref_candidate = v_leader_reference[t0,stage] + u_leader_reference * dt_int

            if v_ref_candidate > v_max:
                v_leader_reference[stage+1] = v_max
            elif v_ref_candidate < 0:
                v_leader_reference[stage+1] = 0
            else:
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
            vehicle_states[kk][t,1] = vehicle_vec[kk].x - (vehicle_vec[kk-1].x - extra_safety_margin) + vehicle_vec[kk].d 
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
        # compute first iteration assumed trajectory
        v_open_loop_0 = np.ones(MPC_N+1) * vehicle_vec[0].v
        x_open_loop_0 = vehicle_vec[0].x - vehicle_vec[0].v * dt_int + vehicle_vec[0].v * dt_int * np.arange(0,MPC_N+1)

        x_ref_i = x_leader_reference[t:t+MPC_N+1] # take reference state
        x_open_loop_prev = vehicle_vec[0].x_open_loop if t > 0 else x_open_loop_0 # assign previous iteration open loop trajectory
        v_open_loop_prev = vehicle_vec[0].v_open_loop if t > 0 else v_open_loop_0 # assign last stage velocity of previous iteration
        x_current = np.array([vehicle_vec[0].v, vehicle_vec[0].x])


        


        # set up solver for current iteration
        MPC_solver_t,x_assumed_open_loop_i = DMPC_obj.set_up_sovler_iteration(  MPC_solver,\
                                                                                MPC_N,\
                                                                                x_ref_i,\
                                                                                x_open_loop_prev,\
                                                                                v_open_loop_prev[-1],\
                                                                                dt_int,\
                                                                                u_min,\
                                                                                u_max,\
                                                                                x_current)
        
        # provide an initial guess for the solver
        x_guess = x_open_loop_prev
        v_guess = v_open_loop_prev
        u_guess = vehicle_vec[0].u_open_loop if t > 0 else np.zeros(MPC_N) # assign last stage velocity of previous iteration

        MPC_solver_t = DMPC_obj.set_initial_guess(MPC_solver_t,v_guess , x_guess, u_guess)
        
        # solve the optimization problem
        # Solve MPC problem
        u_open_loop_leader,v_open_loop_leader,x_open_loop_leader = DMPC_obj.solve_mpc(MPC_solver_t,MPC_N)

        # store assumed leader trajectory
        vehicle_vec[0].x_open_loop = x_open_loop_leader
        vehicle_vec[0].v_open_loop = v_open_loop_leader
        vehicle_vec[0].u_open_loop = u_open_loop_leader

        # assign control input to leader
        vehicle_vec[0].u = u_open_loop_leader[0]
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
                # alpha_controller = 0.95 #0.95 # lowering this number ensures a bit of margin before triggering the emergency brake manoeuvre 
                alpha_controller = 1 #1 - extra_safety_margin/d
                
                u_ff_max = k * (d * alpha_controller + h*(vehicle_vec[kk].v - v_d)) 
                max_p_rel =  d - c/k*(vehicle_states[kk][t,0]) # vehicle_states[kk][t,0] = relative velocity

                # account for discrete time step
                # max_v_rel_change = (u_max-u_min) * dt_int
                # max_p_rel =  d - c/k*(vehicle_states[kk][t,0] + max_v_rel_change) # vehicle_states[kk][t,0] = relative velocity
                p_rel_2_check = vehicle_states[kk][t,1] #+ (vehicle_states[kk][t,0])*dt_int

                if p_rel_2_check >= max_p_rel: # vehicle_states[kk][t,1]
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

            if attack_function == []: # reliable communication
                x_ref_i = vehicle_vec[kk-1].x_open_loop - vehicle_vec[kk].d # reference trajectory is the lead vehicle -d
            
            if kk == 1: # first follower vehicle
                if attack_function == []: # reliable communication
                    x_ref_i = vehicle_vec[kk-1].x_open_loop - vehicle_vec[kk].d # reference trajectory is the lead vehicle -d

                else: # attack 
                    v_attack = vehicle_vec[kk-1].v # initialize attack velocity on ral value
                    x_attack = vehicle_vec[kk-1].x # initialize attack position on real value
                    x_ref_i = np.zeros(MPC_N+1)
                    x_ref_i[0] = x_attack - vehicle_vec[kk].d # assign first value

                    for jj in range(1,MPC_N+1):
                        v_attack += attack_function(dt_int*(t+jj)) * dt_int # fake acceleration signal
                        x_attack += v_attack * dt_int
                        x_ref_i[jj] = x_attack - vehicle_vec[kk].d

                    #x_ref_i = vehicle_vec[kk-1].x_open_loop - vehicle_vec[kk].d # reference trajectory is the lead vehicle -d

            else: # other vehicles are not affected by the attack
                x_ref_i = vehicle_vec[kk-1].x_open_loop - vehicle_vec[kk].d # reference trajectory is the lead vehicle -d
            
            x_open_loop_0 = vehicle_vec[kk].x - vehicle_vec[kk].v * dt_int + vehicle_vec[kk].v * dt_int * np.arange(0,MPC_N+1)
            x_open_loop_prev = vehicle_vec[kk].x_open_loop if t > 0 else x_open_loop_0 # assign previous iteration open loop trajectory
            v_open_loop_prev_N = vehicle_vec[kk].v_open_loop_N if t > 0 else vehicle_vec[kk].v # assign last stage velocity of previous iteration
            
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
    if kk==0 or kk == 1 or kk==n_follower_vehicles: # 
        alpha = 1
    else:
        alpha = 0.3

    plt.plot(t_vec,vehicle_states[kk][:,3],label='vehicle ' + str(int(kk)), color=colors[kk],alpha=alpha) 
#if use_MPC:
    #plt.plot(np.array(range(sim_steps+MPC_N+1)) * dt_int,v_leader_reference,color='black',linestyle='--',label='leader reference')
plt.plot(t_vec,np.ones(len(t_vec))*v_d,linestyle='--',color='gray',label='v_d')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('v [m/s]')
plt.title('Absolute Velocity')


# plot relative velocity leader and follower 
t_vec = np.array(range(sim_steps)) * dt_int
plt.figure()
for kk in range(1,n_follower_vehicles+1):
    if kk==0 or kk == 1 : # or kk==n_follower_vehicles
        alpha = 1
    else:
        alpha = 0.3
    plt.plot(t_vec,vehicle_states[kk][:,0],label='vehicle ' + str(int(kk+1)), color=colors[kk],alpha=alpha)

plt.legend()
plt.xlabel('time [s]')
plt.ylabel('v [m/s]')
plt.title('Relative Velocity')



# plot relative position leader vs follower 
plt.figure()

if scenario == 9 or scenario == 3:
    y_lims = [-6.2,-5.5]
else:
    y_lims = [-d-extra_safety_margin-1,1]

x_lim = [0,simulation_time]
t_vec = np.array(range(sim_steps)) * dt_int

if use_MPC == False and use_ff==True:
    plt.plot(t_vec,np.ones(len(t_vec))*(d*(-1+alpha_controller)-extra_safety_margin),linestyle='--',color='gray',label='d(1-alpha)')
plt.plot(t_vec,np.ones(len(t_vec))*-d-extra_safety_margin,linestyle='--',color='gray',label='d',zorder=20)

# add collision zone box
if scenario == 9 or scenario == 3:
    pass
else:
    plt.plot(t_vec,np.zeros(len(t_vec)),linestyle='-',color="#a40606")
    plt.fill_between(t_vec, y1=0, y2=y_lims[1], color='#a40606', alpha=0.3, label='collision with predecessor')


for kk in range(1,n_follower_vehicles+1):
    if kk==0 or kk == 1 or kk==n_follower_vehicles: # 
        alpha = 1
    else:
        alpha = 0.3
    # determine legend entry
    if kk == 1:
        label = 'first follower'
    elif kk == 2:
        label = 'other followers'
    elif kk == n_follower_vehicles:
        label = 'last follower'

    if kk==1 or kk==2 or kk==n_follower_vehicles:
        plt.plot( t_vec,-(vehicle_states[kk-1][:,4] - vehicle_states[kk][:,4]),label=label, color=colors[kk],alpha=alpha) #'x_rel ' + str(int(kk+1))
    else:
        plt.plot( t_vec,-(vehicle_states[kk-1][:,4] - vehicle_states[kk][:,4]), color=colors[kk],alpha=alpha) #'x_rel ' + str(int(kk+1))

    





if attack_function != []:
    # Plot the "x" marker at (x_marker, 0)
    plt.plot(25, y_lims[0]+0.15, 'v', color='red', markersize=10, label='emergency brake') # check with leader function that it starts at 25s
plt.legend()
plt.ylim(y_lims)
plt.xlim(x_lim)
plt.xlabel('time [s]')
plt.ylabel('distance [m]')
plt.title('Relative position')


#plot acceleration
plt.figure()
plt.plot(t_vec,u_min*np.ones(len(u_vector_leader)),linestyle='--',color='gray',label='u_min')
plt.plot(t_vec,u_max*np.ones(len(u_vector_leader)),linestyle='--',color='gray',label='u_max')
plt.plot(t_vec,u_vector_leader,label='u leader', color=colors[0],alpha = 1)
for kk in range(n_follower_vehicles):
    if kk == 0: # kk==n_follower_vehicles-1 or 
        alpha = 1
    else:
        alpha = 0.3
    plt.plot(t_vec,u_total_followers[:,kk],label='vehicle'+str(kk+2), color=colors[kk+1],alpha=alpha)
plt.ylabel('acceleration [m/s^2]')
plt.xlabel('time [s]')
plt.legend()
plt.title('Acceleration')





plt.show()



