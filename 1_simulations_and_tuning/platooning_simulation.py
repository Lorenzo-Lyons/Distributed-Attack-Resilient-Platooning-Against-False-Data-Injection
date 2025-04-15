import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from classes_definintion import platooning_problem_parameters,Vehicle_model,set_scenario_parameters,generate_color_gradient,generate_color_1st_last_gray
from tqdm import tqdm

from matplotlib import rc
font = {'family' : 'serif',
        #'serif': ['Times New Roman'],
        'size'   : 20}

rc('font', **font)


# choose scenario to simulate
# scenario = 1 # strady state behaviour linear only (staring from far away in the p_rel-v_rel plane)
# scenario = 2 # linear controller only, leader oscillates around equilibrium point
# scenario = 3 # linear + u_ff, leader oscillates around equilibrium point  u_ff = u_i
# scenario = 4 # linear + u_ff, leader oscillates around equilibrium point  u_ff = u_i +kh(v_(i+1)-vD)
# scenario = 5 # linear + u_ff, fake data injection attack sinusoidal wave leader-vehicle1
# scenario = 6 # linear + u_ff, fake data injection attack extremely high acceleration
# scenario = 7 # linear + u_ff, fake data injection attack extremely high acceleration and leader performs emergency brake
# scenario = 8 # MPC (like scenario 1)
# scenario = 9 # MPC (like scenario 3-4)
# scenario = 10 # MPC affected by FDI and leader brakes (like scenario 7)
# scenario = 11 # linear controller with emergency brake
# scenario = 12 # MPC with emergency brake (no attack)



dt_int = 0.1 #[s]
simulation_time = 50




# our method with sinusoidal reference
# scenario = 3


# DMPC with sinusoidal reference
# scenario = 9


#our method with FDI attack and emergency brake
# scenario = 7
# dt_int = 0.0001 #[s] # must increase resolution


# # DMPC with FDI attack and emergency brake
scenario = 10


# # Baseline linear controller with emergency brake
# scenario = 14


# Our linear controller with emergency brake
# scenario = 11







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


# string stable but not safe controller taken from the paper we got the linear controller from
# NO! need to implement a different linear controller that uses constant time headway
# u = k * (x_i-x_i+1-h*v+1+1) + c (v_rel) 
k_baseline = 4.5
h_baseline = d/(v_d)
c_baseline = 4.5 
#c_baseline = 2*np.sqrt(k_baseline) - h_baseline*k_baseline

print('c_baseline:',c_baseline)

# plot the transfer function of the linear controller
# --- check that the system is critically damped --- 
# this should be the case by design but just to be sure we can plot the bode diagram of the transfer function
zero=k_baseline/c_baseline
delta= np.sqrt((c_baseline+k_baseline*h_baseline)**2-4*k_baseline)
pole1=(c_baseline+k_baseline*h_baseline)/(2)-delta/2
pole2=(c_baseline+k_baseline*h_baseline)/(2)+delta/2
print('pole1:',pole1)
print('pole2:',pole2)
print('zero:',zero)

from scipy import signal
import matplotlib.pyplot as plt

sys = signal.TransferFunction([c_baseline, k_baseline], [1, c_baseline+k_baseline*h_baseline,k_baseline])
w, mag, phase = signal.bode(sys)

plt.figure()
plt.semilogx(w, mag)    # Bode magnitude plot
plt.title(' p_(i+1)/p_i Transfer function : max =' + str(mag.max()))
#add ticks on x axis of poles and zeros
plt.axvline(x=zero, color='r', linestyle='--',label='zero')
plt.axvline(x=pole1, color='g', linestyle='--',label='pole1')
plt.axvline(x=pole2, color='darkgreen', linestyle='--',label='pole2')
plt.legend()





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
use_MPC,use_baseline_linear,time_to_brake, time_to_attack = set_scenario_parameters(scenario,d,v_d,c,k,h,v_max,u_min,u_max)




#instantiate vehicle classes
sim_steps = int(np.round(simulation_time/dt_int))+1

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

    u_open_loop_first_follower = np.zeros((sim_steps,MPC_N))


    for t0 in range(0,sim_steps):
        if t0==0:
            # assign initial state
            v_leader_reference[t0,0] = vehicle_vec[0].v
            x_leader_reference[t0,0] = vehicle_vec[0].x
        else:
            v_leader_reference[t0,0] = v_leader_reference[t0-1,1]
            x_leader_reference[t0,0] = x_leader_reference[t0-1,1]

        for stage in range(1,MPC_N+1):
            u_leader_reference = leader_acc_fun(t0*dt_int, (t0 + stage)*dt_int) # pass in both current time and open loop time
            v_ref_candidate = v_leader_reference[t0,stage-1] + u_leader_reference * dt_int

            if v_ref_candidate > v_max:
                v_leader_reference[t0,stage] = v_max
            elif v_ref_candidate < 0:
                v_leader_reference[t0,stage] = 0
            else:
                v_leader_reference[t0,stage] = v_ref_candidate

            x_leader_reference[t0,stage] = x_leader_reference[t0,stage-1] + v_leader_reference[t0,stage-1] * dt_int













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
        vehicle_vec[0].u = leader_acc_fun(t*dt_int,0)
        # check if the leader has reached max velocity
        if vehicle_vec[0].v == v_max:
            vehicle_vec[0].u = 0
        elif vehicle_vec[0].v == 0:
            vehicle_vec[0].u = 0

        u_vector_leader[t] = vehicle_vec[0].u

    # produce leader open loop prediction if using MPC
    else: # use MPC
        # compuute reference trajectory and assumed trajectory
        # compute first iteration assumed trajectory
        v_open_loop_0 = np.ones(MPC_N+1) * vehicle_vec[0].v
        x_open_loop_0 = vehicle_vec[0].x - vehicle_vec[0].v * dt_int + vehicle_vec[0].v * dt_int * np.arange(0,MPC_N+1)

        #x_ref_i = x_leader_reference[t:t+MPC_N+1] # take reference state
        x_ref_i = x_leader_reference[t,:] # take reference state

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
        vehicle_vec[0].u_open_loop = u_open_loop_leader
        vehicle_vec[0].v_open_loop = v_open_loop_leader
        vehicle_vec[0].x_open_loop = x_open_loop_leader

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
                # alpha_controller = 0.97 #0.97 # lowering this number ensures a bit of margin before triggering the emergency brake manoeuvre 
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
            if use_baseline_linear:
                #u = k * (x_i-x_i+1-h*v+1+1) + c (v_rel) 
                u_lin = - k_baseline * (vehicle_vec[kk].x - vehicle_vec[kk-1].x + h_baseline*vehicle_vec[kk-1].v) - c_baseline*(vehicle_vec[kk].v-vehicle_vec[kk-1].v) 
            else:
                u_lin = - k *(vehicle_states[kk][t,1]) - c*(vehicle_states[kk][t,0]) - k*h*(vehicle_vec[kk].v-v_d)

            # compute total acceleration:
            u_no_sat = u_lin + u_ff

        else: # using MPC
            # compuute reference trajectory and assumed trajectory

            # if attack_function == []: # reliable communication
            #     x_ref_i = vehicle_vec[kk-1].x_open_loop - vehicle_vec[kk].d # reference trajectory is the lead vehicle -d
            
            if kk == 1: # first follower vehicle
                if attack_function == [] or t*dt_int<time_to_attack: # reliable communication
                    x_ref_i = vehicle_vec[kk-1].x_open_loop - vehicle_vec[kk].d # reference trajectory is the lead vehicle -d

                else: # attack 
                    v_attack = vehicle_vec[kk-1].v # initialize attack velocity on ral value
                    x_attack = vehicle_vec[kk-1].x # initialize attack position on real value
                    x_ref_i = np.zeros(MPC_N+1)
                    #x_ref_i[0] = x_attack # assign first value

                    for jj in range(0,MPC_N+1):
                        x_ref_i[jj] = x_attack
                        x_attack += v_attack * dt_int
                        v_attack += attack_function(dt_int*(t+jj)) * dt_int # fake acceleration signal
                        
                        

                    # shift back by d
                    x_ref_i = x_ref_i - vehicle_vec[kk].d

            else: # other vehicles are not affected by the attack
                x_ref_i = vehicle_vec[kk-1].x_open_loop - vehicle_vec[kk].d # reference trajectory is the lead vehicle -d
            
            x_open_loop_0 = vehicle_vec[kk].x - vehicle_vec[kk].v * dt_int + vehicle_vec[kk].v * dt_int * np.arange(0,MPC_N+1)
            x_open_loop_prev = vehicle_vec[kk].x_open_loop if t > 0 else x_open_loop_0 # assign previous iteration open loop trajectory
            v_open_loop_prev_N = vehicle_vec[kk].v_open_loop[-1] if t > 0 else vehicle_vec[kk].v # assign last stage velocity of previous iteration
            
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
            
            # if t>0: # warm start
            #     MPC_solver_t =  DMPC_obj.set_initial_guess(MPC_solver_t, vehicle_vec[kk].v_open_loop , vehicle_vec[kk].x_open_loop, vehicle_vec[kk].u_open_loop)

            # solve the optimization problem
            # Solve MPC problem
            u_open_loop,v_open_loop,x_open_loop = DMPC_obj.solve_mpc(MPC_solver_t,MPC_N)

            # store open loop trajectory
            #vehicle_vec[kk].x_assumed = x_assumed_open_loop_i
            vehicle_vec[kk].u_open_loop = u_open_loop
            vehicle_vec[kk].v_open_loop = v_open_loop
            vehicle_vec[kk].x_open_loop = x_open_loop

            if kk == 1:
                u_open_loop_first_follower[t,:] = u_open_loop
            
            

            u_no_sat = u_open_loop[0] # select action to apply



        
        # apply saturation limits 
        if u_no_sat > u_max:
            u = u_max
        elif u_no_sat < u_min:
            u = u_min
        else:
            u = u_no_sat

        # check if the vehicle has reached max velocity
        if vehicle_vec[kk].v == v_max and u > 0:
            u = 0
        elif vehicle_vec[kk].v == 0 and u < 0:
            u = 0


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
if use_MPC:
    plt.plot(np.array(range(sim_steps)) * dt_int,v_leader_reference[:,0],color='black',linestyle='--',label='leader reference')
    for i in range(sim_steps):
        plt.plot(np.arange(i,i+MPC_N+1) * dt_int,v_leader_reference[i,:],color='r',alpha=0.3)


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
# if scenario == 14:
#     fig_pos, (ax_pos, ax_u) = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
#     fig_pos.subplots_adjust(
#     top=0.955,
#     bottom=0.09,
#     left=0.055,
#     right=0.78,
#     hspace=0.405,
#     wspace=0.2
#     )
# else:
#     fig_pos, ax_pos = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
#     fig_pos.subplots_adjust(
#     top=0.88,
#     bottom=0.18,
#     left=0.055,
#     right=0.76,
#     hspace=0.2,
#     wspace=0.2
#     )
#     fig_u, ax_u = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))



fig_pos, ax_pos = plt.subplots(nrows=1, ncols=1, figsize=(16, 3.1))
fig_pos.subplots_adjust(
top=0.98,
bottom=0.235,
left=0.055,
right=0.76,
hspace=0.2,
wspace=0.2
)
fig_u, ax_u = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))




if scenario == 9 or scenario == 3:
    y_lims = [-6.2,-5.5]
else:
    y_lims = [-d-extra_safety_margin-1,2]


x_lim = [0,simulation_time]
t_vec = np.array(range(sim_steps)) * dt_int


# if use_MPC == False and use_ff==True:
#     ax_pos.plot(t_vec,np.ones(len(t_vec))*(d*(-1+alpha_controller)-extra_safety_margin),linestyle='--',color='gray',label='d(1-alpha)',linewidth=3)
ax_pos.plot(t_vec,np.ones(len(t_vec))*-d-extra_safety_margin,linestyle='--',color='gray',label='d',zorder=20,linewidth=3,alpha = 0.5)

# add collision zone box
if scenario == 9 or scenario == 3:
    pass
else:
    ax_pos.plot(t_vec,np.zeros(len(t_vec)),linestyle='-',color="#a40606",linewidth=3)
    ax_pos.fill_between(t_vec, y1=0, y2=y_lims[1], color='#a40606', alpha=0.3, label='collision')


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
        ax_pos.plot( t_vec,-(vehicle_states[kk-1][:,4] - vehicle_states[kk][:,4]),label=label, color=colors[kk],alpha=alpha,linewidth=3) #'x_rel ' + str(int(kk+1))
    else:
        ax_pos.plot( t_vec,-(vehicle_states[kk-1][:,4] - vehicle_states[kk][:,4]), color=colors[kk],alpha=alpha,linewidth=3) #'x_rel ' + str(int(kk+1))







if attack_function != []:
    # Plot the "x" marker at (x_marker, 0)
    ax_pos.axvline(x=time_to_attack, color='orange', linestyle='--',label='FDI attack',linewidth=3)
if time_to_brake != 0:
    ax_pos.axvline(x=time_to_brake, color='orangered', linestyle='--',label='emergency brake',linewidth=3)



ax_pos.legend(bbox_to_anchor=(1.01, 1.05))

ax_pos.set_ylim(y_lims)
ax_pos.set_xlim(x_lim)
ax_pos.set_yticks([0,-3,-6])
ax_pos.set_xlabel('time [s]')
ax_pos.set_ylabel(r'$x_{i+1} - x_i$ [m]')
#ax_pos.set_title('Relative position')







#plot acceleration
# plt.figure()
ax_u.plot(t_vec,u_min*np.ones(len(u_vector_leader)),linestyle='--',color='gray',label='u limits',linewidth=3)
ax_u.plot(t_vec,u_max*np.ones(len(u_vector_leader)),linestyle='--',color='gray',linewidth=3)
ax_u.plot(t_vec,u_vector_leader,label='leader', color=colors[0],alpha = 1,linewidth=3)


# plot first follower open loop u
if use_MPC:
    for ii in range(sim_steps):
        ax_u.plot(np.arange(ii,ii+MPC_N)*dt_int,u_open_loop_first_follower[ii,:],color='r',alpha=0.3,zorder=20)
    


for kk in range(n_follower_vehicles):
    if kk==n_follower_vehicles-1 or kk == 0: #  
        alpha = 1
    else:
        alpha = 0.3
    # determine legend entry
    if kk == 0:
        label = 'first follower'
    elif kk == 1:
        label = 'other followers'
    elif kk == n_follower_vehicles-1:
        label = 'last follower'

    if kk==0 or kk==1 or kk==n_follower_vehicles-1:
        ax_u.plot(t_vec,u_total_followers[:,kk],label=label, color=colors[kk+1],alpha=alpha,linewidth=3) #'x_rel ' + str(int(kk+1))
    else:
        ax_u.plot(t_vec,u_total_followers[:,kk], color=colors[kk+1],alpha=alpha,linewidth=3) #'x_rel ' + str(int(kk+1))

    
    


ax_u.set_ylabel('acceleration [m/s^2]')
ax_u.set_xlabel('time [s]')
ax_u.set_xlim([0,simulation_time])
ax_u.legend(bbox_to_anchor=(1.01, 1.06))
ax_u.set_title('Acceleration')





plt.show()



