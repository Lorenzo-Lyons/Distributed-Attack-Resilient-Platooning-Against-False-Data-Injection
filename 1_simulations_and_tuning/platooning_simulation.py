import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from classes_definintion import platooning_problem_parameters,Vehicle_model,set_scenario_parameters
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
#attack_strat_time = 50 #[s] if the scenario includes an attack set initial attack time
dt_int = 0.01 #[s]
n_follower_vehicles = 3 # number of follower vehicles (so the leader is vehicle 0)
colors = ["71b6cb","00a6d6","5a7b86","000000"] # colors of lines to plot


# chose scenario to simulate
# 1 = strady state behaviour linear only (staring from far away in the p_rel-v_rel plane)
# 2 = linear controller only, leader oscillates around equilibrium point
# 3 = linear + u_ff, leader oscillates around equilibrium point  u_ff = u_i
# 4 = linear + u_ff, leader oscillates around equilibrium point  u_ff = u_i +kh(v_(i+1)-vD)
# 5 = linear + u_ff, fake data injection attack sinusoidal wave leader-vehicle1
# 6 = linear + u_ff, fake data injection attack extremely high acceleration
# 7 = linear + u_ff, fake data injection attack extremely high acceleration and leader performs emergency brake



scenario = 7


# select scenario- i.e. leader behavior, using mpc, sliding mode, ecc
v_rel_follower_1,p_rel_1,v_rel_follower_others,p_rel_others,\
x0_leader,v0_leader,\
leader_acc_fun,use_ff,attack_function = set_scenario_parameters(scenario,d,v_d,c,k,h,v_max,u_min,u_max)




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

    vehicle_vec = [*vehicle_vec, Vehicle_model(vehicle_number,x0,v0,leading_vehicle_number,controller_parameters,vehicle_parameters)]
    vehicle_i_state[0,3] = v0
    vehicle_i_state[0,4] = x0
    vehicle_states = (*vehicle_states, vehicle_i_state)




# set up vectors to store the simulation results
#-leader acceleration
u_vector_leader=np.zeros(sim_steps)


#-total acceleration of all vehicles with saturation
u_total_followers = np.zeros((sim_steps,n_follower_vehicles))

#- collect u_leader predictions for later plots
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
    vehicle_vec[0].u = leader_acc_fun(t*dt_int)
    u_vector_leader[t] = vehicle_vec[0].u


    # -- other vehicles --
    # for follower car use both MPC and linear controller
    # reset initial guess to 0 (otherwise you'll use the previous solution i.e. doing warmstart)
    #MPC_model_dynamic_constraint_obj.u.data = torch.zeros(N-1)

    for kk in range(1,n_follower_vehicles+1):

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
    ax.plot(subsampled_data[:, 0], subsampled_data[:, 1], subsampled_data[:, 2], alpha=0.7, color='#'+colors[kk-1])
    ax.scatter3D(subsampled_data[:, 0], subsampled_data[:, 1], subsampled_data[:, 2],label='vehicle '+str(int(kk+1)), color='#'+colors[kk-1])






# plot absolute velocity 
t_vec = np.array(range(sim_steps)) * dt_int
plt.figure()
for kk in range(n_follower_vehicles+1):
    plt.plot(t_vec,vehicle_states[kk][:,3],label='vehicle ' + str(int(kk)), color='#'+colors[kk-1]) #,color=line_color,alpha=transparency

plt.legend()
plt.xlabel('time [s]')
plt.ylabel('v [m/s]')
plt.title('Absolute Velocity')


# plot relative velocity leader and follower 
t_vec = np.array(range(sim_steps)) * dt_int
plt.figure()
for kk in range(1,n_follower_vehicles+1):
    plt.plot(t_vec,vehicle_states[kk][:,0],label='vehicle ' + str(int(kk+1)), color='#'+colors[kk])

plt.legend()
plt.xlabel('time [s]')
plt.ylabel('v [m/s]')
plt.title('Relative Velocity')



# plot relative position leader vs follower 
t_vec = np.array(range(sim_steps)) * dt_int
plt.figure()
for kk in range(1,n_follower_vehicles+1):
    plt.plot( t_vec,-(vehicle_states[kk-1][:,4] - vehicle_states[kk][:,4]),label='x_rel ' + str(int(kk+1)), color='#'+colors[kk])

plt.legend()
plt.xlabel('time [s]')
plt.ylabel('distance [m]')
plt.title('Relative position')


#plot acceleration
plt.figure()
plt.plot(t_vec,u_min*np.ones(len(u_vector_leader)),linestyle='--',color='gray',label='u_min')
plt.plot(t_vec,u_max*np.ones(len(u_vector_leader)),linestyle='--',color='gray',label='u_max')
plt.plot(t_vec,u_vector_leader,label='u leader', color='#'+colors[0])
for hh in range(n_follower_vehicles):
    plt.plot(t_vec,u_total_followers[:,hh],label='vehicle'+str(hh+2), color='#'+colors[hh+1])
plt.ylabel('acceleration [m/s^2]')
plt.xlabel('time [s]')
plt.legend()
plt.title('Acceleration')





plt.show()



