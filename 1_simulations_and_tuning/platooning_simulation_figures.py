import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from classes_definintion import platooning_problem_parameters,Vehicle_model,set_scenario_parameters,generate_color_gradient,generate_color_1st_last_gray
from tqdm import tqdm

from matplotlib import rc
font = {'family' : 'serif',
        #'serif': ['Times New Roman'],
        'size'   : 16}

rc('font', **font)


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





our_method_color = "#00A6D6"
DMPC_20_color = "#6CC24A"
DMPC_40_color = "#009B77"
Kafash_color = "#6F1D77"

methods_colors = [DMPC_40_color,DMPC_20_color,Kafash_color,our_method_color]

#colors = [leader_color,color_1,color_2] 



# set current directory where the script is located
import os
# set current directory as folder where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# get list of all folders in the folder "simualtion_data"
import re

# Path to simulation_data directory
simulation_data_dir = os.path.join(script_dir, 'simulation_data')

# List only folders
folders = [f for f in os.listdir(simulation_data_dir)
           if os.path.isdir(os.path.join(simulation_data_dir, f))]

# Extract leading number and sort
sorted_folders = sorted(folders, key=lambda name: int(re.match(r'^\d+', name).group()))

# Remove the leading number and return cleaned names
cleaned_folders = [re.sub(r'^\d+', '', name).lstrip() for name in sorted_folders]









# ----------------------------
# plotting simulation results
# ----------------------------



# plot absolute velocity 
plt.figure()
for folder in folders:
    folder_path = os.path.join(simulation_data_dir, folder)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for file in files:
        if file.endswith('.npy'):
            var_name = os.path.splitext(file)[0]  # Remove .npy extension
            file_path = os.path.join(folder_path, file)
            globals()[var_name] = np.load(file_path)

    for i in range(v_sim.shape[1]):
        plt.plot(time_vec,v_sim[:-1,i],label='vehicle ' + str(int(i)), color=methods_colors[i]) 


plt.plot(time_vec,np.ones(len(time_vec))*v_d,linestyle='--',color='gray',label='v_d')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('v [m/s]')
plt.title('Absolute Velocity')





time_to_attack = 1
time_to_brake = 11




fig_pos, ax_pos = plt.subplots(nrows=1, ncols=2, figsize=(16, 3.1))
fig_pos.subplots_adjust(
top=0.98,
bottom=0.2,
left=0.045,
right=0.81,
hspace=0.18,
wspace=0.135
)





y_lims = [-d-1,2.5]
x_lims = [0,time_vec[-1]]



# Assume 'folders' is already sorted and cleaned
for i, folder in enumerate(sorted_folders):
    folder_path = os.path.join(simulation_data_dir, folder)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for file in files:
        if file.endswith('.npy'):
            var_name = os.path.splitext(file)[0]  # Remove .npy extension
            file_path = os.path.join(folder_path, file)
            globals()[var_name] = np.load(file_path)

    # Format label: bold if last folder
    if i == len(folders) - 1:
        label = r'$\mathbf{' + cleaned_folders[i] + '}$'
        linewidth = 3
    else:
        label = cleaned_folders[i]
        linewidth = 2

    ax_pos[0].plot(time_vec, x_sim[:-1, 1] - x_sim[:-1, 0], linewidth=linewidth, label=label,color = methods_colors[i],zorder=20)
    ax_pos[1].plot(time_vec, x_sim[:-1, 2] - x_sim[:-1, 1], linewidth=linewidth, label=label,color = methods_colors[i],zorder=20)


ax_pos[0].plot(time_vec,np.ones(len(time_vec))*-d,linestyle='--',color='gray',label='d',linewidth=2,alpha = 0.5)
ax_pos[0].fill_between(time_vec, y1=0, y2=y_lims[1], color='#EC6842', alpha=0.3, label='collision')
ax_pos[0].axvline(x=time_to_attack, color='orange', linestyle='--',label='FDI attack',linewidth=2)
ax_pos[0].axvline(x=time_to_brake, color='orangered', linestyle='--',label='emergency brake',linewidth=2)

ax_pos[1].plot(time_vec,np.ones(len(time_vec))*-d,linestyle='--',color='gray',label='d',linewidth=2,alpha = 0.5)
ax_pos[1].fill_between(time_vec, y1=0, y2=y_lims[1], color='#EC6842', alpha=0.3, label='collision')
ax_pos[1].axvline(x=time_to_attack, color='orange', linestyle='--',label='FDI attack',linewidth=2)
ax_pos[1].axvline(x=time_to_brake, color='orangered', linestyle='--',label='emergency brake',linewidth=2)

ax_pos[0].set_ylim(y_lims)
ax_pos[0].set_xlim(x_lims)
ax_pos[0].set_yticks([0, -3, -6])
ax_pos[0].set_yticklabels(['0', '3', '6'])  # Remove minus signs from labels
ax_pos[0].set_xlabel('time [s]')
ax_pos[0].set_ylabel(r'$d_1$ [m]')

ax_pos[1].set_ylim(y_lims)
ax_pos[1].set_xlim(x_lims)
ax_pos[1].set_yticks([0, -3, -6])
ax_pos[1].set_yticklabels(['0', '3', '6'])  # Remove minus signs from labels
ax_pos[1].set_xlabel('time [s]')
ax_pos[1].set_ylabel(r'$d_2$ [m]')

ax_pos[1].legend(bbox_to_anchor=(1.01, 1.05))



plt.show()




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



