from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# NOTE: set current working directory as "platooning_experiments_data_processing" before running this file







from matplotlib import rc
font = {'family' : 'serif',
        #'serif': ['Times New Roman'],
        'size'   : 20}

rc('font', **font)



# --- define fuction to add colored background and display legend ---
def change_following_false(bool_list):
    indexes_2_change = []
    for i in range(len(bool_list) - 1):
        if bool_list[i] and not bool_list[i + 1]:
            indexes_2_change = [*indexes_2_change,i+1]
    # change indexes
    bool_list[indexes_2_change] = True       
    return bool_list

def add_masks_and_legend(ax,mask_ff_action,mask_attack,mask_attack_detected,exp1,exp2,exp3,switch_1_5):
    # Add colored background for u_ff active, attack active, cahgen topology 
    # Extend the last True value to an additional step (this is needed for graphical resons)

    # y bottom limit
    if exp3:
        bot_lim = -0.2
    else:
        bot_lim = ax.get_ylim()[0]
    # y top limit
    if exp1:
        top_lim = ax.get_ylim()[1] 
        top_lim = top_lim + 0.1
    else:
        top_lim = ax.get_ylim()[1] 

    ax.set_ylim(bottom=bot_lim, top = top_lim)
    ax.set_xlim(left=0.0, right = time_vec[-1])

    if exp1:
        # uff active
        ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_ff_action, color='navy', alpha=0.1, label='u_ff active')
    elif exp2:
        # attack active
        ax.fill_between(time_vec, top_lim +1, bot_lim-1, where=mask_attack, color='orangered', alpha=0.1, label='attack active')
    elif exp3:
        ax.fill_between(time_vec, top_lim +1, bot_lim-1, where=change_following_false(mask_attack & ~mask_attack_detected), color='orangered', alpha=0.1, label='attack active',linewidth=0)
        ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=change_following_false(mask_attack_detected & ~mask_topology_change), color='palegreen', alpha=0.1, label='attack detected',linewidth=0)
        ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_topology_change, color='green', alpha=0.1, label='topology changed',linewidth=0)
    #set handles order
    handles, labels = ax.get_legend_handles_labels()
    # if exp1 or exp2:
    #     order = [1,2,3,0,4]
    # else:
    #     order = [1,2,3,4,0,5,6,7]
    order = list(range(len(handles)))
    if switch_1_5:
        if exp3:
            order[:5] = [*order[1:5],order[0]] # sitch first and last
        elif exp2:
            order[:4] = [*order[1:4],order[0]]

    # set number of columns
    if exp3:
        ncol = 2
    else:
        ncol  =1

    #plot_legend
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol =ncol, handlelength=1, loc=loc, bbox_to_anchor=anchor)











# --- begin plotting code ---
    
# set default plotting paramters to false values to False
exp1 = False
exp2 = False
exp3 = False

# platooning parameters
v_ref = 1.0
d_safety = 0.5

# file path  (uncomment both lines to select wich experiment to plot)

#experiment 1
exp1 = True
file_path = 'experiment_data/data_experiment_1_2024-03-22-16-30-52.csv' 

#experiment 2
#exp2 = True
#file_path = 'experiment_data/data_experiment_2_03_22_2024_17_48_58.csv'

#experiment 3
# exp3 = True
# file_path = 'experiment_data/data_experiment_3_2024-03-22-17-13-57.csv'

#demo
#file_path = 'platooning_ws/src/platooning_utilities/Data/platooning_full_demo/platooning_data_03_21_2024_16_23_22.csv'



loc = 'upper left'
anchor = (1,1.05)
right = 0.675



# read the raw data
df = pd.read_csv(file_path)

# extract values where safety is off
df = df[df['safety_value']==1]

# reset time
df['elapsed time sensors'] = df['elapsed time sensors'] - df['elapsed time sensors'].to_numpy()[0]


time_vec = df['elapsed time sensors'].to_numpy()

car_num = 4

colors = ["71b6cb","00a6d6","5a7b86","000000"] 



# define masks
mask = np.array(df['safety_value']) == 1
mask_ff_action = np.array(df['add_ff']) == True
mask_attack = np.array(df['attack']) == True
mask_attack_detected = np.array(df['alarm2']) > 1000 # initialize to all False
mask_topology_change = np.array(df['alarm2']) == 0


#produce attack detected array:
attack_detected = False
counter = 0
attack_timer = 5
alrm = np.array(df['alarm2'])
for i in range(len(mask)):
    if alrm[i]>0.5:
        counter = counter + 1
    else:
        counter = 0
    if counter > attack_timer:
        attack_detected = True
    mask_attack_detected[i] = attack_detected
    

# modify masks to avoid color overlap
#mask_ff_action = np.logical_and(mask_ff_action,np.logical_not(mask_attack_detected))
#mask_attack = np.logical_and(mask_attack,np.logical_not(mask_attack_detected))







# Define parameters for the Savitzky-Golay filter
window_length = 5  # Window length of the filter
polyorder = 1    # Polynomial order

# Create subplots for relative distances, absolute velocities, and relative velocities
fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.canvas.set_window_title('Data Plots')

if exp1:
    fig.subplots_adjust(
    top=0.965,
    bottom=0.07,
    left=0.070,
    right=0.875,
    hspace=0.2,
    wspace=0.2
    )
elif exp2:
    fig.subplots_adjust(
    top=0.965,
    bottom=0.07,
    left=0.070,
    right=0.855,
    hspace=0.2,
    wspace=0.2
    )
elif exp3:
    fig.subplots_adjust(
    top=0.965,
    bottom=0.07,
    left=0.070,
    right=0.75,
    hspace=0.2,
    wspace=0.2
    )






# --- relative distances ---
ax = axs[2]
ax.set_title('Relative distances')
ax.plot(time_vec, d_safety * np.ones(len(time_vec)), color="#c8b8db", linestyle='--', label='d', linewidth=3)

if exp3:
    i_init = 0
else:
    i_init = 1

for i in range(i_init,car_num):
    # Filter distance signal for graphical purposes
    df[f'dist{i+1}_filtered'] = savgol_filter(df[f'dist{i+1}'].to_numpy(), window_length, polyorder)
    # Plot filtered data
    ax.plot(time_vec, -df[f'dist{i+1}_filtered'].to_numpy(), label=fr'$\tilde{{p}}_{i+1}$',color='#'+colors[i], linewidth=3)

add_masks_and_legend(ax,mask_ff_action,mask_attack,mask_attack_detected,exp1,exp2,exp3,True)
ax.set_ylabel('[m]')





# ---  absolute velocities ---
ax = axs[0]
ax.set_title('Absolute velocities')
#ax.plot(time_vec, df['v_ref'].to_numpy(), color='k', linestyle='--', label='v ref')
for i in range(car_num):
    # Filter velocity signal
    df[f'v{i+1}_filtered'] = savgol_filter(df[f'v{i+1}'].to_numpy(), window_length, polyorder)
    # Plot filtered data
    ax.plot(time_vec, df[f'v{i+1}_filtered'].to_numpy(), label=fr'$v_{i+1}$', color='#'+colors[i], linewidth=3)
    
ax.set_ylabel('[m\s]')
add_masks_and_legend(ax,mask_ff_action,mask_attack,mask_attack_detected,exp1,exp2,exp3,False)

# # Fill between
# top_lim = ax.get_ylim()[1]  
# bot_lim = ax.get_ylim()[0] 
# ax.set_ylim(bottom=bot_lim,top = top_lim)
# #ax.fill_between(time_vec, top_lim, bot_lim, where=mask, color='gray', alpha=0.1, label='safety value disengaged')

# #ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_ff_action, color='darkgreen', alpha=0.1, label='u_ff active')
# if exp2 or exp3:
#     ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_attack, color='orangered', alpha=0.1, label='attack active')
#     ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_attack_detected, color='navy', alpha=0.1, label='attack detected')
#     ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_topology_change, color='purple', alpha=0.1, label='topology changed')

# ax.legend(ncol =ncol, handlelength=1, loc=loc, bbox_to_anchor=anchor)






# ---relative velocities ---
if exp1 or exp2:
    ax = axs[1]
    ax.set_title('Relative velocities')
    for i in range(1,car_num):
        # Filter relative velocity signal
        df[f'vrel{i+1}_filtered'] = savgol_filter(df[f'vrel{i+1}'].to_numpy(), window_length, polyorder)
        # Plot filtered data
        ax.plot(time_vec, df[f'vrel{i+1}_filtered'].to_numpy(), label=fr'$\tilde{{v}}_{i+1}$', color='#'+colors[i], linewidth=3)
    ax.set_ylabel('[m\s]')
    add_masks_and_legend(ax,mask_ff_action,mask_attack,mask_attack_detected,exp1,exp2,exp3,False)

    # # Fill between
    # top_lim = ax.get_ylim()[1]  
    # bot_lim = ax.get_ylim()[0] 

    # if exp2:
    #     bot_lim = bot_lim - 0.2


    # ax.set_ylim(bottom=bot_lim,top = top_lim)
    # #ax.fill_between(time_vec, top_lim, bot_lim, where=mask, color='gray', alpha=0.1, label='safety value disengaged')
    # #ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_ff_action, color='darkgreen', alpha=0.1, label='u_ff active')
    # if exp2 or exp3:
    #     ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_attack, color='orangered', alpha=0.1, label='attack active')
    #     ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_attack_detected, color='navy', alpha=0.1, label='attack detected')
    #     ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_topology_change, color='purple', alpha=0.1, label='topology changed')

    # ax.set_ylabel('[m\s]')
    # ax.legend(ncol =ncol, handlelength=1, loc=loc, bbox_to_anchor=anchor)


# --- residual signal ---
if exp3:
    ax = axs[1]
    ax.set_title('Residual')

    # Filter alarm signal
    #df['alarm2_filtered'] = savgol_filter(df['alarm2'].to_numpy(), window_length, polyorder)
    # Plot filtered data
    #ax.plot(time_vec, df['alarm2_filtered'].to_numpy(), label=f'alarm 2', color='#'+colors[1], linewidth=3)
    ax.plot(time_vec, 0.5 * np.ones(len(time_vec)), color="gray", linestyle='--', label=r'$\bar{{r}}$', linewidth=3)
    for i in range(4):
        ax.plot(time_vec, df['alarm'+str(i+1)].to_numpy(), label=fr'${{r}}_{1+i}$', color='#'+colors[i], linewidth=3)

    add_masks_and_legend(ax,mask_ff_action,mask_attack,mask_attack_detected,exp1,exp2,exp3,True)

    # ax.plot(time_vec, df['alarm1'].to_numpy(), label=r'$r_1$', color='#'+colors[0], linewidth=3)
    # ax.plot(time_vec, df['alarm2'].to_numpy(), label=r'$r_2$', color='#'+colors[1], linewidth=3)
    # ax.plot(time_vec, df['alarm3'].to_numpy(), label=r'$r_3$', color='#'+colors[2], linewidth=3)
    # ax.plot(time_vec, df['alarm4'].to_numpy(), label=r'$r_4$', color='#'+colors[3], linewidth=3)

    # top_lim = ax.get_ylim()[1]  
    # bot_lim = ax.get_ylim()[0]
    # ax.set_ylim(bottom=bot_lim,top = top_lim)


    # #ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_ff_action, color='darkgreen', alpha=0.1, label='u_ff active')
    # ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_attack, color='orangered', alpha=0.1, label='attack active')
    # ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_attack_detected, color='navy', alpha=0.1, label='attack detected')
    # ax.fill_between(time_vec, top_lim +1, bot_lim -1, where=mask_topology_change, color='purple', alpha=0.1, label='topology changed')
    

    # ax.legend(ncol =ncol, handlelength=1, loc=loc, bbox_to_anchor=anchor)





# set time axis only on last plot
axs[2].set_xlabel('Time [s]')


plt.show()




