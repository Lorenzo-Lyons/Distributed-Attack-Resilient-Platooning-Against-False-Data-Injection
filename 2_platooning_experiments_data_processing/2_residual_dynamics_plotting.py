from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

# This file produces the residual dynamics plot. 




from matplotlib import rc
font = {'family' : 'serif',
        #'serif': ['Times New Roman'],
        'size'   : 20}

rc('font', **font)


#experiment 2
file_path = 'experiment_data/data_experiment_2_03_22_2024_17_48_58.csv'



# read the raw data
df = pd.read_csv(file_path)
# extract values where safety is off
df = df[df['safety_value']==1]

# produce time vector
time_vec = df['elapsed time sensors'].to_numpy()[1:] - df['elapsed time sensors'].to_numpy()[1]

#set colors for cars
colors = ["71b6cb","00a6d6","5a7b86","000000"] 




# reconstruct residual signal as in vehicle 2 since attack detection was disabled during experiment 2.
acc1 = df['u_com1'].to_numpy()[1:]

# follower vehicle data
dt = 0.1
acc2 = (df['v2'].to_numpy()[1:]-df['v2'].to_numpy()[:-1])/dt
vrel2 = df['vrel2'].to_numpy()[1:]

#using a steady state kalman filter
# Kalman filter parameters
K = 0.05 # steady state kalman filter gain

v_estimate = np.zeros(len(time_vec))
residual = np.zeros(len(time_vec))

for t in range(1,len(time_vec)):
    if df['safety_value'].iloc[t] == 1:

        # Prediction step
        x_hat_minus = v_estimate[t-1] + dt * (acc1[t] -acc2[t])

        # Update step
        v_estimate[t] = x_hat_minus * (1-K) + K * vrel2[t]

        #evaluate residual
        residual[t] = np.abs(vrel2[t]-v_estimate[t])




# --- plotting ---

# define mask
mask_attack = np.array(df['attack']) == True



# Plot residual dynamics, vrel2, and v_estimate in the first subplot

fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.canvas.set_window_title('Data Plots')
fig.subplots_adjust(top=0.951,
bottom=0.089,
left=0.072,
right=0.806,
hspace=0.185,
wspace=0.2)



# Plot acceleration communication
ax[0].set_title('Acceleration vehicle 1')
ax[0].plot(time_vec, df['u_des1'].to_numpy()[1:], label='real ', color='#'+colors[0], linewidth=3) 
ax[0].plot(time_vec, df['u_com1'].to_numpy()[1:], label='communicated ', color='#'+colors[1], linestyle='--', linewidth=3) 

top_lim = ax[0].get_ylim()[1]  
bot_lim = ax[0].get_ylim()[0] 
bot_lim = bot_lim - 0.1
top_lim = top_lim + 0.1
ax[0].set_ylim(bottom=bot_lim, top = top_lim)
ax[0].set_xlim(left=0.0, right = time_vec[-1])


ax[0].fill_between(time_vec, bot_lim, top_lim, where=mask_attack[1:], color='orangered', alpha=0.1, label='attack active', linewidth=3)
# set legend parameters
loc = 'upper left'
anchor = (1,1.05)

ncol = 1
right = 0.995
ax[0].legend(ncol =ncol, handlelength=1, loc=loc, bbox_to_anchor=anchor)
ax[0].set_ylabel('Acceleration 1 [m/s^2]')




# Plot residual dynamics
ax[1].set_title('Residual vehicle 2')
ax[1].plot(time_vec, residual, color='#'+colors[1], linestyle='-', label=r'${r}_{2}$', linewidth=3) #'#'+colors[1]

top_lim = ax[1].get_ylim()[1]  
bot_lim = 0.0
top_lim = top_lim + 0.1
ax[1].plot(time_vec, 0.5 * np.ones(len(time_vec)), color="gray", linestyle='--', label=r'$\bar{{r}}$', linewidth=3)
ax[1].set_ylim(bottom=bot_lim, top = top_lim)
ax[1].set_xlim(left=0.0, right = time_vec[-1])
ax[1].set_ylabel('Residual 2 [m/s]')

ax[1].fill_between(time_vec,  bot_lim, top_lim, where=mask_attack[1:], color='orangered', alpha=0.1, label='attack active', linewidth=3)
ax[1].legend(ncol =ncol, handlelength=1, loc=loc, bbox_to_anchor=anchor)


# Set common labels and finalize the layout
#ax[0].set_xlabel('Time [s]')
ax[1].set_xlabel('Time [s]')
plt.show()



plt.show()

