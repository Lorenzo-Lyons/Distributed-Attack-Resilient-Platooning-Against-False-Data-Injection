from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, lfilter
from matplotlib import rc
font = {'family' : 'serif',
        #'serif': ['Times New Roman'],
        'size'   : 20}

rc('font', **font)


# This file simulates the residual dynamics of the attack detection module on the data from experiment 2.
# It was used to set the gain K and residual threshold.

# Select Kalman filter gain
K = 0.05

# experiment 2 data
# NOTE: set current working directory as "platooning_experiments_data_processing" before running this file
file_path = 'experiment_data/data_experiment_2_03_22_2024_17_48_58.csv'




# read the raw data
df = pd.read_csv(file_path)
# extract values where safety is off
df = df[df['safety_value']==1]



time_vec = df['elapsed time sensors'].to_numpy()[1:] - df['elapsed time sensors'].to_numpy()[1]




#set colors for cars
colors = ["71b6cb","00a6d6","5a7b86","000000"] 










# simulate a FDI attack


# --- process the raw data ---

mask_attack = df['attack']
acc1 = df['u_com1'].to_numpy()[1:]






# follower vehicle data
dt = 0.1
acc2 = (df['v2'].to_numpy()[1:]-df['v2'].to_numpy()[:-1])/dt
vrel2 = df['vrel2'].to_numpy()[1:]



#using a kalman filter

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






# define masks
mask = np.array(df['safety_value']) == 1
mask_ff_action = np.array(df['add_ff']) == True
mask_attack = np.array(df['attack']) == True
# mask_attack_detected = np.array(df['alarm2']) > 1000 # initialize to all False



import matplotlib.pyplot as plt

# Assuming you have defined time_vec, vrel2, v_estimate, sigma2, mask, and mask_attack

# Plot residual dynamics, vrel2, and v_estimate in the first subplot

fig, ax = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
fig.canvas.set_window_title('Data Plots')
fig.subplots_adjust(top=0.965,
bottom=0.07,
left=0.07,
right=0.995,
hspace=0.2,
wspace=0.2)





# Plot u_des1, u_com1, and acc1 in the second subplot
ax[0].plot(time_vec, df['u_des1'].to_numpy()[1:], label='acc 1', color='gray') # '#'+colors[0]
ax[0].plot(time_vec, df['u_com1'].to_numpy()[1:], label='acc com 1', color='k', linestyle='--') #'#'+colors[1]

top_lim = ax[0].get_ylim()[1]  
bot_lim = ax[0].get_ylim()[0] 
bot_lim = bot_lim - 0.1
top_lim = top_lim + 0.1
ax[0].set_ylim(bottom=bot_lim, top = top_lim)
ax[0].set_xlim(left=0.0, right = time_vec[-1])


ax[0].fill_between(time_vec, bot_lim, top_lim, where=mask_attack[1:], color='orangered', alpha=0.1, label='attack')
ax[0].legend()
ax[0].set_ylabel('Acceleration $[m/s^2]$')



# Plot residual dynamics in the third subplot
ax[1].plot(time_vec, residual, color='k', linestyle='-', label='alarm 2') #'#'+colors[1]


top_lim = ax[1].get_ylim()[1]  
bot_lim = 0.0
top_lim = top_lim + 0.1
ax[1].set_ylim(bottom=bot_lim, top = top_lim)
ax[1].set_xlim(left=0.0, right = time_vec[-1])



ax[1].fill_between(time_vec,  bot_lim, top_lim, where=mask_attack[1:], color='orangered', alpha=0.1, label='attack')
ax[1].legend()

# Set common labels and finalize the layout
ax[0].set_xlabel('Time [s]')
ax[1].set_xlabel('Time [s]')
plt.tight_layout()
plt.show()



plt.show()

