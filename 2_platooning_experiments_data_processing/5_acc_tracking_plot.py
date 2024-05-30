from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline



# this file plots the low-level controller acceleration tracking performance. 
# It was used to tune the gains for the experiments




# NOTE: set current working directory as "platooning_experiments_data_processing" before running this file
file_path = 'experiment_data/data_experiment_1_2024-03-22-16-30-52.csv' 



# read the raw data
df = pd.read_csv(file_path)
time_vec = df['elapsed time sensors'].to_numpy()[:-1]
mask = np.array(df['safety_value']) == 1
mask = mask[:-1]

dt = 0.1

#set colors for cars
car_num = 4
colors = ["9ee493","6d98ba","ff8360","2e282a"]


# --- process the raw data ---
measured_acc_1 = (df['v1'].to_numpy()[1:]-df['v1'].to_numpy()[:-1])/dt
measured_acc_2 = (df['v2'].to_numpy()[1:]-df['v2'].to_numpy()[:-1])/dt
measured_acc_3 = (df['v3'].to_numpy()[1:]-df['v3'].to_numpy()[:-1])/dt
measured_acc_4 = (df['v4'].to_numpy()[1:]-df['v4'].to_numpy()[:-1])/dt


desired_acc_1 = df['u_des1'].to_numpy()[:-1]
desired_acc_2 = df['u_des2'].to_numpy()[:-1]
desired_acc_3 = df['u_des3'].to_numpy()[:-1]
desired_acc_4 = df['u_des4'].to_numpy()[:-1]




# evaluate covariance
# Calculate the covariance matrix
covariance_matrix = np.cov(measured_acc_1-desired_acc_1, rowvar=False)

print("Covariance Matrix of discrepancy between desired and measured acceleration:")
print(covariance_matrix)




mask_ff_action = np.array(df['add_ff']) == True
mask_ff_action = mask_ff_action[1:]

# plot raw data
# Assuming you have measured_acc_1, measured_acc_2, measured_acc_3, and time_vec defined

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Plotting acceleration 1
axes[0].plot(time_vec, measured_acc_1, label='acc measured 1', color='#'+colors[0])
#axes[0].plot(time_vec, imu_acc_1, label='imu_acc_1', color='orangered')
axes[0].plot(time_vec, desired_acc_1, label='acc reference 1', color='k',linestyle='--')
axes[0].fill_between(time_vec, axes[0].get_ylim()[0], axes[0].get_ylim()[1], where=mask, color='gray', alpha=0.1, label='safety value disingaged')
axes[0].fill_between(time_vec, axes[0].get_ylim()[0], axes[0].get_ylim()[1], where=mask_ff_action, color='darkgreen', alpha=0.1, label='add ff active')
axes[0].legend()
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Acceleration 1 (m/s^2)')

# Plotting acceleration 2
axes[1].plot(time_vec, measured_acc_2, label='acc measured 2', color='#'+colors[1])
#axes[1].plot(time_vec, imu_acc_1, label='imu_acc_2', color='orangered')
axes[1].plot(time_vec, desired_acc_2, label='acc reference 2', color='k',linestyle='--')
axes[1].fill_between(time_vec, axes[1].get_ylim()[0], axes[1].get_ylim()[1], where=mask, color='gray', alpha=0.1, label='safety value disingaged')
axes[1].fill_between(time_vec, axes[1].get_ylim()[0], axes[1].get_ylim()[1], where=mask_ff_action, color='darkgreen', alpha=0.1, label='add ff active')
axes[1].legend()
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Acceleration 2 (m/s^2)')

# Plotting acceleration 3
axes[2].plot(time_vec, measured_acc_3, label='acc measured 3',color='#'+colors[2])
#axes[2].plot(time_vec, imu_acc_3, label='imu_acc_3', color='orangered')
axes[2].plot(time_vec, desired_acc_3, label='acc reference 3', color='k',linestyle='--')
axes[2].fill_between(time_vec, axes[2].get_ylim()[0], axes[2].get_ylim()[1], where=mask, color='gray', alpha=0.1, label='safety value disingaged')
axes[2].fill_between(time_vec, axes[2].get_ylim()[0], axes[2].get_ylim()[1], where=mask_ff_action, color='darkgreen', alpha=0.1, label='add ff active')
axes[2].legend()
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Acceleration 3 (m/s^2)')


# Plotting acceleration 4
axes[3].plot(time_vec, measured_acc_4, label='acc measured 4', color='#'+colors[3])
#axes[2].plot(time_vec, imu_acc_3, label='imu_acc_3', color='orangered')
axes[3].plot(time_vec, desired_acc_4, label='acc reference 4', color='k',linestyle='--')
axes[3].fill_between(time_vec, axes[3].get_ylim()[0], axes[3].get_ylim()[1], where=mask, color='gray', alpha=0.1, label='safety value disingaged')
axes[3].fill_between(time_vec, axes[3].get_ylim()[0], axes[3].get_ylim()[1], where=mask_ff_action, color='darkgreen', alpha=0.1, label='add ff active')
axes[3].legend()
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('Acceleration 4 (m/s^2)')




# Fine-tune the figure layout
fig.subplots_adjust(top=0.985, bottom=0.11, left=0.095, right=0.995, hspace=0.2, wspace=0.2)

plt.show()

