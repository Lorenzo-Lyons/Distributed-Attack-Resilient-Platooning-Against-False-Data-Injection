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





sim_folder_name = 'simulation_data_statistics_constant_attack'
#sim_folder_name = 'simulation_data_statistics_sinusoidal_attack'
#sim_folder_name = 'simulation_data_statistics_random_attack'


# set current directory where the script is located
import os
# set current directory as folder where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# get list of all folders in the folder "simualtion_data"
import re

# Path to simulation_data directory
simulation_data_dir = os.path.join(script_dir, sim_folder_name)

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




# TEMP, eventually have this for all simulations
sim_name = sorted_folders[0]

# load simualtion parameters
time_vec = np.load(os.path.join(simulation_data_dir, sim_name, 'time_vec.npy')) 
time_to_attack = np.load(os.path.join(simulation_data_dir, sim_name, 'time_to_attack.npy'))
time_to_brake = np.load(os.path.join(simulation_data_dir, sim_name, 'time_to_brake.npy'))

# get extraction indexes between time to attack and time to brake
time_after_attack_starts = 2
time_to_attack_idx = np.argmin(np.abs(time_vec - (time_to_attack+time_after_attack_starts)))
time_to_brake_idx = np.argmin(np.abs(time_vec - time_to_brake))







# Set the simulation folder path
sim_folder = os.path.join(sim_folder_name, sim_name)

# Get all files in the folder that match the pattern "0x_sim", "1x_sim", etc.
sim_files = [f for f in os.listdir(sim_folder)
             if os.path.isfile(os.path.join(sim_folder, f)) and re.match(r'^\d+x_sim.npy$', f)]
# sort the files
sim_files = sorted(sim_files, key=lambda name: int(re.match(r'(\d+)', name).group()))

# Count the number of simulation files
sims_already_in_folder = len(sim_files)

print(f"Simulation folder: {sim_folder}")
print(f"Found {sims_already_in_folder} simulation runs.")





# Define bins for the histogram (shared across all cars)
# Example: from -10 to 10, 50 bins
bins = np.linspace(-60, 5, 100)  # 50 bins between -10 and -4
num_bins = len(bins) - 1  # histogram bins count

# Initialize histogram accumulators: one histogram per car (we'll infer car count from the first file)
histograms = None  # will be a 2D array: (num_cars, num_bins)


car_crashes = np.zeros(sims_already_in_folder)
file_count = 0




for file in sim_files:
    x_sim = np.load(os.path.join(sim_folder,file))  # shape: (time_steps, num_cars)
    # get the data between time to attack and time to brake
    x_sim_attack_active = x_sim[time_to_attack_idx:time_to_brake_idx+1,:]  # shape: (time_steps, num_cars)


    # evaluate distance between cars
    distances =  np.diff(x_sim_attack_active, axis=1)  # shape: (time_steps, num_cars-1)


    if histograms is None:
        num_cars = distances.shape[1]
        histograms = np.zeros((num_cars, num_bins), dtype=int)
    
    for car_idx in range(distances.shape[1]):
        counts, _ = np.histogram(distances[:, car_idx], bins=bins)
        histograms[car_idx] += counts



    # check if there was a crash or not
    distances_final = np.diff(x_sim[-1,:])
    if np.any(distances_final > 0):
        car_crashes[file_count] = 0
    else:
        car_crashes[file_count] = 1
    
    file_count += 1
    


# print out statistics on survival rate
print('survival rate = ',np.sum(car_crashes)/file_count)


# # Now histograms[i] holds the total bin counts for car i
# # To visualize or access:
# for i, hist in enumerate(histograms):
#     print(f"Histogram for Car {i}:")
#     print(hist)


# # Plotting
# bin_centers = 0.5 * (bins[:-1] + bins[1:])

# for i, hist in enumerate(histograms):
#     plt.figure(figsize=(8, 4))
#     plt.bar(bin_centers, hist, width=(bins[1] - bins[0]), edgecolor='black')
#     plt.title(f'Histogram of Values for Car {i}')
#     plt.xlabel('Value')
#     plt.ylabel('Count')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


# Define bin centers
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Plotting all histograms on the same plot with a lower alpha
plt.figure(figsize=(10, 6))

# Normalize each histogram
for i, hist in enumerate(histograms):
    # Normalize the histogram to make it a probability distribution
    hist_norm = hist / np.sum(hist)  # Normalize so the sum of all bars equals 1
    
    # Plot each normalized histogram with a low alpha for transparency
    plt.bar(bin_centers, hist_norm, width=(bins[1] - bins[0]), alpha=1/x_sim.shape[1], edgecolor='none', label=f'Car {i}', color=methods_colors[-1])

plt.title('Overlayed Normalized Histograms of All Cars')
plt.xlabel('Value')
plt.ylabel('Probability Density')  # Y-axis now represents a probability density
plt.grid(True)

# Optional: Add a legend to differentiate cars
plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), title="Cars")

plt.tight_layout()
plt.show()

