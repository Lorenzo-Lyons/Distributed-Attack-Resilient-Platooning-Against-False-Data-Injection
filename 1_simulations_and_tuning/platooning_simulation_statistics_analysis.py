import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from classes_definintion import platooning_problem_parameters,Vehicle_model,set_scenario_parameters,generate_color_gradient,generate_color_1st_last_gray
from tqdm import tqdm
import os
import re

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


def select_color(folder_name,methods_colors):
    if folder_name == '1DMPC N=40':
        color = methods_colors[0]
    elif folder_name == '2DMPC N=20':
        color = methods_colors[1]
    elif folder_name == '3Kafash et al.':
        color = methods_colors[2]
    elif folder_name == '4ACC + CACC':
        color = methods_colors[3]
    return color





#sim_folder_name = 'simulation_data_statistics_constant_attack'
#sim_folder_name = 'simulation_data_statistics_sinusoidal_attack'
sim_folder_name = 'simulation_data_statistics_random_attack'


# set current directory where the script is located


# set current directory as folder where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# get list of all folders in the folder "simualtion_data"


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
statistics = {}




# Plotting all histograms on the same plot with a lower alpha
plt.figure(figsize=(7, 4))

plt.subplots_adjust(
    top=1.0,
    bottom=0.148,
    left=0.129,
    right=0.986,
    hspace=0.2,
    wspace=0.2
)

# TEMP, eventually have this for all simulations
for sim_name in sorted_folders:
    # Format label: bold if last folder
    if sim_name[1:] == 'ACC + CACC':
        plot_label = r'$\mathbf{' + sim_name[1:] + '}$'
        plot_linewidth = 3
        line_plot_alpha = 0.8
        alpha = 0.4
    else:
        plot_label = sim_name[1:]
        plot_linewidth = 2
        line_plot_alpha = 0.5
        alpha = 0.4
    
    if plot_label == 'Kafash et al.':
        line_plot_alpha = 0.3
        alpha = 0.2
    #sim_name = sorted_folders[0]

    # load simualtion parameters
    time_vec = np.load(os.path.join(simulation_data_dir, sim_name, 'time_vec.npy')) 
    time_to_attack = np.load(os.path.join(simulation_data_dir, sim_name, 'time_to_attack.npy'))
    time_to_brake = np.load(os.path.join(simulation_data_dir, sim_name, 'time_to_brake.npy'))

    # get extraction indexes between time to attack and time to brake
    time_after_attack_starts = 20
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


    print('')
    print(f"Simulation folder: {sim_folder}")
    print(f"Found {sims_already_in_folder} simulation runs.")





    # Define bins for the histogram (shared across all cars)
    # Example: from -10 to 10, 50 bins
    bins = np.linspace(-50, 20, 300) 
    num_bins = len(bins) - 1  # histogram bins count

    # Initialize histogram accumulators: one histogram per car (we'll infer car count from the first file)
    histograms = None  # will be a 2D array: (num_cars, num_bins)


    car_crashes = np.zeros(sims_already_in_folder)
    file_count = 0



    total_car_crashes = 0
    total_car_crashes_during_attack = 0

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



        # check if there was a crash or not during attack
        distances_final_during_attack = np.diff(x_sim_attack_active[-1,:])
        total_car_crashes_during_attack = total_car_crashes_during_attack + np.sum(distances_final_during_attack > 0)

        # check if there was a crash or not during the whole simulation
        distances_final = np.diff(x_sim[-1,:])
        total_car_crashes = total_car_crashes + np.sum(distances_final > 0)
        # if sim_name == '3Kafash et al.':
        #     b = 1
        # else:
        #     b = 1

        file_count += 1
        

    succes_rate_attack = 100 - (total_car_crashes_during_attack / (file_count * (x_sim.shape[1] - 1))) * 100
    succes_rate = 100 - (total_car_crashes / (file_count * (x_sim.shape[1] - 1))) * 100


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

    # Sum histograms
    hist_sum = np.sum(histograms, axis=0)

    # Normalize the total histogram
    hist_sum_norm = hist_sum / np.sum(hist_sum)

    # print out statistics
    mean = np.sum(hist_sum_norm*bin_centers)
    std = np.sqrt(np.sum(hist_sum_norm*(bin_centers-mean)**2))
    # Find indices where histogram is non-zero
    non_zero_indices = np.where(hist_sum_norm > 0)[0]

    # Use those indices to find min and max values
    min_val = bin_centers[non_zero_indices[0]]
    max_val = bin_centers[non_zero_indices[-1]]
    
    print('statistics of ', plot_label)
    print('mean = ',mean)
    print('std = ',std)
    print('max = ',max_val)
    print('min = ',min_val)
    print('survival rate during attack = ',succes_rate_attack)
    print('survival rate = ',succes_rate)

    # add to data structure
    statistics[sim_name[1:]] = {
        'mean': mean,
        'std': std,
        'max': max_val,
        'min': min_val,
        'succes_rate_attack': succes_rate_attack,
        'succes_rate': succes_rate
    }







    # Plot the normalized summed histogram
    # if sim_name == '4ACC + CACC':
    #     plt.bar(bin_centers, hist_sum_norm, width=(bins[1] - bins[0]), alpha=0.4, edgecolor='none',color=select_color(sim_name,methods_colors),label = sim_name) #, label=f'Car {i}'
    # else:   
    # Plot the filled bars without edges
    # Plot the filled bars without edges
    plt.bar(
        bin_centers,
        hist_sum_norm,
        width=(bins[1] - bins[0]),
        alpha=alpha,
        edgecolor='none',
        color=select_color(sim_name, methods_colors),
        label=plot_label
    )

    # Prepare coordinates for the skyline
    bar_width = bins[1] - bins[0]
    left_edges = bin_centers - bar_width / 2
    right_edges = bin_centers + bar_width / 2

    # Interleave left and right x-positions
    x_outline = np.empty(2 * len(hist_sum_norm))
    x_outline[0::2] = left_edges
    x_outline[1::2] = right_edges

    # Repeat y-values to match x (left, right of each bar)
    y_outline = np.repeat(hist_sum_norm, 2)

    # Draw the skyline outline
    plt.plot(
        x_outline,
        y_outline,
        color=select_color(sim_name, methods_colors),     # Skyline color
        linewidth=plot_linewidth,       # Thickness
        alpha=line_plot_alpha,         # Transparency
        zorder=20,
    )

    # # Normalize each histogram
    # for i, hist in enumerate(histograms):
    #     # Normalize the histogram to make it a probability distribution
    #     hist_norm = hist / np.sum(hist)  # Normalize so the sum of all bars equals 1
        
    #     # Plot each normalized histogram with a low alpha for transparency
    #     plt.bar(bin_centers, hist_norm, width=(bins[1] - bins[0]), alpha=1/x_sim.shape[1], edgecolor='none', color=select_color(sim_name,methods_colors)) #, label=f'Car {i}'


plt.axvline(x=0, color='red', linestyle='--',label='collision',linewidth=1)
plt.xlabel('distance [m]')  # X-axis now represents distance
plt.ylabel('probability density')  # Y-axis now represents a probability density
plt.grid(False)

# Optional: Add a legend to differentiate cars
plt.legend(loc='upper left')# loc='upper right', bbox_to_anchor=(1.05, 1)
# Set the x-ticks at the same locations as before
xticks = np.arange(-20, 6, 5)

# Set the tick labels with the opposite sign (convert negative values to positive)
xtick_labels = [-tick for tick in xticks]
# Apply boldface to the 0 label
# for i, label in enumerate(xtick_labels):
#     if label == 0:
#         xtick_labels[i] = r'$\mathbf{0}$'  # LaTeX formatting for boldface

# Apply the new tick labels
plt.xticks(xticks, xtick_labels)


plt.xlim(-20, 5)

figure_path = os.path.join(simulation_data_dir, 'comparison_constant_attack.pdf')
plt.savefig(figure_path, format='pdf')




# produce latex table


# Order of rows to match your table
row_order = cleaned_folders

## Start LaTeX table string
latex_table = r"""\begin{table}[]
\begin{tabular}{c|c|c|c|c|c|c|}
\cline{2-7}
                                        & \begin{tabular}[c]{@{}c@{}}mean\\dist.\\ {[}m{]}\end{tabular} & \begin{tabular}[c]{@{}c@{}}std.\\ dev.\\ {[}m{]}\end{tabular} & \begin{tabular}[c]{@{}c@{}}max\\dist.\\ {[}m{]}\end{tabular} & \begin{tabular}[c]{@{}c@{}}min\\dist.\\ {[}m{]}\end{tabular} & \begin{tabular}[c]{@{}c@{}}safe\\ runs\\ (attack)\end{tabular} & \begin{tabular}[c]{@{}c@{}}safe\\ runs\\ (brake)\end{tabular} \\ \hline
"""

# # Add rows
# for label in row_order:
#     stats = statistics[label]
#     # Bold the row for "ACC + CACC"
#     row_name = r"\textbf{ACC+CACC}" if label == "ACC + CACC" else label
#     row = (
#         f"\multicolumn{{1}}{{|l|}}{{{row_name}}} & "
#         f"{stats['mean']:.2f} & "
#         f"{stats['std']:.2f} & "
#         f"{stats['max']:.2f} & "
#         f"{stats['min']:.2f} & "
#         f"{stats['succes_rate_attack']:.2f}\% & "
#         f"{stats['succes_rate']:.2f}\% \\\\ \\hline"
#     )
#     latex_table += row + "\n"

# # Finish table
# latex_table += r"""\end{tabular}
# \caption{}
# \label{tab:my-table}
# \end{table}
# """





# Add rows
for label in row_order:
    stats = statistics[label]

    is_bold_row = label == "ACC + CACC"
    row_name = r"\textbf{ACC+CACC}" if is_bold_row else label

    def fmt(val, is_percent=False):
        s = f"{val:.2f}\\%" if is_percent else f"{val:.2f}"
        return rf"\textbf{{{s}}}" if is_bold_row else s

    row = (
        f"\multicolumn{{1}}{{|l|}}{{{row_name}}} & "
        f"{fmt(-stats['mean'])} & "
        f"{fmt(stats['std'])} & "
        f"{fmt(-stats['max'])} & "
        f"{fmt(-stats['min'])} & "
        f"{fmt(stats['succes_rate_attack'], is_percent=True)} & "
        f"{fmt(stats['succes_rate'], is_percent=True)} \\\\ \\hline"
    )

    latex_table += row + "\n"

# Finish table
latex_table += r"""\end{tabular}
\caption{}
\label{tab:my-table}
\end{table}
"""




# Save to .txt file
table_path = os.path.join(simulation_data_dir, 'latex_table_comparison.txt')  
with open(table_path, "w") as f:
    f.write(latex_table)

print("LaTeX table saved to 'table_output.txt'.")
















plt.show()

