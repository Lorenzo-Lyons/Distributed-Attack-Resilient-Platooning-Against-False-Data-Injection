import rosbag
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# change the current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#vehicle_colors = ['#999999','#f0ad0d','#47e9ff','#37f00d','#f00d0d']
vehicle_colors = ['#999999',"#d8973c","#63b4d1","#0c7c59","#ff3c38"]

# Folder containing rosbags
#bag_folder = 'rebuttal_experiments/CACC_vs_ACC_data_only' 
bag_folder = 'rebuttal_experiments/CACC_vs_ACC_data_only_sin_v' 
#bag_folder = os.path.join('rebuttal_experiments/CACC_vs_ACC_data_only_constant_v')


csv_folder = os.path.join(bag_folder, 'csv_files')
topics = [f'/relative_state_{i}' for i in range(2, 5)] # skip relative state `1 cause no distance to predecessor`

# Create the CSV folder if it doesn't exist
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)





# Read all rosbag files
for filename in os.listdir(bag_folder):
    if filename.endswith('.bag'):
        # Storage for data
        data = {topic: {'time': [], 'distance': []} for topic in topics}

        bag_path = os.path.join(bag_folder, filename)
        print(f"Reading {bag_path}...")

        csv_experiment_folder = os.path.join(csv_folder, filename.replace('.bag', ''))

        # check if the csv file already exists
        # remove ".bag" from filename for csv file name
        if not os.path.exists(csv_experiment_folder):
            os.makedirs(csv_experiment_folder)
        file_name_no_bag = filename.replace('.bag', '')
        expected_files = [f'{file_name_no_bag}_relative_state_{i}_data.csv' for i in range(2, 5)]
        present_files = os.listdir(csv_experiment_folder)

        if all(f in present_files for f in expected_files):
            print(f"Data for {filename} already processed. Skipping...")
            continue

        
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=topics):
                try:
                    # Customize this according to your actual message structure
                    # Example assumes msg has a float64 field named "data"
                    data[topic]['time'].append(t.to_sec())
                    data[topic]['distance'].append(msg.data[1])
                
                except AttributeError as e:
                    print(f"Error in {topic}: {e}")
        
        # save as a pandas DataFrame
        for topic in topics:
            df = pd.DataFrame(data[topic])
            # remove ".bag" from filename
            filename = filename.replace('.bag', '')
            # check if the folder exists, if not create it
            csv_folder_path = os.path.join(csv_folder, filename)
            if not os.path.exists(csv_folder_path):
                os.makedirs(csv_folder_path)
            file_name = os.path.join(csv_folder_path,f'{filename}_{topic[1:]}_data.csv')
            df.to_csv(file_name, index=False)
            print(f"Saved {topic[1:]} data to CSV.")
        



# define time clipping for each experiment
time_clipping = {
    'CACC_sin_v_p2': (20, 68),
    'ACC_sin_v_p2': (20, 68),
    'CACC_constant_v': (20, 68),
    'ACC_constant_v': (20, 68),
}




def process_data(df, time_clip, jump_threshold=0.05):
    import numpy as np
    import pandas as pd

    # Reset time and clip
    df['time'] = df['time'] - df['time'].iloc[0]
    df = df[(df['time'] >= time_clip[0]) & (df['time'] <= time_clip[1])].reset_index(drop=True)

    # Create new time vector from first to last time in steps of 0.1s (10Hz)
    new_time = np.arange(df['time'].iloc[0], df['time'].iloc[-1] + 0.1, 0.1)

    # Interpolate distance to new time points
    new_distance = np.interp(new_time, df['time'], df['distance'])

    # Remove single-point peaks or dips based on threshold
    for i in range(1, len(new_distance) - 1):
        prev_val = new_distance[i - 1]
        curr_val = new_distance[i]
        next_val = new_distance[i + 1]
        if (abs(curr_val - prev_val) > jump_threshold and
            abs(curr_val - next_val) > jump_threshold and
            (curr_val > prev_val and curr_val > next_val or
             curr_val < prev_val and curr_val < next_val)):
            new_distance[i] = (prev_val + next_val) / 2

    # Create new DataFrame with cleaned time and distance
    df_new = pd.DataFrame({'time': new_time, 'distance': new_distance})
    # change distantce to negative
    df_new['distance'] = -df_new['distance']
    return df_new





# list all directories in the csv_folder
experiment_names = os.listdir(csv_folder)


# Plotting
fig, axs = plt.subplots(2, 1, figsize=(14, 3),sharey=True) # , sharex=True, sharey=True
axs = axs.flatten()
import os
import pandas as pd

# Assuming these are already defined somewhere earlier in your script:
# - process_data
# - csv_folder
# - experiment_names = ['ACC_eccecc', 'CACC_eccecc']
# - time_clipping = {...}
# - axs
# - vehicle_colors

# Initialize stats storage
stats = {
    'mean': {'ACC': {}, 'CACC': {}},
    'std': {'ACC': {}, 'CACC': {}},
    'max': {'ACC': {}, 'CACC': {}},
    'min': {'ACC': {}, 'CACC': {}}
}


d_target = 0.5  # target distance in meters


print('')
print('--------  statistics  ------')
print('')

for i, experiment_name in enumerate(experiment_names):
    experiment_folder = os.path.join(csv_folder, experiment_name)

    files = os.listdir(experiment_folder)
    files = [f for f in files if f.endswith('.csv')]
    files.sort(key=lambda x: int(x.split('_')[-2]))  # car number is second to last

    print('')
    for file in files:
        df = pd.read_csv(os.path.join(experiment_folder, file))

        # Process the data
        if experiment_name in time_clipping:
            time_clipping_range = time_clipping[experiment_name]
        else:
            time_clipping_range = (0, df['time'].max())
        df = process_data(df, time_clipping_range)

        # Determine exp type: ACC or CACC
        exp_type = file.split('_')[0]
        car_number = int(file.split('_')[-2])  # Assuming format: type_carNumber_xyz.csv

        # Save plot
        ax = axs[i]
        ax.plot(df['time'].to_numpy(), df['distance'].to_numpy(), label=str(car_number),
                color=vehicle_colors[car_number])
        # set title to ACC or CACC
        # check if CACC is in experiment_name
        if 'CACC' in experiment_name:
            ax.set_title(f"CACC")
        else:
            ax.set_title(f"ACC")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Distance")
        ax.grid(True)

        # Compute statistics
        mean_distance = df['distance'].mean()
        std_distance = df['distance'].std()
        max_distance = df['distance'].max()
        min_distance = df['distance'].min()

        print(f"Experiment: {experiment_name}, Car: {car_number}, Mean: {mean_distance:.2f}, Std: {std_distance:.2f}, Max: {max_distance:.2f}, Min: {min_distance:.2f}")

        # Store statistics
        stats['mean'][exp_type][car_number] = mean_distance
        stats['std'][exp_type][car_number] = std_distance
        stats['max'][exp_type][car_number] = max_distance
        stats['min'][exp_type][car_number] = min_distance

        # Add reference line and legend
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label=r'$d_{target}$')
    ax.legend()





# produce caption for the table
if 'sin' in bag_folder:
    caption = r"Distances for ACC and CACC platooning with leader executing a sinusoidal velocity profile. The data is collected over a $60$s period for each control algorithm. The mean represents the time-averaged value, and the standard deviation quantifies the temporal variation around this mean."
    latex_label = r"tab:distances_sin_v"
else:
    caption = r"Distances for CACC and ACC platooning with leader executing a constant velocity profile."
    latex_label = r"tab:distances_constant_v"




row_names = {
    'mean': r"mean {[}m{]}",
    'std': r"std dev {[}m{]}",
    'max': r"max {[}m{]}",
    'min': r"min {[}m{]}"
}

latex_lines = [
    r"\begin{table}[h]",
    r"\begin{tabular}{c|cc|cc|cc|}",
    r"\cline{2-7}",
    r"                                      & \multicolumn{2}{c|}{car 1}                                    & \multicolumn{2}{c|}{car 2}                                    & \multicolumn{2}{c|}{car 3}                                    \\ \cline{2-7}",
    r"\multicolumn{1}{l|}{}                 & \multicolumn{1}{l|}{ACC} & \multicolumn{1}{l|}{{CACC}} & \multicolumn{1}{l|}{ACC} & \multicolumn{1}{l|}{{CACC}} & \multicolumn{1}{l|}{ACC} & \multicolumn{1}{l|}{{CACC}} \\ \hline"
]

target_value = 0.5  # Reference value to compare against

for metric in ['mean', 'std', 'max', 'min']:
    row = rf"\multicolumn{{1}}{{|c|}}{{{row_names[metric]}}} "

    for car in [2, 3, 4]:  # updated here
        acc_val = stats[metric]['ACC'].get(car, None)
        cacc_val = stats[metric]['CACC'].get(car, None)

        if acc_val is not None and cacc_val is not None:
            # Determine which value should be bolded
            if metric in ['mean', 'max', 'min']:
                # Closest to 0.5
                acc_dist = abs(acc_val - target_value)
                cacc_dist = abs(cacc_val - target_value)
                if acc_dist < cacc_dist:
                    acc_fmt = rf"\textbf{{{acc_val:.2f}}}"
                    cacc_fmt = rf"{cacc_val:.2f}"
                else:
                    acc_fmt = rf"{acc_val:.2f}"
                    cacc_fmt = rf"\textbf{{{cacc_val:.2f}}}"
            elif metric == 'std':
                # Lower is better
                if acc_val < cacc_val:
                    acc_fmt = rf"\textbf{{{acc_val:.2f}}}"
                    cacc_fmt = rf"{cacc_val:.2f}"
                else:
                    acc_fmt = rf"{acc_val:.2f}"
                    cacc_fmt = rf"\textbf{{{cacc_val:.2f}}}"
            else:
                acc_fmt = rf"{acc_val:.2f}"
                cacc_fmt = rf"{cacc_val:.2f}"

            row += rf"& \multicolumn{{1}}{{c|}}{{{acc_fmt}}} & {cacc_fmt} "
        else:
            row += "& & "
    row += r"\\ \hline"
    latex_lines.append(row)


# add caption and label
latex_lines += [
    r"\cline{2-7}",
    r"\end{tabular}",
    r"\caption{" + caption + "}",
    r"\label{" + latex_label + "}",
    r"\end{table}"
]


# save the latex table in the same folder as the bag
latex_folder = os.path.join(bag_folder, 'latex_tables')
# create the folder if it doesn't exist
if not os.path.exists(latex_folder):
    os.makedirs(latex_folder)

# save the latex table to a file
    # check if teh word "sin" is in the folder name
if 'sin' in bag_folder:
    latex_file_path = os.path.join(latex_folder, 'distances_statistics_sin_v.tex')
else:
    latex_file_path = os.path.join(latex_folder, 'distances_statistics_constant_v.tex')


# Write the LaTeX table to file
with open(latex_file_path, 'w') as f:
    for line in latex_lines:
        f.write(line + '\n')

print("LaTeX table written to latex_table.txt")




# save the figure as a pdf
fig.savefig(os.path.join(bag_folder, 'CACC_vs_ACC_distances.pdf'), bbox_inches='tight')



plt.tight_layout()
plt.show()





