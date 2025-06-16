import rosbag
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import os

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray



from matplotlib import rc
font = {'family' : 'serif',
        #'serif': ['Times New Roman'],
        'size'   : 20}

rc('font', **font)



# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Configuration
bag_file = '2025-06-04-21-41-50_first_success.bag'
output_video = 'attack_pipeline_plot_animation.mp4'
fps = 5
window_duration = 11  # seconds
image_width, image_height = 1280, 720

# Data containers
data_acc = []
data_ucom = []


vehicle_colors = ['#999999',"#d8973c","#63b4d1","#0c7c59","#f4796b"]
# vehicle_colors = [
#     '#999999',  # Index 0 (unused or placeholder)
#     '#f0ad0d',  # Vehicle 1
#     '#47e9ff',  # Vehicle 2
#     '#37f00d',  # Vehicle 3
#     '#f00d0d',  # Vehicle 4
# ]


linewidth = 3


# Temporary combined storage
all_messages = []

with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        ts = t.to_sec()

        # Velocity references
        if topic.startswith("/v_ref_"):
            try:
                car_idx = int(topic[-1])
                all_messages.append((ts, f'v_ref_{car_idx}', msg.data))
            except:
                pass

        # Alarms
        elif topic.startswith("/alarm_"):
            try:
                car_idx = int(topic[-1])
                all_messages.append((ts, f'alarm_{car_idx}', msg.data))
            except:
                pass

        # Relative distances
        elif topic.startswith("/relative_state_"):
            try:
                car_idx = int(topic[-1])
                if isinstance(msg.data, (list, tuple)) and len(msg.data) > 1:
                    all_messages.append((ts, f'rel_{car_idx}', -msg.data[1]))
            except:
                pass




# Sort all messages by timestamp
all_messages.sort(key=lambda x: x[0])

# Velocity references
data_vref   = {i: [(ts, val) for ts, tag, val in all_messages if tag == f'v_ref_{i}'] for i in range(1, 5)}

# Alarms
data_alarms = {i: [(ts, val) for ts, tag, val in all_messages if tag == f'alarm_{i}'] for i in range(1, 5)}

# Relative distances
data_rels   = {i: [(ts, val) for ts, tag, val in all_messages if tag == f'rel_{i}'] for i in range(1, 5)}




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample time_clip for plotting and processing
time_clip = [18.5, 34]

# --- Smoothing & interpolation for relative distance ---
def process_data(df, time_clip,tstart, jump_threshold=0.05):
    df['time'] = df['time'] -tstart #- df['time'].iloc[0]
    df = df[(df['time'] >= time_clip[0]) & (df['time'] <= time_clip[1])].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame({'time': [], 'distance': []})

    new_time = np.arange(df['time'].iloc[0], df['time'].iloc[-1] + 0.1, 0.1)
    new_distance = np.interp(new_time, df['time'], df['distance'])

    for i in range(1, len(new_distance) - 1):
        prev_val = new_distance[i - 1]
        curr_val = new_distance[i]
        next_val = new_distance[i + 1]
        if (abs(curr_val - prev_val) > jump_threshold and
            abs(curr_val - next_val) > jump_threshold and
            (curr_val > prev_val and curr_val > next_val or
             curr_val < prev_val and curr_val < next_val)):
            new_distance[i] = (prev_val + next_val) / 2

    df_new = pd.DataFrame({'time': new_time, 'distance': -new_distance})
    return df_new


# --- PLOTTING ---
fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

fig.subplots_adjust(
top=0.946,
bottom=0.125,
left=0.058,
right=0.908,
hspace=0.369,
wspace=0.2
)


loc = 'upper left'
anchor = (1,1.07) #1.05



# get smallest value between initial times
tstart_vref = min([min(data_vref[i], key=lambda x: x[0])[0] for i in range(1, 5) if data_vref[i]])
tstart_alarm = min([min(data_alarms[i], key=lambda x: x[0])[0] for i in range(1, 5) if data_alarms[i]])
tstart_rel = min([min(data_rels[i], key=lambda x: x[0])[0] for i in range(1, 5) if data_rels[i]])

# get smallest of the 3
tstart = min(tstart_vref, tstart_alarm, tstart_rel)


offset_vref = [0, -0.01, 0.02, 0.01, 0.0]  # Offset for each vehicle's velocity reference

# Velocity Reference Plot with offset
for i in range(1, 5):
    vref_data = data_vref.get(i, [])
    if vref_data:
        ts, vals = zip(*vref_data)
        ts = np.array(ts) - tstart
        ts_clip, vals_clip = zip(*[(t, v) for t, v in zip(ts, vals) if time_clip[0] <= t <= time_clip[1]])
        vals_clip = np.array(vals_clip) + offset_vref[i]  # Apply offset
        axes[0].plot(ts_clip, vals_clip, label = fr'$v_{{D,{i}}}$', color=vehicle_colors[i],linewidth=linewidth)

axes[0].set_ylabel('Velocity [m/s]')
axes[0].legend(loc=loc, bbox_to_anchor=anchor, ncol=1, handlelength=1)
axes[0].set_title(r'$v_D$')
# set limits
axes[0].set_xlim(time_clip[0], time_clip[1])



# Create a common time base for 0.1s intervals
global_time = np.arange(time_clip[0], time_clip[1] + 0.1, 0.1)

# Alarm Plot (Downsampled to 0.1s)
for i in range(1, 5):
    alarm_data = data_alarms.get(i, [])
    if alarm_data:
        ts, vals = zip(*alarm_data)
        ts = np.array(ts) - tstart
        df = pd.DataFrame({'time': ts, 'value': vals})
        df = df[(df['time'] >= time_clip[0]) & (df['time'] <= time_clip[1])]

        # Interpolate using nearest value to keep alarms discrete (not interpolated linearly)
        interp_vals = np.interp(global_time, df['time'].to_numpy(), df['value'].to_numpy(), left=0, right=0)

        axes[1].plot(global_time, interp_vals, label = fr'$r_{{{i}}}$', color=vehicle_colors[i], linewidth=linewidth)
# plot horizontal line at y=0.75
axes[1].axhline(y=0.75, color='gray', linestyle='--', linewidth=linewidth, label=r'$\bar{r}$')
axes[1].set_ylabel('residual [m/s]')
axes[1].legend(loc=loc, bbox_to_anchor=anchor, ncol=1, handlelength=1)
axes[1].set_title('Residuals')
# set limits
axes[1].set_xlim(time_clip[0], time_clip[1])
# x label to time
axes[1].set_xlabel('Time [s]')


# # Relative Distance Plot
# for i in range(1, 5):
#     rel_data = data_rels.get(i, [])
#     if rel_data:
#         ts, vals = zip(*rel_data)
#         df_rel = pd.DataFrame({'time': ts, 'distance': vals})
#         df_rel_proc = process_data(df_rel, time_clip,tstart)
#         if not df_rel_proc.empty:
#             axes[2].plot(df_rel_proc['time'].to_numpy(), -df_rel_proc['distance'].to_numpy(), label=f'd_rel_{i}', color=colors[i-1])

# axes[2].set_xlabel('Time [s]')
# axes[2].set_ylabel('Distance [m]')
# axes[2].legend()
# axes[2].set_title('Relative Distances')

plt.show()







