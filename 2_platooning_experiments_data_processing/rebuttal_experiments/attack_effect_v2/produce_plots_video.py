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






generate_video = False  # Set to False if you only want to generate the plots



# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Configuration
bag_file = '2025-06-04-21-14-34_good.bag'
output_video = 'attack_effect_plot_animation.mp4'
fps = 5
window_duration = 11  # seconds
image_width, image_height = 1280, 720

# Data containers
data_acc = []
data_ucom = []
data_rel = []


vehicle_colors_figure = ['#999999',"#d8973c","#63b4d1","#0c7c59","#ff3c38"]

vehicle_colors = [
    '#999999',  # Index 0 (unused or placeholder)
    '#f0ad0d',  # Vehicle 1
    '#47e9ff',  # Vehicle 2
    '#37f00d',  # Vehicle 3
    '#f00d0d',  # Vehicle 4
]


linewidth_acc = 5
linewidth_ucomm = 5

line_width_figure = 3

#plotting parameter visuals
font_size = 24


# Temporary combined storage
all_messages = []

with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        ts = t.to_sec()
        if topic == "/acc_saturated_1":
            all_messages.append((ts, 'acc', msg.data))
        elif topic == "/u_com_1":
            all_messages.append((ts, 'ucom', msg.data))
        elif topic == "/relative_state_2":
            if isinstance(msg.data, (list, tuple)) and len(msg.data) > 1:
                all_messages.append((ts, 'rel', -msg.data[1]))

# ✅ Sort globally by timestamp
all_messages.sort(key=lambda x: x[0])

# ✅ Now split back to per-topic lists
data_acc = [(ts, val) for ts, tag, val in all_messages if tag == 'acc']
data_ucom = [(ts, val) for ts, tag, val in all_messages if tag == 'ucom']
data_rel = [(ts, val) for ts, tag, val in all_messages if tag == 'rel']

# ✅ Determine global time range
all_timestamps = [ts for ts, _, _ in all_messages]
start_time = min(all_timestamps)
end_time = max(all_timestamps)
duration = end_time - start_time





##### GENERATE FIGURE #####
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def process_data(df, time_clip, jump_threshold=0.05):
    df['time'] = df['time'] - df['time'].iloc[0]
    df = df[(df['time'] >= time_clip[0]) & (df['time'] <= time_clip[1])].reset_index(drop=True)

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

# --- Unpack the data ---
acc_t, acc_vals_raw = zip(*data_acc) if data_acc else ([], [])
ucom_t, ucom_vals_raw = zip(*data_ucom) if data_ucom else ([], [])
rel_t, rel_vals = zip(*data_rel) if data_rel else ([], [])

# clip the acceleration values to a range
time_clip = [15, 53.4]

# Create DataFrame from acc data
df_acc = pd.DataFrame({'time': acc_t, 'acc': acc_vals_raw})
df_acc['time'] = df_acc['time'] - df_acc['time'].iloc[0]  # Start from zero
df_acc = df_acc[(df_acc['time'] >= time_clip[0]) & (df_acc['time'] <= time_clip[1])].reset_index(drop=True)

df_ucomm = pd.DataFrame({'time': ucom_t, 'ucom': ucom_vals_raw})
df_ucomm['time'] = df_ucomm['time'] - df_ucomm['time'].iloc[0]  # Start from zero
df_ucomm = df_ucomm[(df_ucomm['time'] >= time_clip[0]) & (df_ucomm['time'] <= time_clip[1])].reset_index(drop=True)



# --- Create DataFrame for relative data and process it ---
df_rel = pd.DataFrame({'time': rel_t, 'distance': rel_vals})
df_rel_proc = process_data(df_rel, time_clip)

# set time axis to min - max_time for the acceleration data
acc_t = np.array(acc_t) - acc_t[0] if acc_t else []
ucom_t = np.array(ucom_t) - ucom_t[0] if ucom_t else []
# clip between min and max val

# --- Create figure and subplots ---
fig, axs = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
fig.canvas.manager.set_window_title('ROS Bag Data Plot')

fig.subplots_adjust(
top=0.93,
bottom=0.12,
left=0.07,
right=0.81,
hspace=0.2,
wspace=0.2
)

loc = 'upper left'
anchor = (1,1.07) #1.05
right = 0.675


# --- Subplot 1: acc and ucom ---
ax1 = axs[0]
ax1.plot(df_acc['time'].to_numpy(), df_acc['acc'].to_numpy(), color=vehicle_colors_figure[1], linewidth=line_width_figure,label=r'real $\dot{v}_1$')
ax1.plot(df_ucomm['time'].to_numpy(), df_ucomm['ucom'].to_numpy(), color=vehicle_colors_figure[2], linewidth=line_width_figure, linestyle='--', label=r'broadcasted $\dot{v}_1$')
ax1.set_ylabel(r'$\dot{v}$ [m/s$^2$]')
ax1.set_title('Vehicle 1 acceleration')
ax1.legend(loc=loc, bbox_to_anchor=anchor, ncol=1, handlelength=1)
# set
ax1.set_ylim([-1.1, 1.1])

# --- Subplot 2: processed relative_state_2 ---
ax2 = axs[1]
ax2.plot(df_rel_proc['time'].to_numpy(), -df_rel_proc['distance'].to_numpy(), label=r'p_1-p_2', color=vehicle_colors_figure[2],linewidth=line_width_figure)
ax2.set_ylabel('Distance [m]')
ax2.set_title('Distance between vehicle 1 and 2')
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=line_width_figure, label='d')
ax2.legend(loc=loc, bbox_to_anchor=anchor, ncol=1, handlelength=1)
# add gary line at 0.5

ax2.set_xlabel('Time [s]')
# set x-axis limits
ax1.set_xlim([time_clip[0], time_clip[1]])

plt.show()



















##### GENERATE VIDEO #####
if generate_video:

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (image_width, image_height))

    # Generate frames
    fig, axs = plt.subplots(2, 1, figsize=(image_width / 100, image_height / 100), dpi=100)

    times = np.arange(start_time, end_time, 1.0 / fps)

    # Set global background
    fig.patch.set_facecolor('black')
    axs[0].set_facecolor('black')
    axs[1].set_facecolor('black')

    for current_time in times:
        # Collect rolling data
        acc_window = [(t, v) for t, v in data_acc if current_time - window_duration <= t <= current_time]
        ucom_window = [(t, v) for t, v in data_ucom if current_time - window_duration <= t <= current_time]
        rel_window = [(t, v) for t, v in data_rel if current_time - window_duration <= t <= current_time]

        # Clear subplots
        axs[0].clear()
        axs[1].clear()

        # Set black background and white ticks/titles
        for ax in axs:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')

        # Plot acc_saturated_1 in color of vehicle 1
        if acc_window:
            axs[0].plot(
                [t - current_time for t, _ in acc_window],
                [v for _, v in acc_window],
                label=r'real $\dot{v}_1$',
                color=vehicle_colors[1],
                linewidth=linewidth_acc,
                zorder=2
            )

        # Plot u_com_1 in color of vehicle 2
        if ucom_window:
            axs[0].plot(
                [t - current_time for t, _ in ucom_window],
                [v for _, v in ucom_window],
                label=r'broadcasted $\dot{v}_1$ ',
                color=vehicle_colors[2],
                linewidth=linewidth_ucomm,
                zorder=3,
                linestyle='--'
            )



        # Axis settings
        axs[0].set_xlim([-window_duration, 0])
        axs[0].set_ylim([-1.1, 1.1])
        axs[0].set_yticks([-1, -0.5, 0, 0.5, 1])
        axs[0].set_xlabel("time [s]", fontsize=font_size)
        axs[0].set_ylabel(r"[m/s$^2$]", fontsize=font_size)
        axs[0].tick_params(colors='white', labelsize=font_size)

        # One and only legend call
        legend0 = axs[0].legend(
            facecolor='black',
            edgecolor='white',
            loc='upper left',
            fontsize=font_size
        )
        for text in legend0.get_texts():
            text.set_color('white')



        # Plot relative_state_1[1] as white
        if rel_window:
            axs[1].plot(
                [t - current_time for t, _ in rel_window],
                [v for _, v in rel_window],
                color=vehicle_colors[2],
                label=r'$d_2$',
                linewidth = linewidth_ucomm,
            )


        # Axis settings
        axs[1].set_xlim([-window_duration, 0])
        axs[1].set_ylim([-0.1, 1.1])
        axs[1].set_yticks([0, 0.25, 0.5, 0.75 ,1])
        axs[1].set_xlabel("time [s]", fontsize=font_size)
        axs[1].set_ylabel(r"[m]", fontsize=font_size)
        axs[1].tick_params(colors='white', labelsize=font_size)

        # One and only legend call
        legend1 = axs[1].legend(
            facecolor='black',
            edgecolor='white',
            loc='upper left',
            fontsize=font_size
        )
        for text in legend1.get_texts():
            text.set_color('white')






        # Render the frame
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img, (image_width, image_height))

        video_writer.write(img_resized)


    video_writer.release()
    plt.close()
    print("Video saved as:", output_video)
