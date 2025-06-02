import rosbag
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import os

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Configuration
bag_file = 'TEST02_2025-05-28-21-02-44.bag'
output_video = 'full_pipeline_plot_animation.mp4'
fps = 5
window_duration = 11  # seconds
image_width, image_height = 1280, 720



# Data containers
data_acc = []       # /acc_saturated_1
data_acc2 = []      # /acc_saturated_2
data_ucom = []
data_rel = []
data_attack = []        # /attack_detection
data_attack_2 = []      # /attack_detected_2

vehicle_colors = [
    '#999999',  # Index 0 (unused or placeholder)
    '#f0ad0d',  # Vehicle 1
    '#47e9ff',  # Vehicle 2
    '#37f00d',  # Vehicle 3
    '#f00d0d',  # Vehicle 4
]

linewidth_acc = 5
linewidth_ucomm = 5
font_size = 24

# Temporary combined storage
all_messages = []

print("Reading bag file...")
with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        ts = t.to_sec()
        if topic == "/acc_saturated_1":
            all_messages.append((ts, 'acc', msg.data))
        elif topic == "/acc_saturated_2":
            all_messages.append((ts, 'acc2', msg.data))
        elif topic == "/u_com_1":
            all_messages.append((ts, 'ucom', msg.data))
        elif topic == "/relative_state_2":
            if isinstance(msg.data, (list, tuple)) and len(msg.data) > 1:
                all_messages.append((ts, 'rel', msg.data[0]))
        elif topic == "/attack_detection":
            all_messages.append((ts, 'attack', msg.data))
        elif topic == "/attack_detected_2":
            all_messages.append((ts, 'attack2', msg.data))  # ✅ New topic

# ✅ Sort globally by timestamp
all_messages.sort(key=lambda x: x[0])

# ✅ Split back into per-topic lists
data_acc        = [(ts, val) for ts, tag, val in all_messages if tag == 'acc']
data_acc2       = [(ts, val) for ts, tag, val in all_messages if tag == 'acc2']
data_ucom       = [(ts, val) for ts, tag, val in all_messages if tag == 'ucom']
data_rel        = [(ts, val) for ts, tag, val in all_messages if tag == 'rel']
data_attack     = [(ts, val) for ts, tag, val in all_messages if tag == 'attack']
data_attack_2   = [(ts, val) for ts, tag, val in all_messages if tag == 'attack2']  # ✅

# ✅ Determine global time range
all_timestamps = [ts for ts, _, _ in all_messages]
start_time = min(all_timestamps)
end_time = max(all_timestamps)
duration = end_time - start_time

# Parameters
K = 0.05
dt = 0.1
delay = 0.2  # seconds
v_rel_estimate = 0.0

ref_times = np.arange(start_time, end_time, dt)

def get_nearest_value(data_list, t):
    if not data_list:
        return 0.0
    return min(data_list, key=lambda x: abs(x[0] - t))[1]

# Initialize storage
residuals = []
acc1_vals = []
ucom_vals = []

attack_detected_on_car_2 = False  # ✅ Initialize

for t in ref_times:
    acc1 = get_nearest_value(data_acc, t)
    acc2 = get_nearest_value(data_acc2, t)
    raw_rel_data = get_nearest_value(data_rel, t)
    attack = get_nearest_value(data_attack, t)
    attack2 = get_nearest_value(data_attack_2, t)
    ucom = get_nearest_value(data_ucom, t)

    # Processed relative data (adjusted for delay)
    rel_data = raw_rel_data + delay * acc2

    # Predict step
    x_hat_minus = v_rel_estimate + dt * (acc2 - acc1)

    # Update step
    if attack:
        v_rel_estimate = x_hat_minus * (1 - K) + K * rel_data

    if attack2:
        attack_detected_on_car_2 = True

    # Residual
    if attack_detected_on_car_2:
        residual = 0.0
    else:
        residual = abs(rel_data - v_rel_estimate)

    residuals.append((t, residual))
    acc1_vals.append((t, acc1))
    ucom_vals.append((t, ucom))

# Extract times and values
res_times, res_vals = zip(*residuals)
acc1_times, acc1_data = zip(*acc1_vals)
ucom_times, ucom_data = zip(*ucom_vals)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Subplot: acc1 and u_com_1
ax1.plot(acc1_times, acc1_data, label="acc1", color=vehicle_colors[1], linewidth=2)
ax1.plot(ucom_times, ucom_data, label="u_com_1", color=vehicle_colors[3], linewidth=2)
ax1.set_ylabel("Control / Acceleration", fontsize=font_size)
ax1.legend(fontsize=font_size)
ax1.grid(True)
ax1.tick_params(labelsize=font_size)

# Subplot: Residual
ax2.plot(res_times, res_vals, color=vehicle_colors[2], linewidth=2)
ax2.set_xlabel("Time [s]", fontsize=font_size)
ax2.set_ylabel("Residual 2", fontsize=font_size)
ax2.grid(True)
ax2.tick_params(labelsize=font_size)

plt.tight_layout()

















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
            label=r'$\dot{v}_1$ sent to 2',
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
