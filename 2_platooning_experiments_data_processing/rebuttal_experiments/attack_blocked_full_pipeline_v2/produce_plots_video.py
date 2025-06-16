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
bag_file = '2025-06-04-21-41-50_first_success.bag'
output_video = 'attack_pipeline_plot_animation.mp4'
fps = 5
window_duration = 11  # seconds
image_width, image_height = 1280, 720

# Data containers
data_acc = []
data_ucom = []



vehicle_colors = [
    '#999999',  # Index 0 (unused or placeholder)
    '#f0ad0d',  # Vehicle 1
    '#47e9ff',  # Vehicle 2
    '#37f00d',  # Vehicle 3
    '#f00d0d',  # Vehicle 4
]


linewidth_acc = 5
linewidth_ucomm = 5

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
        elif topic == "/alarm_2":
            all_messages.append((ts, 'alarm', msg.data))


# ✅ Sort globally by timestamp
all_messages.sort(key=lambda x: x[0])

# ✅ Now split back to per-topic lists
data_acc = [(ts, val) for ts, tag, val in all_messages if tag == 'acc']
data_ucom = [(ts, val) for ts, tag, val in all_messages if tag == 'ucom']
data_alarm = [(ts, val) for ts, tag, val in all_messages if tag == 'alarm']


# ✅ Determine global time range
all_timestamps = [ts for ts, _, _ in all_messages]
start_time = min(all_timestamps)
end_time = max(all_timestamps)
duration = end_time - start_time



# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (image_width, image_height))

# Generate frames
fig, axs = plt.subplots(2, 1, figsize=(image_width / 100, image_height / 100), dpi=100, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
ax_main = axs[0]
ax_alarm = axs[1]


times = np.arange(start_time, end_time, 1.0 / fps)

# Set global background
fig.patch.set_facecolor('black')
ax_main.set_facecolor('black')
ax_alarm.set_facecolor('black')

for current_time in times:
    # Collect rolling data
    acc_window = [(t, v) for t, v in data_acc if current_time - window_duration <= t <= current_time]
    ucom_window = [(t, v) for t, v in data_ucom if current_time - window_duration <= t <= current_time]
    alarm_window = [(t, v) for t, v in data_alarm if current_time - window_duration <= t <= current_time]

    # --- Clear both subplots ---
    ax_main.clear()
    ax_alarm.clear()

    # Set black background and white ticks/titles
    # --- Set consistent style ---
    for ax in [ax_main, ax_alarm]:
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.yaxis.set_tick_params(color='white')


    # --- Plotting on the main axis ---
    # Plot acc_saturated_1 in color of vehicle 1
    if acc_window:
        ax_main.plot(
            [t - current_time for t, _ in acc_window],
            [v for _, v in acc_window],
            label=r'real $\dot{v}_1$',
            color=vehicle_colors[1],
            linewidth=linewidth_acc,
            zorder=2
        )
 
    # Plot u_com_1 in color of vehicle 2
    if ucom_window:
        ax_main.plot(
            [t - current_time for t, _ in ucom_window],
            [v for _, v in ucom_window],
            label=r'broadcasted $\dot{v}_1$',
            color=vehicle_colors[2],
            linewidth=linewidth_ucomm,
            zorder=3,
            linestyle='--'
        )


    if alarm_window:
        ax_alarm.plot(
            [t - current_time for t, _ in alarm_window],
            [v for _, v in alarm_window],
            label='Alarm Triggered',
            color='red',
            linewidth=2,
            zorder=2
        )

    # Axis settings
    ax_main.set_xlim([-window_duration, 0])
    ax_main.set_ylim([-1.1, 1.1])
    ax_main.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_main.set_xlabel("time [s]", fontsize=font_size)
    ax_main.set_ylabel(r"[m/s$^2$]", fontsize=font_size)
    ax_main.tick_params(colors='white', labelsize=font_size)


    # Format alarm plot
    ax_alarm.set_xlim([-window_duration, 0])
    ax_alarm.set_ylim([-0.1, 1.1])
    ax_alarm.set_yticks([0, 1])
    ax_alarm.set_xlabel("time [s]", fontsize=font_size)
    ax_alarm.set_ylabel("Alarm", fontsize=font_size)
    ax_alarm.tick_params(colors='white', labelsize=font_size)
    ax_alarm.legend(facecolor='black', edgecolor='white', loc='upper left', fontsize=font_size)




    # One and only legend call
    legend0 = ax_main.legend(
        facecolor='black',
        edgecolor='white',
        loc='upper left',
        fontsize=font_size
    )
    for text in legend0.get_texts():
        text.set_color('white')


    # One and only legend call
    legend1 = ax_alarm.legend(
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
