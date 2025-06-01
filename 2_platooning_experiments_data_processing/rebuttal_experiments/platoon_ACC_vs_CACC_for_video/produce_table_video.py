import rosbag
import matplotlib.pyplot as plt
import cv2
import numpy as np
from datetime import datetime

# cange the current working directory to the script's directory
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)





# --------------------
# CONFIGURATION
# --------------------
bag_file = '2025-05-28-19-56-28_good.bag'      # ← Change this to your .bag file path
output_video = 'vehicle_distances_table.mp4'
vehicles = [2, 3, 4]
fps = 5                          # Frames per second
image_width, image_height = 1280, 720
font_size = 24
# content_font_size = 24
CPP_font_size = 40  # Correct Platooning Position font size




# Configurations
vehicles = ['2', '3', '4']
tracked_keys = ['relative_state']
comms_topic = '/add_mpc_gamepad'
topics = {key: f'/{key}_{{}}' for key in tracked_keys}

# Data containers
vehicle_states = {v: {} for v in vehicles}
messages = []
comms = set()

print("Reading bag file...")

with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        ts = t.to_sec()

        # Handle communication activation (global)
        if topic == comms_topic:
            comms_activated = getattr(msg, 'data', None)
            if isinstance(comms_activated, bool):  # Sanity check
                comms.add((ts, comms_activated))
                messages.append((ts, 'global', 'comms_active', comms_activated))
            continue

        # Handle relative_state_<vehicle>
        for key, template in topics.items():
            for v in vehicles:
                full_topic = template.format(v)
                if topic == full_topic:
                    val = getattr(msg, 'data', None)
                    if isinstance(val, (list, tuple)) and len(val) > 1:
                        second_val = -val[1]  # Extract second value (distence but also chenge sign)
                        messages.append((ts, v, key, second_val))




# sort in chronological order
messages.sort(key=lambda x: x[0])
comms = sorted(comms, key=lambda x: x[0])




vehicle_colors = ['#999999','#f0ad0d','#47e9ff','#37f00d','#f00d0d']



from matplotlib.patches import Rectangle

def draw_table(states, timestamp):
    fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)
    plt.axis('off')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    headers = ['Vehicle', 'Communication\n(ACC or CACC)', 'Distance to\npredecessor']


    comms_state = False
    if comms:
        closest_event = min(comms, key=lambda x: abs(x[0] - timestamp))
        comms_state = closest_event[1]


    # Prepare data for the table
    rows = []
    distance_data = []

    # Determine control mode per vehicle
    for v in vehicles:
        dist = states[v].get('relative_state', 'N/A')
        dist_str = f"{dist:.2f}" if isinstance(dist, (int, float)) else 'N/A'

        control_mode = '⇆' if comms_state else '×'
        row = [str(v), control_mode, '']  # Leave distance cell empty, we'll draw the bar
        rows.append(row)

        if isinstance(dist, (int, float)):
            distance_data.append((len(rows) - 1, dist_str, dist))





    # Create table
    table = plt.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    cell_height = 0.15
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_height(cell_height)
        cell.set_edgecolor('white')
        cell.set_facecolor('black')

        if row_idx == 0:
            cell.set_text_props(color='white', weight='bold', fontsize=font_size)
        elif col_idx == 0 or col_idx == 1:  # Vehicle or Communication column
            text = cell.get_text().get_text()
            try:
                vehicle_id = int(text)
                color = vehicle_colors[vehicle_id]
                cell.set_text_props(color=color, fontsize=CPP_font_size, weight='bold')
            except (ValueError, IndexError):
                cell.set_text_props(color='white', fontsize=CPP_font_size, weight='bold')
        else:
            cell.set_text_props(color='white', fontsize=font_size)


    # Draw distance bars
    distance_col = 2

    for row_idx, dist_str, dist in distance_data:
        try:
            cell = table[row_idx, distance_col]  # Distance column
        except KeyError:
            continue

        #x, y = cell.get_xy()
        
        w = cell.get_width()
        h = cell.get_height()

        x = (distance_col) * w
        y = (len(rows) - row_idx+0.5) * h

        # Normalize bar width centered at 0.5 (ideal)
        max_dist = 0.8
        min_dist = 0.2
        bar_center = 0.5
        bar_width = (dist - bar_center) / (0.5 * (max_dist - min_dist)) * (0.5 * w) 
        #bar_width = max(min(bar_offset, 0.5), -0.5) * w * 2  # Clamp to cell width

        ax.add_patch(Rectangle(
            (x + w / 2, y + h * 0.125),
            bar_width,
            h * 0.5,
            color=vehicle_colors[int(rows[row_idx][0])] if rows[row_idx][0].isdigit() else '#3fa9f5',
            zorder=3,
        ))

        ax.text(
            x + w / 2,
            y + h * 0.375,
            dist_str,
            ha='center',
            va='center',
            fontsize=font_size - 2,
            color='white',
            zorder=4
        )

    #plt.title(f"Vehicle Distances at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}",
    #         fontsize=16, color='white')

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img









# --------------------
# CREATE VIDEO
# --------------------
print("Creating video...")
video_writer = cv2.VideoWriter(output_video,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (image_width, image_height))

last_frame_time = None
frame_interval = 1.0 / fps

for ts, v, key, val in messages:
    if v in vehicle_states:
        vehicle_states[v][key] = val


    # only draw at the desired frame rate
    if last_frame_time is None or (ts - last_frame_time) >= frame_interval:
        frame = draw_table(vehicle_states, ts)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
        last_frame_time = ts

video_writer.release()
print(f"✅ Video saved as {output_video}")
