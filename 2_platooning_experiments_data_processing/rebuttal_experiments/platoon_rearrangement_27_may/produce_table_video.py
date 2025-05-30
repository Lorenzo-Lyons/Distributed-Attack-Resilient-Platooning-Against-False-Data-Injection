import rosbag
import matplotlib.pyplot as plt
import cv2
import numpy as np
from datetime import datetime

# cange the current working directory to the script's directory
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)



# ---------------------------
# load observed predecessor csv file
# ---------------------------
import pandas as pd
observed_predecessor_csv_name = 'vehicles_observed_predecessors.csv'  # ← Change this to your CSV file path
# Load the CSV containing all vehicle poses
observed_predecessor_csv = pd.read_csv(observed_predecessor_csv_name)
print(f"Loaded {len(observed_predecessor_csv)} rows from {observed_predecessor_csv_name}")

def get_observed_predecessors_at_time(timestamp):
    idx = (observed_predecessor_csv['time'] - timestamp).abs().idxmin()
    row = observed_predecessor_csv.loc[idx]
    return [int(row[f'Observed_Predecessor{v}']) for v in vehicles]








# --------------------
# CONFIGURATION
# --------------------
bag_file = 'TEST6.bag'      # ← Change this to your .bag file path
output_video = 'vehicle_states_table.mp4'
vehicles = [1, 2, 3, 4]
fps = 5                          # Frames per second
image_width, image_height = 1280, 720

# Only include these topics
tracked_keys = ['overtaking', 'right_indicator', 'topology' ,'vehicle_to_follow', 'v_ref']
topics = {key: f'/{key}_{{}}' for key in tracked_keys}

# Data structures
vehicle_states = {v: {} for v in vehicles}
messages = []

# --------------------
# READ MESSAGES FROM BAG
# --------------------
print("Reading bag file...")


with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        ts = t.to_sec()
        for key, template in topics.items():
            for v in vehicles:
                full_topic = template.format(v)
                if topic == full_topic:
                    val = getattr(msg, 'data', str(msg))
                    messages.append((ts, v, key, val))

# Sort messages chronologically
messages.sort(key=lambda x: x[0])

# --------------------
# RENDERING FUNCTION
# --------------------
def draw_table(states, timestamp):
    fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)
    plt.axis('off')

    headers = ['Vehicle', 'Lane', 'Merging into\nSlow Lane','Assigned\nPredecessor','Observed\nPredecessor','Vehicle to\nFollow', 'v ref','Correct\nPlatooning\nPosition']
    rows = []
    observed_preds = get_observed_predecessors_at_time(timestamp)

    for idx, v in enumerate(vehicles):
        row = [f'{v}']

        assigned_pred = None
        observed_pred = observed_preds[idx]
        is_slow_lane = False

        for key in tracked_keys:
            raw = states[v].get(key, None)

            if isinstance(raw, bool) and key == 'right_indicator':
                cell = '✓' if raw else '✗'
            elif isinstance(raw, bool) and key == 'overtaking':
                cell = 'Fast' if raw else 'Slow'
                is_slow_lane = not raw  # slow lane if overtaking is False
            elif key == 'topology' and raw is not None:
                cell = int(raw[v-1])
            elif key == 'vehicle_to_follow' and raw is not None:
                try:
                    assigned_pred = int(raw)
                    cell = str(assigned_pred)
                except (ValueError, TypeError):
                    cell = 'N/A'
            else:
                cell = raw if raw is not None else 'N/A'

            row.append(cell)

        # Insert observed predecessor
        row.insert(4, str(observed_pred))

        # Evaluate correct platooning condition
        if is_slow_lane and assigned_pred == observed_pred:
            condition = '✓'
        else:
            condition = '✗'

        row.append(condition)
        rows.append(row)


    table = plt.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    # Set row height
    cell_height = 0.15
    for pos, cell in table.get_celld().items():
        cell.set_height(cell_height)


    # adding some more visually pleasing styling

    # Title
    plt.title(f"Vehicle States at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}",
              fontsize=16)

    # Draw colored circles around vehicle numbers
    for i, v in enumerate(vehicles):
        # Table rows start at row index 1 (0 is header), col 0 = "Vehicle"
        cell = table[(i + 1, 0)]
        x, y = cell.get_xy()
        width, height = cell.get_width(), cell.get_height()

        # Circle properties
        circle_center = (x + width / 2, y + height / 2)
        radius = min(width, height) / 2.2

        # Choose color for each vehicle (customize as you like)
        color_map = {1: 'skyblue', 2: 'lightgreen', 3: 'salmon', 4: 'orange'}
        color = color_map.get(v, 'gray')

        circle = plt.Circle(circle_center, radius, color=color, zorder=10)
        ax.add_patch(circle)

        # Draw the vehicle number again on top of the circle
        ax.text(circle_center[0], circle_center[1], str(v),
                color='black', ha='center', va='center', fontsize=12, fontweight='bold', zorder=11)

    # Final render
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
    vehicle_states[v][key] = val

    # only draw at the desired frame rate
    if last_frame_time is None or (ts - last_frame_time) >= frame_interval:
        frame = draw_table(vehicle_states, ts)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
        last_frame_time = ts

video_writer.release()
print(f"✅ Video saved as {output_video}")
