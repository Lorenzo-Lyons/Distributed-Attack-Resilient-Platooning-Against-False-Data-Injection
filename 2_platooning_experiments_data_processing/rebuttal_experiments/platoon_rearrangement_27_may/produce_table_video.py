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
font_size = 12
content_font_size = 24
CPP_font_size = 40  # Correct Platooning Position font size


# Only include these topics
tracked_keys = ['overtaking', 'right_indicator', 'topology' ,'vehicle_to_follow', 'v_ref']
# Add this topic for global attack events
attack_topic = '/simulate_attack_detected_on_car'

topics = {key: f'/{key}_{{}}' for key in tracked_keys}

# Data structures
vehicle_states = {v: {} for v in vehicles}
messages = []
detected_attacks = set()

# --------------------
# READ MESSAGES FROM BAG
# --------------------
print("Reading bag file...")


with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        ts = t.to_sec()

        # Handle attack detection
        if topic == attack_topic:
            try:
                attacked_vehicle = int(getattr(msg, 'data', -1))
                detected_attacks.add((ts, attacked_vehicle))
                messages.append((ts, attacked_vehicle, 'attack_detected', True))
            except ValueError:
                continue
            continue  # skip rest of loop for this topic


        for key, template in topics.items():
            for v in vehicles:
                full_topic = template.format(v)
                if topic == full_topic:
                    val = getattr(msg, 'data', str(msg))
                    messages.append((ts, v, key, val))

# Sort messages chronologically
messages.sort(key=lambda x: x[0])


vehicle_colors = ['#999999','#f0ad0d','#47e9ff','#37f00d','#f00d0d']

# --------------------
# RENDERING FUNCTION
# --------------------
def draw_table(states, timestamp):
    fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)
    plt.axis('off')

    # Set background colors
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    plt.axis('off')


    headers = ['Vehicle', 'Attack\ndetected','Lane', 'Merging into\nSlow Lane','Assigned\nPredecessor','Observed\nPredecessor','Vehicle to\nFollow', 'v ref','Correct\nPlatooning\nPosition']
    rows = []
    observed_preds = get_observed_predecessors_at_time(timestamp)

    for idx, v in enumerate(vehicles):
        # Check if an attack is currently detected on this vehicle
        attack_detected = any(abs(ts - timestamp) < 0.1 and v == attacked_v for ts, attacked_v in detected_attacks)
        attack_marker = '⚠' if attack_detected else ''
        row = [f'{v}',attack_marker]

        assigned_pred = None
        observed_pred = observed_preds[idx]
        is_slow_lane = False

        for key in tracked_keys:
            raw = states[v].get(key, None)

            if isinstance(raw, bool) and key == 'right_indicator':
                cell = '→' if raw else ''
            elif isinstance(raw, bool) and key == 'overtaking':
                if raw:  # Fast lane
                    cell = '● | ○'
                    is_slow_lane = False
                else:     # Slow lane
                    cell = '○ | ●' 
                    is_slow_lane = True
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
        row.insert(5, str(observed_pred))

        # Evaluate correct platooning condition
        if is_slow_lane and assigned_pred == observed_pred:
            condition = '✓'
        else:
            condition = '✗'

        # check if the assigned predecessor is 0 and set the attack detected marker to '' if so
        if observed_pred == 0:
            row[1] = ''


        row.append(condition)
        rows.append(row)


    table = plt.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    # Set row height
    cell_height = 0.15
    for pos, cell in table.get_celld().items():
        cell.set_height(cell_height)

        # Set text and background colors
        cell.set_text_props(color='white')
        cell.set_facecolor('black')
        cell.set_edgecolor('white')

        # Set font size only for data cells (not header)
        for (row, col), cell in table.get_celld().items():
            if row > 0:  # Data cells only
                cell.get_text().set_fontsize(content_font_size)  # Or your preferred size




        # Custom vehicle-based text colors
        # Apply dynamic per-cell vehicle color
        for i, _ in enumerate(vehicles):

            # Set color for attack column
            attack_cell = table[(i + 1, 1)]
            if attack_cell.get_text().get_text() == '⚠':
                attack_cell.set_text_props(color='#edba63', fontsize=CPP_font_size)

            for col_idx in [0, 4, 5, 6]:  # Vehicle number, Assigned Pred, Observed Pred, Vehicle to Follow
                cell = table[(i + 1, col_idx)]
                text = cell.get_text().get_text()
                try:
                    vehicle_id = int(text)
                    color = vehicle_colors[vehicle_id]
                    cell.set_text_props(color=color)
                except ValueError:
                    pass  # skip non-integer cells like 'N/A'

            # Color and enlarge the "Correct Platooning Position" checkmark/cross
            correctness_cell = table[(i + 1, 8)]  # Column 8 = last column
            correctness_value = correctness_cell.get_text().get_text()
            if correctness_value == '✓':
                correctness_cell.set_text_props(color='limegreen', fontsize=CPP_font_size)
            elif correctness_value == '✗':
                correctness_cell.set_text_props(color='red', fontsize=CPP_font_size)




    # adding some more visually pleasing styling

    # Title
    plt.title(f"Vehicle States at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}",
              fontsize=16)


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
