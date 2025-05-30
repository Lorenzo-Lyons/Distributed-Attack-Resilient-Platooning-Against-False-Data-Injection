#!/usr/bin/env python
import os
import rosbag
import csv
import math
from collections import defaultdict
from tf.transformations import quaternion_matrix

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

bag_path = 'TEST6.bag'
output_csv = 'vehicles_poses.csv'
vehicle_ids = [1, 2, 3, 4]
reference_frame = 'map'

# Helpers
def transform_to_matrix(trans):
    translation = [trans.translation.x, trans.translation.y, trans.translation.z]
    rotation = [trans.rotation.x, trans.rotation.y, trans.rotation.z, trans.rotation.w]
    matrix = quaternion_matrix(rotation)
    matrix[0:3, 3] = translation
    return matrix

def get_yaw_from_matrix(matrix):
    return math.atan2(matrix[1, 0], matrix[0, 0])

def get_full_transform_chain(child, transforms):
    chain = []
    frame = child
    while frame != reference_frame:
        if frame not in transforms:
            return None
        parent, matrix = transforms[frame]
        chain.append(matrix)
        frame = parent
    full_transform = chain[0]
    for m in chain[1:]:
        full_transform = m.dot(full_transform)
    return full_transform

# Load static TFs
bag = rosbag.Bag(bag_path)
static_transforms = {}
for topic, msg, t in bag.read_messages(topics=['/tf_static']):
    for transform in msg.transforms:
        parent = transform.header.frame_id.strip('/')
        child = transform.child_frame_id.strip('/')
        static_transforms[child] = (parent, transform_to_matrix(transform.transform))

# Process dynamic TFs
rows = []
current_transforms = dict(static_transforms)
bag = rosbag.Bag(bag_path)  # Reopen to restart reading from beginning
for topic, msg, t in bag.read_messages(topics=['/tf']):
    timestamp = msg.transforms[0].header.stamp.to_sec()

    # Update dynamic transforms
    for transform in msg.transforms:
        parent = transform.header.frame_id.strip('/')
        child = transform.child_frame_id.strip('/')
        mat = transform_to_matrix(transform.transform)
        current_transforms[child] = (parent, mat)

    # Record data for all vehicles at this timestamp
    row = [timestamp]
    for i in vehicle_ids:
        target_frame = f'base_link_{i}'
        full_transform = get_full_transform_chain(target_frame, current_transforms)
        if full_transform is not None:
            x, y = full_transform[0:2, 3]
            yaw = get_yaw_from_matrix(full_transform)
            row.extend([x, y, yaw])
        else:
            row.extend([float('nan')] * 3)

    rows.append(row)

bag.close()

# Save CSV
header = ['time']
for i in vehicle_ids:
    header += [f'x{i}', f'y{i}', f'yaw{i}']

print(f"Saving {len(rows)} entries to {output_csv}...")
with open(output_csv, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)


# load the CSV file to check the data
import pandas as pd
import math

# Load the CSV containing all vehicle poses
df = pd.read_csv(output_csv)

# Prepare output
results = []

for idx, row in df.iterrows():
    time = row['time']

    # Skip rows with NaN values
    if row.isnull().any():
        continue

    perceived_front = []

    for v in vehicle_ids:
        xi = row[f'x{v}']
        yi = row[f'y{v}']
        yawi = row[f'yaw{v}']

        min_dist = float('inf')
        front_vehicle = 0

        for other_v in vehicle_ids:
            if other_v == v:
                continue

            xj = row[f'x{other_v}']
            yj = row[f'y{other_v}']

            # Relative position in global frame
            dx = xj - xi
            dy = yj - yi

            # Transform to robot v's frame
            rel_x = math.cos(-yawi) * dx - math.sin(-yawi) * dy

            if rel_x > 0 and rel_x < min_dist:
                min_dist = rel_x
                front_vehicle = other_v

        perceived_front.append(front_vehicle)

    results.append([time] + perceived_front)

# Save to CSV
output_df = pd.DataFrame(results, columns=['time'] + [f'Observed_Predecessor{i}' for i in vehicle_ids])
output_csv_name = 'vehicles_observed_predecessors.csv'
output_df.to_csv(output_csv_name, index=False)

print("Observed predecessors saved to", output_csv_name)




