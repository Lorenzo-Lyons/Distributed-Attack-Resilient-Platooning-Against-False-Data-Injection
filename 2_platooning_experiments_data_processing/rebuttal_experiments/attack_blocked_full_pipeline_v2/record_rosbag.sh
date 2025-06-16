#!/bin/bash

# run this in this file's folder in the command line
# ./record_rosbag.sh

# Path to the topic list file
TOPIC_FILE="topics_to_record.txt"

# Check if the file exists
if [[ ! -f "$TOPIC_FILE" ]]; then
  echo "Error: $TOPIC_FILE not found!"
  exit 1
fi

echo "Starting rosbag recording for topics listed in $TOPIC_FILE..."
rosbag record $(cat "$TOPIC_FILE")

