from rosbags.rosbag2 import Reader
from pathlib import Path

bag_path = Path("jun_21_3_vis_0.mcap")

with Reader(bag_path) as reader:
    topics = set()
    for conn in reader.connections:
        topics.add((conn.topic, conn.msgtype))

for topic, msgtype in topics:
    print(topic, "->", msgtype)