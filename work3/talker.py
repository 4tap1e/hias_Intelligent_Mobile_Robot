import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import subprocess
import numpy as np
import time, math, rosbag2_py

from work3.utils import read_lms_file

# source /opt/ros/humble/setup.bash
# source install/setup.bash

class talker(Node):
    def __init__(self, ang_rng, ang_res, unit, pose, time, distances, min, max):
        super().__init__('lms_publisher')
        self.publisher_lms = self.create_publisher(LaserScan, 'lms', 100)
        self.publisher_pose = self.create_publisher(PoseStamped, 'pose', 100)
        self.ang_rng = ang_rng
        self.ang_res = ang_res
        self.unit =  unit
        self.pose = pose
        self.time = time
        self.distances = distances
        self.current_index = 0
        self.timer = self.create_timer(0.5, self.publish_lms)                          # 每 1ms 发布一次
        self.min = min
        self.max = max

    def publish_lms(self):
        if self.current_index < len(self.time):
            # 记录 LaserScan 消息
            scan_msg = LaserScan()
            # 修复时间戳：假设 self.time 为毫秒时间戳
            timestamp_ms = self.time[self.current_index]
            scan_msg.header.stamp.sec = int(timestamp_ms)                               # 记录为lms中的时间戳
            scan_msg.header.stamp.nanosec = int((timestamp_ms % 1000) * 1e3)            # 没有啥用
            scan_msg.header.frame_id = "lms_link"
            scan_msg.range_min = self.min
            scan_msg.range_max = self.max
            scan_msg.ranges = [float(x) for x in self.distances[self.current_index]]    # lms中的distances

            # 记录 PoseStamped 消息
            pose_msg = PoseStamped()
            idx = self.current_index
            pose_msg.header.stamp.sec = int(timestamp_ms)                               # 记录为nav中的时间戳
            pose_msg.header.stamp.nanosec = int((timestamp_ms % 1000) * 1e3)            # 没有啥用 
            pose_msg.header.frame_id = "pose"
            pose_msg.pose.position.x = self.pose[idx][0]                                # x坐标
            pose_msg.pose.position.y = self.pose[idx][1]                                # y坐标
            pose_msg.pose.orientation.w = self.pose[idx][2]                             # 航向角theta_car

            # 发布消息
            # self.current_index += 1
            self.publisher_lms.publish(scan_msg)
            print(f"lms: {self.current_index}/{len(self.time)}")
            self.publisher_pose.publish(pose_msg)
            print(f"pose: {self.current_index}/{len(self.time)}")
            self.current_index += 1
        else:
            self.timer.cancel()

def main(args=None):
    i = 1
    bag_path = "/home/code/lms_bag{}".format(i)
    # jmx
    # lms_path = "/home/work3/data/URG_X_20130903_195003.lms"
    # pose_data_path =  '/home/work3/data/ld.nav'

    # local
    lms_path = "/home/code/数据-20130903/URG_X_20130903_195003.lms"
    pose_data_path =  '/home/code/数据-20130903/ld.nav'

    # 读取 lms 数据
    data_ori = read_lms_file(lms_path)
    ang_rng = data_ori['AngRng']
    ang_res = data_ori['AngRes']
    unit = data_ori['unit']
    data_frames = data_ori['frames']
    lidar, tmp, j = {
        'pose': [], 'time' : [], 'distances' : [], 'max' : [0.0], 'min': [5102.0]
    }, 0, 0
    with open(pose_data_path, 'r') as file:
        for i, line in enumerate(file):
            # if not line.startswith('time') and not line.startswith(' '):
            if line.startswith('7'):
                parts = line.split()
                while int(parts[0]) > data_frames[tmp]['timestamp']:
                    tmp += 1
                if int(parts[0]) == data_frames[tmp]['timestamp']:
                    yaw = float(parts[3])         # 航向角
                    x = float(parts[4])           # 横坐标
                    y = float(parts[5])           # 纵坐标
                    lidar['pose'].append((x, y, yaw))
                    lidar['time'].append(int(parts[0]))
                    lidar['distances'].append(data_frames[tmp]['distances'])
                    j+=1
                    if float(np.array(lidar['max'])) < max(lidar['distances'][j-1]):
                        lidar['max'] =  float(max(lidar['distances'][j-1]))
                    if float(np.array(lidar['min'])) > min(lidar['distances'][j-1]):
                        lidar['min'] =  float(min(lidar['distances'][j-1]))
    pose, t ,distances, r_max ,r_min = lidar['pose'], lidar['time'], lidar['distances'], lidar['max'], lidar['min']
    record_cmd = [
        "ros2", "bag", "record", "/lms", "/pose",
        "-o", bag_path,
        "--storage", "sqlite3"                    # 使用 SQLite 存储格式
    ]
    record_process = subprocess.Popen(record_cmd)

    # 等待记录器启动
    time.sleep(1)

    # 发布 lms 数据到 topic
    rclpy.init(args=args)
    publisher_node = talker(ang_rng, ang_res, unit, pose, t, distances, r_min, r_max)
    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        publisher_node.destroy_node()
        rclpy.try_shutdown()

    # 停止记录器
    record_process.kill()
    record_process.terminate()
    record_process.kill()
    print("Bag 文件生成完成")

if __name__ == '__main__':
    main()