import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message, deserialize_message
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import subprocess
import numpy as np
import time, math, rosbag2_py

from work3.utils import (lidar_to_cartesian, transform_to_global, 
                    visualize_occupancy_grid_kinds, create_occupancy_grid)

# source /opt/ros/humble/setup.bash
# source install/setup.bash

class listener(Node):
    def __init__(self, bag_path):
        super().__init__('lms_publisher')
        # reader
        self.bag_path = bag_path
        self.reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri= self.bag_path,
            storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.reader.open(storage_options, converter_options)

        self.publisher_lms = self.create_publisher(LaserScan, 'lms', 10)
        self.publisher_pose = self.create_publisher(PoseStamped, 'pose', 10)
        # 存储读取到的数据
        self.distances = []
        self.poses = []
        self.times = []
        self.theta_car = []
        # self.i, self.j = 0, 0

        self.timer = self.create_timer(0.5, self.timer_callback)
        # 标记是否已完成读取
        self.read_finished = False
    
    # reader method
    def timer_callback(self):
        # 每次回调读取一条消息（避免阻塞）
        if not self.reader.has_next():
            self.get_logger().info("No more messages to read. Stopping timer.")
            self.timer.cancel()

            self.read_finished = True
            self.work2()
            return
        
        topic, data, _ = self.reader.read_next()
        
        if topic == '/lms':
            # 解析 LaserScan 消息
            scan_msg = deserialize_message(data, LaserScan)
            self.distances.append(scan_msg.ranges)
            # self.i += 1
            # if self.i > 10:
            #     print("a")
            self.get_logger().info(f"Read {len(self.distances)} LaserScan messages")
        elif topic == '/pose':
            # 解析 PoseStamped 消息
            pose_msg = deserialize_message(data, PoseStamped)
            self.poses.append((pose_msg.pose.position.x, pose_msg.pose.position.y))
            self.times.append(pose_msg.header.stamp.sec)
            self.theta_car.append(pose_msg.pose.orientation.w)
            # self.j += 1
            # if self.j > 10:
            #     print("a")
            self.get_logger().info(f"Read {len(self.poses)} Pose messages")
    
    def work2(self):
        gsize = 900                                   # 栅格图边长
        grid_size = (gsize, gsize)                    # 栅格图大小
        resolution = 0.1                              # 栅格分辨率
        ang_rng = 180.0
        ang_res = 0.5
        unit = 100.0

        global_grid = np.zeros(grid_size, dtype=int)
        car_grids = np.zeros(grid_size, dtype=int)
        car_points = self.poses
        for i in range(0, len(self.poses)):
            distances = self.distances[i]
            pose = self.poses[i]
            pose_x, pose_y = pose[0], pose[1]
            pose = (pose_x, pose_y, self.theta_car[i])

            # 激光雷达坐标转换以及读取栅格图坐标命中数，并转换到全局坐标系
            local_points = lidar_to_cartesian(distances, ang_rng, ang_res, unit)
            global_points = transform_to_global(local_points, pose)

            # 创建栅格地图并简易累积投票更新全局栅格地图
            occupancy_grid, car_grid = create_occupancy_grid(global_points, car_points, grid_size, resolution)
            global_grid = np.maximum(global_grid, occupancy_grid)
            car_grids = np.maximum(car_grid, car_grids)
        
        # 可视化栅格地图并记录栅格地图边长
        visualize_occupancy_grid_kinds(global_grid, car_grids, cmap='ocean_r', save_path="grid_map_car_work3_{}.png".format(gsize))


def main(args=None):
    i = 7
    bag_path = "/home/code/lms_bag{}".format(i)
    try:
        rclpy.init(args=args)
        sbr = listener(bag_path)
        rclpy.spin(sbr)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    
if __name__ == '__main__':
    main()