import numpy as np
from utils import (read_lms_file, lidar_to_cartesian, transform_to_global, 
                    visualize_occupancy_grid_kinds, create_occupancy_grid,
                    visualize_occupancy_grid_binary)

# 读取激光雷达扫描数据, 并记录相关参数
data = read_lms_file("./data/URG_X_20130903_195003.lms")
ang_rng = data['AngRng']                      # 扫描范围
ang_res = data['AngRes']                      # 角分辨率
unit = data['unit']                           # 单位
thresh = 1                                    # 判定阈值
gsize = 1000                                   # 栅格图边长
grid_size = (gsize, gsize)                    # 栅格图大小
resolution = 0.1                              # 栅格分辨率
data_frames = data['frames']                  # 帧信息
# 读取位姿数据
pose_data_path =  './data/ld.nav'
lidar, tmp = {
    'pose': [], 'time' : [], 'distances' : []
}, 0
with open(pose_data_path, 'r') as file:
    for i, line in enumerate(file):
        # if not line.startswith('time') and not line.startswith(' '):
        if line.startswith('7'):
            parts = line.split()
            while int(parts[0]) > data['frames'][tmp]['timestamp']:
                tmp += 1
            if int(parts[0]) == data['frames'][tmp]['timestamp']:
                yaw = float(parts[3])         # 航向角
                x = float(parts[4])           # 横坐标
                y = float(parts[5])           # 纵坐标
                lidar['pose'].append((x, y, yaw))
                lidar['time'].append(int(parts[0]))
                lidar['distances'].append(data['frames'][tmp]['distances'])
# 初始化以及更新全局栅格地图
global_grid = np.zeros(grid_size, dtype=int)
for i in range(0, len(lidar['pose'])):
    distances = lidar['distances'][i]
    pose = lidar['pose'][i]
    # 激光雷达坐标转换以及读取栅格图坐标命中数，并转换到全局坐标系
    local_points = lidar_to_cartesian(distances, ang_rng, ang_res, unit)
    global_points = transform_to_global(local_points, pose)
    # 创建栅格地图并简易累积投票更新全局栅格地图
    occupancy_grid = create_occupancy_grid(global_points, lidar['pose'], grid_size, resolution)
    global_grid = np.maximum(global_grid, occupancy_grid)
# 可视化栅格地图并记录栅格地图边长
visualize_occupancy_grid_kinds(global_grid, cmap='gray', save_path="grid_map_{}.png".format(gsize))
