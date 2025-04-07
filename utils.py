import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# 读取激光雷达lms数据
def read_lms_file(filename):
    # 读取文件头
    with open(filename, 'rb') as f:
        # 解析文件头（3个float，共12字节）
        header_data = f.read(12)
        # 字节序为小端
        AngRng, AngRes, unit = struct.unpack('fff', header_data)
        MAXDATLEN = int(round(AngRng / AngRes)) + 1
    
        # 读取数据帧
        frames = []
        while True:
            # 读取时间戳（4字节long）
            time_header = f.read(4)
            if len(time_header) != 4:
                break
            # 读取数据体
            milli = struct.unpack('i', time_header)[0]        # 小端long
            data_size = MAXDATLEN * 2                         # 每个数据点2字节
            dat_chunk = f.read(data_size)
            
            if len(dat_chunk) != data_size:
                print(f"数据不完整，最后一帧丢弃")
                break
            
            # 解包数据（unsigned short数组）
            distances = struct.unpack(f'{MAXDATLEN}H', dat_chunk)
            
            # 转换为实际距离（米）
            real_distances = [d for d in distances]
            
            frames.append({
                'timestamp': milli,
                'distances': real_distances
            })
    
    return {
        'AngRng': AngRng,
        'AngRes': AngRes,
        'unit': unit,
        'frames': frames
    }


# 将激光雷达的距离数据转换为笛卡尔坐标 (x, y)
def lidar_to_cartesian(distances, ang_rng, ang_res, unit):
    points = []                               # 初始坐标
    start_angle = 0.0                         # 起始角度
    for i, distance in enumerate(distances):
        a = start_angle + i * ang_res         # 当前角度
        angle_rad = math.radians(a)           # 转为弧度
        x = distance / unit * math.cos(angle_rad)
        y = distance / unit * math.sin(angle_rad)
        points.append((x, y))

    return points


# 将局部坐标点转换到全局坐标系
def transform_to_global(points, pose):
    x_car, y_car, theta_car = pose           # 读取位姿
    global_points = []                       # 初始坐标
    for x_local, y_local in points:
        x_global = x_car + x_local * math.cos(theta_car) - y_local * math.sin(theta_car)
        y_global = y_car + x_local * math.sin(theta_car) + y_local * math.cos(theta_car)
        global_points.append((x_global, y_global))
        
    return global_points


# 使用简易投票法创建栅格地图
def create_occupancy_grid(points, grid_size=(55, 55), resolution=0.1):
     # 网格初始化
    width, height = grid_size               
    occupancy_grid = np.zeros((height, width), dtype=int)
    
    # 读取坐标并防止负值错误表达
    for x, y in points:
        grid_x = int(x / resolution)
        grid_y = int(y / resolution)
        if (-1 * width) <= grid_x <= width and (-1 * height) <= grid_y <= height:
            occupancy_grid[grid_x + 300, grid_y + 300] += 1
        
    occupancy_grid = (occupancy_grid > 0).astype(int)
    return occupancy_grid


# 可视化栅格地图
def visualize_occupancy_grid_kinds(occupancy_grid, cmap, save_path=None):
    # 创建图形
    size = 15

    plt.figure(figsize=(size, size))
    plt.imshow(occupancy_grid, cmap=cmap, origin='lower')  # ,origin='lower' 确保原点在左下角
    plt.colorbar(label="Occupancy Value")                  # 添加颜色条
    plt.title("Grid Map")
    plt.xlabel(" X ")
    plt.ylabel(" Y ")
    plt.grid()

    # 显示或保存图像
    if save_path:
        plt.savefig(save_path)
        print(f"栅格地图已保存到 {save_path}")
    plt.show()
    # return size


# 可视化栅格地图
def visualize_occupancy_grid_binary(occupancy_grid, threshold, save_path=None):
    # 创建自定义颜色映射
    cmap = mcolors.ListedColormap(['white', 'black'])
    # 设置阈值筛选
    bounds = [occupancy_grid.min(), threshold, max(threshold, occupancy_grid.max())]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 创建图形
    size = 10
    plt.figure(figsize=(size, size))
    plt.imshow(occupancy_grid, cmap=cmap, norm=norm, origin='lower')  # 使用自定义颜色映射
    plt.title("Grid Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()

    # 显示或保存图像
    if save_path:
        plt.savefig(save_path)
        print(f"栅格地图已保存到 {save_path}")
    plt.show()
