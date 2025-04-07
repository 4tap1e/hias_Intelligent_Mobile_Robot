import numpy as np
import math
import struct
import matplotlib.pyplot as plt


def read_lms_file(filename):
    # 读取文件头
    with open(filename, 'rb') as f:
        # 解析文件头（3个float，共12字节）
        header_data = f.read(12)
        if len(header_data) != 12:
            raise ValueError("文件头不完整")
        
        # 字节序为小端
        AngRng, AngRes, unit = struct.unpack('fff', header_data)
        MAXDATLEN = int(round(AngRng / AngRes)) + 1
        
        print(f"扫描参数: 范围{AngRng}° 分辨率{AngRes}° 单位{unit}")
        print(f"每帧数据量: {MAXDATLEN}个点")

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
            real_distances = [d/unit for d in distances]
            
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


def lidar_to_cartesian(distances, ang_rng=180.0, ang_res=1.0, unit=1.0):
    """
    将激光雷达的距离数据转换为笛卡尔坐标 (x, y)。

    参数：
    - distances: 激光雷达的距离数据列表（单位：米）
    - ang_rng: 激光雷达的扫描范围（角度，默认 180°）
    - ang_res: 激光雷达的角分辨率（每步的角度增量，默认 1°）
    - unit: 距离值的单位（默认 1.0）

    返回：
    - points: 转换后的二维坐标点列表 [(x1, y1), (x2, y2), ...]
    """
    points = []
    # start_angle = -ang_rng / 2  # 起始角度
    start_angle = 0 # 起始角度

    for i, distance in enumerate(distances):
        a = start_angle + i * ang_res          # 当前角度（度）
        angle_rad = math.radians(a)            # 转换为弧度
        x = distance / unit * math.cos(angle_rad)
        y = distance / unit * math.sin(angle_rad)
        points.append((x, y))

    return points


def transform_to_global(points, pose):
    """
    将局部坐标点转换到全局坐标系。

    参数：
    - points: 局部坐标点列表 [(x1, y1), (x2, y2), ...]
    - pose: 小车的位姿 (x_car, y_car, theta_car)

    返回：
    - global_points: 全局坐标点列表 [(x1, y1), (x2, y2), ...]
    """
    x_car, y_car, theta_car = pose
    global_points = []
    i = 0
    for x_local, y_local in points:
        x_global = x_car + x_local * math.cos(theta_car) - y_local * math.sin(theta_car)
        y_global = y_car + x_local * math.sin(theta_car) + y_local * math.cos(theta_car)
        global_points.append((x_global, y_global))
        i += 1
    # 转换成矩阵运算 （坐标转换）
        # if i > 200 :
        #     print("a")
    
    # max_x, max_y, tmp = 0.0 , 0.0, 0.0
    # for i in range(0, len(global_points)):
    #     if max_x <= global_points[i][0]:
    #         max_x = global_points[i][0]
    #     if max_y <= global_points[i][1]:
    #         max_y = global_points[i][1]
    #     if max_x > max_y:
    #         max = max_x
    #     else: max = max_y
    # print(f"max is: {max}")
        
    return global_points


def create_occupancy_grid(points, grid_size=(55, 55), resolution=0.1):
    """
    使用简易投票法创建栅格地图。
    参数：
    - points: 全局坐标点列表 [(x1, y1), (x2, y2), ...]
    - grid_size: 栅格地图的大小 (width, height)
    - resolution: 栅格分辨率（每个栅格的实际边长，单位：米）
    返回：
    - occupancy_grid: 栅格地图（二维数组，1 表示有障碍物，0 表示空闲）
    """
    width, height = grid_size
    occupancy_grid = np.zeros((height, width), dtype=int)

    for x, y in points:
        grid_x = int(x / resolution)
        grid_y = int(y / resolution)
        # if 0 <= grid_x < width and 0 <= grid_y < height:
        if (-1 * width) <= grid_x <= width and (-1 * height) <= grid_y <= height:
            occupancy_grid[grid_x + 300, grid_y + 300] += 1
        
    occupancy_grid = (occupancy_grid > 0).astype(int)
    return occupancy_grid


def visualize_occupancy_grid(occupancy_grid, cmap='gray', save_path=None):
    """
    可视化栅格地图。
    参数：
    - occupancy_grid: 栅格地图（二维数组，1 表示障碍物，0 表示空闲）
    - cmap: 颜色映射，默认为 'gray'（灰度图）
    - save_path: 如果需要保存图像，指定保存路径（默认为 None）
    返回：
    - None
    """
    # 创建图形
    size = 15

    plt.figure(figsize=(size, size))
    plt.imshow(occupancy_grid, cmap=cmap, origin='lower')  # ,origin='lower' 确保原点在左下角
    plt.colorbar(label="Occupancy Value")                  # 添加颜色条
    plt.title("Grid Map")
    plt.xlabel("X (cells)")
    plt.ylabel("Y (cells)")
    plt.grid()

    # 保存并显示图像
    if save_path:
        plt.savefig(save_path)
        print(f"栅格地图已保存到 {save_path}")
    plt.show()


def visualize_occupancy_grid_kinds(occupancy_grid, car_grids, cmap, save_path=None):
    # 创建图形
    size = 15
    grid = car_grids * 0.5 + occupancy_grid * 0.5
    plt.figure(figsize=(size, size))
    plt.imshow(grid, cmap=cmap, origin='lower')            # ,origin='lower' 确保原点在左下角
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