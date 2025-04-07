import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 脉冲每到一个周期，则给后续脉冲数变成实际值，可能还要考虑29999， 0的情况
# 读取车轮编码器数据
def read_encoder_data(file_path):
    times = []
    counts = [] 
    time = 0                                              # 圈数
    with open(file_path, 'r') as file:
        # 读取车轮编码器数据
         for line in file:
            if line.startswith('E'):
                parts = line.split() 
                count = int(parts[3]) + 30000 * time 
                times.append(int(parts[1]))               # 毫秒时间
                counts.append(count)                      # 计数 
                if int(parts[3]) == 30000:
                    time += 1 
        
    return np.array(times), np.array(counts)


# 读取IMU数据
def read_imu_data(file_path):
    times = []
    yaws = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('IMU'):
                parts = line.split()
                if int(parts[3]) >= 180:                  # 数据 >=180 时有效
                    times.append(int(parts[1]))           # 毫秒时间
                    yaws.append(float(parts[6]))          # 航向角
    return np.array(times), np.array(yaws)


# 航位推算
def read_reckoning(encoder_times, encoder_counts, imu_times, imu_yaws):
    displacement_per_count = 0.003846154                  # 转换计数为位移(米)
    displacements = np.diff(encoder_counts) * displacement_per_count
    print(f"the first time is: {displacements[0]}")
    
    # 线性插值IMU数据以匹配编码器时间长度
    yaws_interp = np.interp(encoder_times[1:], imu_times, imu_yaws)
    
    # 初始化位置和轨迹坐标
    x, y = 0.0, 0.0
    trajectory = [(x, y)]
    i = 0
    
    # 计算轨迹trajectory
    for disp, yaw in zip(displacements, yaws_interp):
        # 转换为弧度
        yaw_rad = np.radians(yaw)  
        
        # 假定航向角为位移方向与x轴形成的夹角
        dx = disp * np.cos(yaw_rad) 
        dy = disp * np.sin(yaw_rad)
        # dx = disp * np.sin(yaw_rad)
        # dy = disp * np.cos(yaw_rad)
        x += dx
        y += dy
        trajectory.append((x, y))
        
        i += 1
        # 人为在一圈附近直接结束记录轨迹
        if i > 27549:
            break
    
    return np.array(trajectory), i


# 主程序
if __name__ == "__main__":
    # 文件路径(若要复现则请修改此处)
    encoder_file = './data/COMPort_X_20130903_195003.txt' 
    imu_file = './data/InterSense_X_20130903_195003.txt'
    
    # 读取数据
    imu_times, imu_yaws = read_imu_data(imu_file)
    encoder_times, encoder_counts = read_encoder_data(encoder_file)
    
    # 航位推算
    trajectory, i = read_reckoning(encoder_times, encoder_counts, imu_times, imu_yaws)
    
    # 绘制轨迹
    plt.figure(figsize=(10, 10))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='trajectory')
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', label='start')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label='end')
    plt.xlabel('X (meter)')
    plt.ylabel('Y (meter)')
    plt.title('work1 trajectory')
    plt.legend()
    plt.grid(True)
    plt.savefig('output{}.png'.format(i))