# SLAM 
## 代码实现
### 1. 安装依赖
#### 在运行代码之前，确保已安装以下依赖：
ORB-SLAM3：从 ORB-SLAM3 GitHub 克隆并编译。\
ROS 2：确保已安装 ROS 2（如 Foxy 或 Humble）。\
OpenCV 和 Eigen3：ORB-SLAM3 依赖的库。

#### 安装 ORB-SLAM3 依赖
```
sudo apt-get install libopencv-dev libeigen3-dev libboost-all-dev
```

#### 克隆并编译 ORB-SLAM3
```
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ORB_SLAM3 

cd ORB_SLAM3
chmod +x build.sh
./build.sh
```
### 2. 创建 ROS 2 包
创建一个新的 ROS 2 包来集成 ORB-SLAM3。
#### 创建 ROS 2 包
```
ros2 pkg create --build-type ament_python orb_slam3_ros2
cd orb_slam3_ros2
```
### 3.修改 setup.py
确保 setup.py 正确安装节点。
# orb_slam3_ros2/setup.py
```python
from setuptools import setup
package_name = 'orb_slam3_ros2'
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ORB-SLAM3 ROS 2 Integration',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orb_slam3_node = orb_slam3_ros2.orb_slam3_node:main',
        ],
    },
)
```
### 4. 编译并运行
编译 ROS 2 包并运行节点。
```
# 编译
colcon build --packages-select orb_slam3_ros2
source install/setup.bash

# 运行节点
ros2 run orb_slam3_ros2 orb_slam3_node
```
### 5. 可视化
使用 RViz 可视化 ORB-SLAM3 的输出：

添加以下显示：

Camera：显示原始图像。

Path：显示相机轨迹。

PointCloud2：显示地图点

### 1. 实时相机位姿估计
输出：ORB-SLAM3 会实时估计相机的 6 自由度位姿（位置和方向）。

形式：

位姿以 SE(3) 变换矩阵或 [x, y, z, qx, qy, qz, qw]（位置 + 四元数）的形式输出。

在 ROS 2 中，可以通过自定义消息或 geometry_msgs/msg/PoseStamped 发布位姿。

用途：

用于机器人的定位。

可视化相机在环境中的运动轨迹。

### 2. 稀疏地图
输出：ORB-SLAM3 会生成一个稀疏的 3D 点云地图。

形式：

地图由一系列 3D 点组成，每个点表示环境中的一个特征点。

可以通过 sensor_msgs/msg/PointCloud2 发布点云数据。

用途：

用于环境建模。

可用于路径规划或避障。

### 3. 关键帧和共视图
输出：ORB-SLAM3 会生成关键帧和共视图。

形式：

关键帧是 ORB-SLAM3 选择的具有代表性的图像帧。

共视图表示关键帧之间的空间关系。

用途：

用于优化地图和位姿。

提高系统的鲁棒性和精度。

### 4. 轨迹可视化
输出：ORB-SLAM3 会生成相机的运动轨迹。

形式：

轨迹可以通过 nav_msgs/msg/Path 发布。

在 RViz 中可视化轨迹。

用途：

直观展示相机的运动路径。

用于评估 SLAM 系统的性能。

### 5. 实时图像显示
输出：ORB-SLAM3 会实时显示处理后的图像。

形式：

图像中会标记出检测到的特征点。

可以通过 sensor_msgs/msg/Image 发布处理后的图像。

用途：

用于调试和可视化。

直观展示 ORB-SLAM3 的工作状态。

### 6. 日志和调试信息
输出：ORB-SLAM3 会输出日志和调试信息。

形式：

日志信息包括特征点数量、关键帧数量、跟踪状态等。

通过 ROS 2 的日志系统输出。

用途：

用于监控系统状态。

用于调试和优化参数。

### 7. 地图保存与加载
输出：ORB-SLAM3 支持将地图保存到文件。

形式：

地图以二进制文件的形式保存。

可以在后续运行时加载地图。

用途：

用于长期建图。

避免每次运行时重新建图。

### 8. 性能指标
输出：ORB-SLAM3 会输出性能指标。

形式：

包括跟踪时间、地图点数量、关键帧数量等。

通过 ROS 2 的日志系统输出。

用途：

用于评估系统性能。

用于优化参数和算法。

### 9. 可视化效果
在 RViz 中，预计可以看到以下可视化效果：

相机轨迹：一条连续的路径，表示相机的运动轨迹。

稀疏点云：一系列 3D 点，表示环境中的特征点。

关键帧：关键帧的位置和方向。

实时图像：处理后的图像，标记出特征点。