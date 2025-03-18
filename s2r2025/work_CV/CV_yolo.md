### 1. 安装依赖
确保已安装以下依赖：

YOLO：使用 OpenCV 的 DNN 模块加载 YOLO 模型。

深度相机驱动：例如 librealsense（适用于 Intel RealSense 相机）。

OpenCV 和 NumPy。
#### 安装 OpenCV 和 NumPy
```
pip install opencv-python opencv-python-headless numpy
```

### 2. 创建 ROS 2 包
创建一个新的 ROS 2 包来处理目标检测和坐标定位。
```
# 创建 ROS 2 包
ros2 pkg create --build-type ament_python object_localization
cd object_localization
```

### 3 修改 setup.py
确保 setup.py 正确安装节点。
```python
# object_localization/setup.py

from setuptools import setup

package_name = 'object_localization'

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
    description='Object Localization with YOLO and Depth Camera',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_localization_node = object_localization.object_localization_node:main',
        ],
    },
)
```
### 4. 编译并运行
编译 ROS 2 包并运行节点。
```
# 编译
colcon build --packages-select object_localization
source install/setup.bash

# 运行节点
ros2 run object_localization object_localization_node
```

### 5. 可视化
使用 RViz 可视化物品的世界坐标。

在图像窗口中查看目标检测结果