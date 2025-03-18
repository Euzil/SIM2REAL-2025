# orb_slam3_ros2/orb_slam3_ros2/orb_slam3_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import sys
import numpy as np

# 导入 ORB-SLAM3
sys.path.append("/path/to/ORB_SLAM3")  # 替换为 ORB-SLAM3 的路径
from ORB_SLAM3 import System, Sensor

class ORBSLAM3Node(Node):
    def __init__(self):
        super().__init__('orb_slam3_node')

        # 初始化 ORB-SLAM3
        vocab_path = "/path/to/ORB_SLAM3/Vocabulary/ORBvoc.txt"  # 替换为 ORBvoc.txt 的路径
        config_path = "/path/to/ORB_SLAM3/Examples/Monocular/TUM1.yaml"  # 替换为配置文件路径
        self.slam = System(vocab_path, config_path, Sensor.MONOCULAR, True)

        # 初始化 CV Bridge
        self.bridge = CvBridge()

        # 订阅 RGB 图像话题
        self.subscription = self.create_subscription(
            Image,
            '/mmk2/head_camera/color/image_raw',  # 替换为你的图像话题
            self.image_callback,
            10)
        self.subscription  # 防止未使用警告

    def image_callback(self, msg):
        # 将 ROS 图像消息转换为 OpenCV 图像
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 将图像传递给 ORB-SLAM3
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.slam.TrackMonocular(cv_image, timestamp)

        # 获取相机位姿
        pose = self.slam.GetCurrentPose()
        if pose is not None:
            self.get_logger().info(f"Current Pose: {pose}")

    def destroy_node(self):
        # 关闭 ORB-SLAM3
        self.slam.Shutdown()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ORBSLAM3Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()