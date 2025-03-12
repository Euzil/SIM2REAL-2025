import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

class BoundingBoxWithDepth(Node):
    def __init__(self):
        super().__init__('bounding_box_with_depth')
        self.bridge = CvBridge()

        # 订阅 RGB 图像、深度图像和相机内参
        self.rgb_sub = self.create_subscription(
            Image, '/mmk2/head_camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/mmk2/head_camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/mmk2/head_camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        # 发布边界框
        self.marker_pub = self.create_publisher(MarkerArray, '/bounding_boxes', 10)

        # 存储数据
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None

        # 定时器，定期处理数据
        self.timer = self.create_timer(0.1, self.process_data)  # 10Hz

    def rgb_callback(self, msg):
        # 将 RGB 图像转换为 OpenCV 格式
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info("Received RGB image")

    def depth_callback(self, msg):
        # 将深度图像转换为 OpenCV 格式，并转换为米为单位
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_image = self.depth_image.astype(np.float32) / 1000.0  # 转换为米
        self.get_logger().info("Received depth image")

    def camera_info_callback(self, msg):
        # 存储相机内参
        self.camera_info = msg
        self.get_logger().info("Received CameraInfo data")

    def process_data(self):
        if self.rgb_image is None or self.depth_image is None or self.camera_info is None:
            self.get_logger().warn("Waiting for RGB image, depth image, or camera info...")
            return

        # 边缘检测
        gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)

        # 检测轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 获取相机内参
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # 创建 MarkerArray 消息
        marker_array = MarkerArray()

        for i, contour in enumerate(contours):
            # 计算轮廓的 2D 边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 计算 3D 边界框
            depth = self.depth_image[y + h // 2, x + w // 2]  # 取中心点的深度
            if depth == 0:  # 忽略无效点
                continue

            # 将 2D 边界框转换为 3D 坐标
            X_min = (x - cx) * depth / fx
            Y_min = (y - cy) * depth / fy
            X_max = (x + w - cx) * depth / fx
            Y_max = (y + h - cy) * depth / fy
            Z_min = depth
            Z_max = depth + 0.1  # 假设物体高度为 0.1 米

            # 创建 Marker 消息
            marker = Marker()
            marker.header.frame_id = 'camera_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.scale.x = abs(X_max - X_min)
            marker.scale.y = abs(Y_max - Y_min)
            marker.scale.z = abs(Z_max - Z_min)
            marker.pose.position.x = (X_min + X_max) / 2
            marker.pose.position.y = (Y_min + Y_max) / 2
            marker.pose.position.z = (Z_min + Z_max) / 2
            marker.pose.orientation.w = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5  # 透明度
            marker.lifetime.sec = 1  # 显示时间

            marker_array.markers.append(marker)

        # 发布 MarkerArray 消息
        self.marker_pub.publish(marker_array)
        self.get_logger().info("Published bounding boxes")

def main(args=None):
    rclpy.init(args=args)
    node = BoundingBoxWithDepth()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()