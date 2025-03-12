import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

class GlobalMappingWithMarkers(Node):
    def __init__(self):
        super().__init__('global_mapping_with_markers')
        self.bridge = CvBridge()

        # 订阅 RGB 图像和相机内参
        self.rgb_sub = self.create_subscription(
            Image, '/mmk2/head_camera/color/image_raw', self.rgb_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/mmk2/head_camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        # 发布全局环境模型
        self.marker_pub = self.create_publisher(MarkerArray, '/global_environment', 10)

        # 存储数据
        self.rgb_image = None
        self.camera_info = None

        # 全局环境模型
        self.global_markers = []

        # 全局计数器，用于生成唯一的 Marker ID
        self.marker_id_counter = 0

        # 定时器，定期处理数据
        self.timer = self.create_timer(0.1, self.process_data)  # 10Hz

    def rgb_callback(self, msg):
        # 将 RGB 图像转换为 OpenCV 格式
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info("Received RGB image")

    def camera_info_callback(self, msg):
        # 存储相机内参
        self.camera_info = msg
        self.get_logger().info("Received CameraInfo data")

    def process_data(self):
        if self.rgb_image is None or self.camera_info is None:
            self.get_logger().warn("Waiting for RGB image or camera info...")
            return

        # 降低分辨率
        rgb_image = cv2.resize(self.rgb_image, (320, 240))

        # 转换为灰度图像
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # 边缘检测
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)

        # 检测轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建 MarkerArray 消息
        marker_array = MarkerArray()

        for contour in contours:
            # 计算轮廓的 2D 边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 创建 Marker 消息
            marker = Marker()
            marker.header.frame_id = 'map'  # 使用全局坐标系
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = self.marker_id_counter  # 使用全局计数器生成唯一 ID
            self.marker_id_counter += 1  # 递增计数器
            marker.type = Marker.CUBE  # 使用立方体表示边界框
            marker.action = Marker.ADD
            marker.scale.x = w / 100.0  # 边界框宽度（单位：米）
            marker.scale.y = h / 100.0  # 边界框高度（单位：米）
            marker.scale.z = 0.1  # 边界框深度（单位：米）
            marker.pose.position.x = (x + w / 2) / 100.0  # 边界框中心点 X 坐标
            marker.pose.position.y = (y + h / 2) / 100.0  # 边界框中心点 Y 坐标
            marker.pose.position.z = 0.0  # 边界框中心点 Z 坐标
            marker.pose.orientation.w = 1.0
            marker.color.r = 1.0  # 边界框颜色（红色）
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5  # 透明度
            marker.lifetime.sec = 1  # 设置生命周期为 1 秒

            # 将当前 Marker 添加到全局环境模型
            self.global_markers.append(marker)

        # 发布全局环境模型
        marker_array.markers = self.global_markers
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} markers")

def main(args=None):
    rclpy.init(args=args)
    node = GlobalMappingWithMarkers()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()