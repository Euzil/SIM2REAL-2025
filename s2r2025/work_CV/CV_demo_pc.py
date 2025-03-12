import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

class EdgeDetectionWithDepth(Node):
    def __init__(self):
        super().__init__('edge_detection_with_depth')
        self.bridge = CvBridge()

        # 订阅 RGB 图像、深度图像和相机内参
        self.rgb_sub = self.create_subscription(
            Image, '/mmk2/head_camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/mmk2/head_camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/mmk2/head_camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        # 发布边缘检测结果和点云
        self.edge_pub = self.create_publisher(Image, '/edge_detection', 10)
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/point_cloud', 10)

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

        # 发布边缘检测结果
        edge_msg = self.bridge.cv2_to_imgmsg(edges, encoding='mono8')
        self.edge_pub.publish(edge_msg)

        # 生成点云
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        height, width = self.depth_image.shape
        points = []
        colors = []

        # 遍历深度图像，生成点云
        for v in range(height):
            for u in range(width):
                Z = self.depth_image[v, u]
                if Z == 0:  # 忽略无效点
                    continue
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                points.append([X, Y, Z])
                r, g, b = self.rgb_image[v, u]
                colors.append((r << 16) | (g << 8) | b)  # 将颜色打包为整数

        # 创建 Header 对象
        header = Header()
        header.stamp = self.get_clock().now().to_msg()  # 设置时间戳
        header.frame_id = 'camera_link'  # 设置参考坐标系

        # 创建 PointCloud2 消息
        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.UINT32, count=1),
        ]
        point_cloud_data = []
        for point, color in zip(points, colors):
            point_cloud_data.append([point[0], point[1], point[2], color])

        pc2_msg = point_cloud2.create_cloud(header, fields, point_cloud_data)
        self.point_cloud_pub.publish(pc2_msg)
        self.get_logger().info("Published edge detection result and point cloud")

def main(args=None):
    rclpy.init(args=args)
    node = EdgeDetectionWithDepth()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()