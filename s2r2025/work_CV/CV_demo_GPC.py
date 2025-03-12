import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import cv2

class GlobalMappingWithMotion(Node):
    def __init__(self):
        super().__init__('global_mapping_with_motion')
        self.bridge = CvBridge()

        # 订阅 RGB 图像、深度图像和相机内参
        self.rgb_sub = self.create_subscription(
            Image, '/mmk2/head_camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/mmk2/head_camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/mmk2/head_camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        # 发布全局点云
        self.global_cloud_pub = self.create_publisher(PointCloud2, '/global_point_cloud', 10)

        # 发布运动指令
        self.cmd_vel_pub = self.create_publisher(Twist, '/mmk2/cmd_vel', 10)

        # 存储数据
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None

        # 全局点云
        self.global_cloud = o3d.geometry.PointCloud()

        # 帧计数器
        self.frame_count = 0

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
        self.depth_image = cv2.resize(self.depth_image, (320, 240))  # 降低分辨率
        self.get_logger().info("Received depth image")

    def camera_info_callback(self, msg):
        # 存储相机内参
        self.camera_info = msg
        self.get_logger().info("Received CameraInfo data")

    def process_data(self):
        if self.rgb_image is None or self.depth_image is None or self.camera_info is None:
            self.get_logger().warn("Waiting for RGB image, depth image, or camera info...")
            return

        # 强制交替执行运动和拍照生成
        if self.frame_count % 2 == 0:
            # 偶数帧：执行运动
            self.move_robot()
            self.get_logger().info("Published motion command: rotate")
        else:
            # 奇数帧：执行拍照生成
            self.generate_point_cloud()
            self.get_logger().info("Published global point cloud")

        self.frame_count += 1

    def move_robot(self):
        # 发布运动指令
        twist_msg = Twist()
        twist_msg.linear.x = 0.0  # 前进速度
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0  # 旋转速度
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.5  # 绕 Z 轴旋转
        self.cmd_vel_pub.publish(twist_msg)

    def generate_point_cloud(self):
        # 获取相机内参
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
                colors.append(self.rgb_image[v, u] / 255.0)  # 归一化颜色

        # 转换为 Open3D 点云
        current_cloud = o3d.geometry.PointCloud()
        current_cloud.points = o3d.utility.Vector3dVector(points)
        current_cloud.colors = o3d.utility.Vector3dVector(colors)

        # 体素滤波
        voxel_size = 0.01  # 体素大小（单位：米）
        current_cloud = current_cloud.voxel_down_sample(voxel_size)

        # 将当前点云拼接到全局点云
        if len(self.global_cloud.points) == 0:
            self.global_cloud = current_cloud
        else:
            # 每隔 5 帧进行一次 ICP 配准
            if self.frame_count % 5 == 0:
                transformation = o3d.pipelines.registration.registration_icp(
                    current_cloud, self.global_cloud, max_correspondence_distance=0.05,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
                current_cloud.transform(transformation.transformation)
            self.global_cloud += current_cloud

        # 发布全局点云
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'  # 使用全局坐标系
        points = np.asarray(self.global_cloud.points)
        colors = (np.asarray(self.global_cloud.colors) * 255).astype(np.uint8)
        rgb_colors = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
        point_cloud_data = list(zip(points[:, 0], points[:, 1], points[:, 2], rgb_colors))

        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.UINT32, count=1),
        ]
        pc2_msg = point_cloud2.create_cloud(header, fields, point_cloud_data)
        self.global_cloud_pub.publish(pc2_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GlobalMappingWithMotion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()