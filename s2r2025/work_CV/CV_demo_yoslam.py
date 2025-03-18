# slam_object_localization/slam_object_localization/slam_object_localization_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class SLAMObjectLocalizationNode(Node):
    def __init__(self):
        super().__init__('slam_object_localization_node')

        # 初始化 CV Bridge
        self.bridge = CvBridge()

        # 订阅 RGB 图像和深度图像
        self.rgb_sub = self.create_subscription(
            Image, '/mmk2/head_camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/mmk2/head_camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # 订阅相机内参
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/mmk2/head_camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        # 发布物品的全局位置
        self.object_positions_pub = self.create_publisher(MarkerArray, '/object_positions', 10)

        # 存储数据
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.camera_pose = None  # 相机在全局坐标系中的位姿

        # 目标物品的类别名称
        self.target_class = "bottle"  # 替换为你的目标物品类别

        # 加载 YOLO 模型
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # 替换为你的 YOLO 模型路径
        with open("coco.names", "r") as f:  # 替换为你的类别文件路径
            self.classes = f.read().strip().split("\n")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # 初始化 ORB-SLAM3
        self.slam = self.initialize_slam()

    def initialize_slam(self):
        # 初始化 ORB-SLAM3
        vocab_path = "/path/to/ORB_SLAM3/Vocabulary/ORBvoc.txt"  # 替换为 ORBvoc.txt 的路径
        config_path = "/path/to/ORB_SLAM3/Examples/Monocular/TUM1.yaml"  # 替换为配置文件路径
        return System(vocab_path, config_path, Sensor.MONOCULAR, True)

    def rgb_callback(self, msg):
        # 将 RGB 图像转换为 OpenCV 格式
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 将图像传递给 ORB-SLAM3
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.slam.TrackMonocular(self.rgb_image, timestamp)

        # 获取相机位姿
        self.camera_pose = self.slam.GetCurrentPose()

    def depth_callback(self, msg):
        # 将深度图像转换为 OpenCV 格式
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        # 存储相机内参
        self.camera_info = msg

    def detect_objects(self):
        if self.rgb_image is None or self.depth_image is None or self.camera_info is None or self.camera_pose is None:
            return []  # 如果没有数据，返回空列表

        # 获取图像尺寸
        height, width, _ = self.rgb_image.shape

        # 使用 YOLO 进行目标检测
        blob = cv2.dnn.blobFromImage(self.rgb_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # 解析检测结果
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # 置信度阈值
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # 处理检测到的物品
        object_positions = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            class_name = self.classes[class_ids[i]]
            confidence = confidences[i]

            # 获取物品中心点的深度
            center_x = x + w // 2
            center_y = y + h // 2
            depth = self.depth_image[center_y, center_x]  # 单位：毫米

            # 将像素坐标转换为相机坐标
            fx = self.camera_info.k[0]  # 相机内参 fx
            fy = self.camera_info.k[4]  # 相机内参 fy
            cx = self.camera_info.k[2]  # 相机内参 cx
            cy = self.camera_info.k[5]  # 相机内参 cy

            if depth > 0:
                # 计算相机坐标系下的坐标
                z = depth / 1000.0  # 转换为米
                x_cam = (center_x - cx) * z / fx
                y_cam = (center_y - cy) * z / fy

                # 将相机坐标系下的坐标转换为全局坐标系
                x_global = self.camera_pose[0, 3] + x_cam
                y_global = self.camera_pose[1, 3] + y_cam
                z_global = self.camera_pose[2, 3] + z

                # 存储物品的位置和类别
                object_positions.append({
                    'class_name': class_name,
                    'position': (x_global, y_global, z_global),
                    'confidence': confidence
                })

                # 在图像上绘制检测结果
                cv2.rectangle(self.rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(self.rgb_image, f"{class_name} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow("Object Detection", self.rgb_image)
        cv2.waitKey(1)

        return object_positions

    def publish_object_positions(self, object_positions):
        # 创建 MarkerArray 消息
        marker_array = MarkerArray()

        if not object_positions:
            # 如果未检测到目标物品，发布一个无效的 Marker
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'  # 使用全局坐标系
            marker.id = 0
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.scale.z = 0.2  # 文本大小
            marker.text = "--"  # 未检测到目标物品
            marker.color.a = 1.0  # 透明度
            marker.color.r = 1.0  # 红色
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.lifetime.sec = 1  # 设置生命周期为 1 秒
            marker_array.markers.append(marker)
        else:
            for idx, obj in enumerate(object_positions):
                # 创建 Marker 消息
                marker = Marker()
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.header.frame_id = 'map'  # 使用全局坐标系
                marker.id = idx
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = obj['position'][0]
                marker.pose.position.y = obj['position'][1]
                marker.pose.position.z = obj['position'][2]
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0  # 透明度
                marker.color.r = 1.0 if obj['class_name'] == self.target_class else 0.0  # 目标物品为红色，其他为绿色
                marker.color.g = 0.0 if obj['class_name'] == self.target_class else 1.0
                marker.color.b = 0.0
                marker.lifetime.sec = 1  # 设置生命周期为 1 秒

                # 将 Marker 添加到 MarkerArray
                marker_array.markers.append(marker)

        # 发布 MarkerArray
        self.object_positions_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = SLAMObjectLocalizationNode()
    while rclpy.ok():
        rclpy.spin_once(node)
        object_positions = node.detect_objects()
        node.publish_object_positions(object_positions)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()