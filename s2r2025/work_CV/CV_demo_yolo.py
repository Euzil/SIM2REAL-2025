# object_localization/object_localization/object_localization_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectLocalizationNode(Node):
    def __init__(self):
        super().__init__('object_localization_node')

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

        # 发布物品的世界坐标
        self.object_position_pub = self.create_publisher(PointStamped, '/object_position', 10)

        # 存储数据
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None

        # 目标物品的类别名称
        self.target_class = "bottle"  # 替换为你的目标物品类别

        # 加载 YOLO 模型
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # 替换为你的 YOLO 模型路径
        with open("coco.names", "r") as f:  # 替换为你的类别文件路径
            self.classes = f.read().strip().split("\n")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def rgb_callback(self, msg):
        # 将 RGB 图像转换为 OpenCV 格式
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        # 将深度图像转换为 OpenCV 格式
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        # 存储相机内参
        self.camera_info = msg

    def detect_objects(self):
        if self.rgb_image is None or self.depth_image is None or self.camera_info is None:
            return None  # 如果没有数据，返回 None

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
        target_position = None
        for i in indices:
            i = i[0]
            class_name = self.classes[class_ids[i]]
            if class_name == self.target_class:  # 只处理目标物品
                box = boxes[i]
                x, y, w, h = box
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
                    x = (center_x - cx) * z / fx
                    y = (center_y - cy) * z / fy

                    # 存储目标物品的位置
                    target_position = (x, y, z)

                    # 在图像上绘制检测结果
                    cv2.rectangle(self.rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(self.rgb_image, f"{class_name} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow("Object Detection", self.rgb_image)
        cv2.waitKey(1)

        return target_position

    def publish_object_position(self, position):
        # 创建 PointStamped 消息
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'camera_link'  # 替换为你的相机坐标系

        if position is not None:
            # 如果检测到目标物品，发布其位置
            point_msg.point.x = position[0]
            point_msg.point.y = position[1]
            point_msg.point.z = position[2]
        else:
            # 如果未检测到目标物品，发布无效值
            point_msg.point.x = float('nan')
            point_msg.point.y = float('nan')
            point_msg.point.z = float('nan')

        # 发布消息
        self.object_position_pub.publish(point_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectLocalizationNode()
    while rclpy.ok():
        rclpy.spin_once(node)
        position = node.detect_objects()
        node.publish_object_position(position)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()