#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
import json


class ImageProcessing(Node):
    def __init__(self):
        super().__init__('Image_Processing_node')
        self.subscription = self.create_subscription(
            Image, 'camera_stream', self.video_listener_callback, 30)
        
        self.publisher_ = self.create_publisher(Image, 'detection_stream', 30)
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.yolo_model = YOLO('weights_openvino_model') # TO DO: Change model name
        self.get_logger().info('Image Processing node has been started.')


    def video_listener_callback(self, msg: Image):
        try:
            # Convert ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error('CvBridge Error: {}'.format(e))
            return

        height, width, _ = image.shape

        # Calculate yellow square dimensions (10% from top and bottom, 25% from sides)
        yellow_x1 = int(width * 0.25)
        yellow_y1 = int(height * 0.1)
        yellow_x2 = int(width * 0.75)
        yellow_y2 = int(height * 0.9)

        # Draw the yellow box on the image
        cv2.rectangle(image, (yellow_x1, yellow_y1), (yellow_x2, yellow_y2), (0, 255, 255), 2)

        # Run YOLOv8 detection
        results = self.yolo_model(image)

        # Check if any UAV meets the criteria
        uav_detected = False
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = bbox.xyxy[0]  # Bounding box coordinates

                # Calculate width and height of the bounding box
                width_bbox = x2 - x1
                height_bbox = y2 - y1

                # Check if all corners of the bounding box are inside the yellow square
                # and if the width and height of the bounding box are at least 5% of the image dimensions
                if (yellow_x1 <= x1 <= yellow_x2 and yellow_y1 <= y1 <= yellow_y2 and
                    yellow_x1 <= x2 <= yellow_x2 and yellow_y1 <= y2 <= yellow_y2 and
                    width_bbox >= width * 0.06 and height_bbox >= height * 0.06):
                    uav_detected = True

                    # Draw red bounding box
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    break  # Only need one UAV meeting criteria
            if uav_detected:
                break

        # Convert OpenCV image to ROS Image message
        try:
            detection_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            self.publisher_.publish(detection_msg)
        except CvBridgeError as e:
            self.get_logger().error('CvBridge Error: {}'.format(e))


        # Display the image with detections
        cv2.imshow("Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
