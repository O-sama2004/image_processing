#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
import json
from collections import deque
import time


class ImageProcessing(Node):
    def __init__(self):
        super().__init__('Image_Processing_node')

        # Create mission mode subscriber
        self.mission_mode_subscription = self.create_subscription(
            String, 'mission_mode', self.mission_mode_callback, 10)

        # Create video feed subscriber 30 fps
        self.camera_feed_subscription = self.create_subscription(
            Image, 'camera_stream', self.video_listener_callback, 30)
                
        self.bridge = CvBridge()

        # Create Detection feed publisher 30 fps
        self.detection_feed = self.create_publisher(Image, 'detection_stream', 30)

        # Create Detection Coordinates publisher
        self.detection_coordinates = self.create_publisher(Point, 'detection_coordinates', 30)

         # Create QR code message publisher
        self.qr_code_message_publisher = self.create_publisher(String, 'QR_code_message', 10)

        # Load YOLO model
        self.yolo_model = YOLO('weights_openvino_model') # NOTEE: Change model name if needed
        self.get_logger().info('Image Processing node has been started.')

        # Initialize variables for detection tracking
        self.detection_start_time = None
        self.frame_buffer = deque(maxlen=120)  # Assuming 30 FPS, 4 seconds = 120 frames
        self.video_count = 0


    def mission_mode_callback(self, msg):
        self.mission_mode = msg.data
        self.get_logger().info(f'Mission mode set to: {self.mission_mode}')


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

        # Mode Selection
        if self.mission_mode == "dogfight":
            
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

                        # Publish detection coordinates
                        point_msg = Point()
                        point_msg.x = (x1 + x2) / 2  # Center X coordinate
                        point_msg.y = (y1 + y2) / 2  # Center Y coordinate
                        self.detection_coordinates.publish(point_msg)

                        break  # Only need one UAV meeting criteria
                if uav_detected:
                    break

            # Handle detection tracking and video saving
            current_time = time.time()
            self.frame_buffer.append(image)

            if uav_detected:
                if self.detection_start_time is None:
                    self.detection_start_time = current_time
                elif current_time - self.detection_start_time >= 4:
                    self.save_video()
                    self.detection_start_time = None
            else:
                self.detection_start_time = None
                self.frame_buffer.clear()

            # Convert OpenCV image to ROS Image message
            try:
                detection_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
                self.detection_feed.publish(detection_msg)
            except CvBridgeError as e:
                self.get_logger().error('CvBridge Error: {}'.format(e))

            # Display the image with detections on a tab
            # cv2.imshow("Detection", image)
            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     rclpy.shutdown()

        # QR code processing 
        else:
            self.get_logger().info('KAMIKAZE mission mode is active.')
            
            # Initialize the QR Code detector
            qrDecoder = cv2.QRCodeDetector()

            # Process the frame
            # Check if a QR code is detected in the original frame
            data, bbox, _ = qrDecoder.detectAndDecode(image)

            if bbox is not None and data:

                # Calculate the center of the QR code bounding box
                x_center = (bbox[0][0][0] + bbox[0][2][0]) / 2
                y_center = (bbox[0][0][1] + bbox[0][2][1]) / 2
                
                # Publish the coordinates
                point_msg = Point()
                point_msg.x = x_center
                point_msg.y = y_center
                self.detection_coordinates.publish(point_msg)

                # Publish QR code message
                self.qr_code_message_publisher.publish(String(data=data))

            else:
                # If no QR code is detected in the original frame, proceed with processing
                imgContour = image.copy()
                imgThres = self.preProcessing(image)
                biggest = self.getContours(imgThres)

                if biggest.size != 0:
                    imgWarped = self.getWarp(image, biggest, width, height)

                    # Check for QR code in the warped image
                    data, bbox, _ = qrDecoder.detectAndDecode(imgWarped)
                    if bbox is not None and data:

                        # Calculate the center of the QR code bounding box
                        x_center = (bbox[0][0][0] + bbox[0][2][0]) / 2
                        y_center = (bbox[0][0][1] + bbox[0][2][1]) / 2
                        
                        # Publish the coordinates
                        point_msg = Point()
                        point_msg.x = x_center
                        point_msg.y = y_center
                        self.detection_coordinates.publish(point_msg)

                        # Publish QR code message
                        self.qr_code_message_publisher.publish(String(data=data))

                        # Resize the warped image to half its size for display
                        # imgWarpedHalfSize = cv2.resize(imgWarped, (widthImg // 2, heightImg // 2))
                        # cv2.imshow("Final Warped QR Code (Half Size)", imgWarpedHalfSize)
                else:
                    print("No QR code detected!")######################################################################################
            
            pass


    # Function for pre-processing the image - used in QR code processing
    def preProcessing(img):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 200, 200)
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
        imgThres = cv2.erode(imgDial, kernel, iterations=1)

        return imgThres


    # Function to find contours and detect quadrilateral shapes (potential QR code regions)
    def getContours(img):
        biggest = np.array([])
        maxArea = 0
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if area > maxArea and len(approx) == 4:  # Looking for a quadrilateral
                    biggest = approx
                    maxArea = area
        
        return biggest


    # Function to reorder the points to maintain consistency
    def reorder(myPoints):
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), np.int32)
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        return myPointsNew


    # Function to warp the image based on the biggest contour detected
    def getWarp(self, img, biggest, widthImg, heightImg):
        biggest = self.reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        
        return imgOutput


    def save_video(self):
        self.video_count += 1
        filename = f'detection_video_{self.video_count}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30, (self.frame_buffer[0].shape[1], self.frame_buffer[0].shape[0]))

        for frame in self.frame_buffer:
            out.write(frame)

        out.release()
        self.get_logger().info(f'Saved video: {filename}')


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
