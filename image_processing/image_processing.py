#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import math 
class lock(Node):
    def __init__(self):
        super().__init__('image_processing')
        self.publisher = self.create_publisher(Float32, 'image_processing_topic', 10)
        self.subscriper = self.create_subscription(Float32, 'image_processing_topic', self.subscriper_callback, 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        First_Number = Float32()
        First_Number.data = 5.0
        self.publisher.publish(First_Number)
   
  
    
    def subscriper_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
  
def main(args=None):
    rclpy.init(args=args)
    node = lock()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()