#!/usr/bin/env python
import cv2
import numpy as np
import rospy
import time
from matplotlib import pyplot as plt , cm, colors
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist,TransformStamped


class motor:

    def sonar_callback(self,data):
        self.sonar_warning = data.data

    def opencv_callback(self,data):
        self.camera = data.data

    def run(self):
        rospy.init_node('motor_control')
        self.sonar_warning = 'safy'
        self.camera = 'no line'
        Pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        Sub = rospy.Subscriber('/lane_detect',String,self.opencv_callback)
        sonar_sub = rospy.Subscriber('/sonar_range',String,self.sonar_callback)
        
        while not rospy.is_shutdown():
            if ((self.sonar_warning == 'FRONT DETECT') or (self.sonar_warning =='LEFT DETECT') or (self.sonar_warning =='RIGHT DETECT')):
                while self.sonar_warning != 'safy':
                    forwardspeed = 0.0
                    anglespeed = 0.0
                    twist.angular.z = anglespeed
                    twist.linear.x = forwardspeed
                    Pub.publish(twist)
            else:
                if (self.camera == 'Straight'):
                    forwardspeed = 0.2
                    anglespeed = 0.0
                elif (self.camera == 'Right Curve'):
                    anglespeed = -0.3
                    forwardspeed = 0.1
                elif (self.camera == 'Left Curve'):
                    forwardspeed = 0.1
                    anglespeed = 0.3
                elif (self.camera == 'no line'):
                    forwardspeed = 0.0
                    anglespeed = 0.0
                elif (self.camera == 'only left'):
                    forwardspeed = 0.1
                    anglespeed = 0.3
                elif (self.camera == 'only right'):
                    forwardspeed = 0.1
                    anglespeed = -0.3
            twist = Twist()
            twist.angular.z = anglespeed
            twist.linear.x = forwardspeed
            Pub.publish(twist)
        rospy.spin()


if __name__ == '__main__':
    node = motor()
    node.run()

