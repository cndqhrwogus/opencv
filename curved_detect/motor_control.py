#!/usr/bin/env python
import cv2
import numpy as np
import rospy
import time
from matplotlib import pyplot as plt , cm, colors
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist,TransformStamped


Pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
def sonar_callback(data):
    global sonar_warning
    sonar_warning = data.data

def callback(data):
    end_time = time.time() + 3
    if sonar_warning == 'FRONT DETECT' or 'LEFT DETECT' or 'RIGHT DETECT':
        forwardspeed = 0.0
        anglespeed = 0.0
    else:
        if data.data == 'Straight':
            forwardspeed = 0.2
            anglespeed = 0.0
        elif data.data == 'Right Curve':
            anglespeed = -0.3
            forwardspeed = 0.1
        elif data.data == 'Left Curve':
            forwardspeed = 0.1
            anglespeed = 0.3
        elif data.data == 'no line':
            forwardspeed = -0.2
            anglespeed = 0.0
        elif data.data == 'only left':
            forwardspeed = 0.1
            anglespeed = 0.3
        elif data.data == 'only right':
            forwardspeed = 0.1
            anglespeed = -0.3
    twist = Twist()
    twist.angular.z = anglespeed
    twist.linear.x = forwardspeed
    if data.data == 'no line':
        while time.time() < end_time:
            Pub.publish(twist)
    else:
        Pub.publish(twist)

def run():
    rospy.init_node('motor_control')
    Sub = rospy.Subscriber('/lane_detect',String,callback)
    sonar_sub = rospy.Subscriber('/sonar_range',String,sonar_callback)
    rospy.spin()

if __name__ == '__main__':
    run()

