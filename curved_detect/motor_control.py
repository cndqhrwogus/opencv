#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Float32, Bool
from geometry_msgs.msg import Twist,TransformStamped
from math import pi,sqrt,atan2
import traceback
import math
import time
import signal
import sys
def degrees(r):
    return 180.0 * r/math.pi
def signal_handler(sig,frame):
    print(' ')
    print('Exit due to detection of Ctrl+C')
    sys.exit(0)
class motor_control:
    
    def callback(data):
        line = data
        return line
    
    def motor_power_callback(self,msg):
        self.motor_power_on = msg.data
        
    def __init__(self):
        global callback
        rospy.init_node('motor_control')
        self.debug_prints = 1
        signal.signal(signal.SIGINT,signal_handler)
        self.startTime = rospy.Time.now()
        self.cmdPub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        self.laneDetectSub = rospy.Subscriber('/lane_detect',Float32,self.callback)
        self.motorPowerSub = rospy.Subscriber('/motor_power_active',Bool,self.motor_power_callback)
        self.suppressCmd = False
        self.motor_power_on = True
        self.motors_enabled = True
        self.linear_rate = rospy.get_param("~linear_rate",0.2)
        self.angular_rate = rospy.get_param("~angular_rate",0.7)
    def run(self):
        loopRateSec = 0.035
        rate = rospy.Rate(1/loopRateSec)
        linSpeed = 0.0
        angSpeed = 0.0
        times_since_last_fid = 0
        rotation_duration = 0
        self.motors_enabled = True
        while not rospy.is_shutdown():
            if self.motor_power_on == False:
                if self.motors_enabled == True:
                    print("motor power off, Try motor turn on")
                self.motors_enabled = False
            else:
                if self.motors_enabled == False:
                    print("MOtor power has been re-enabled")
                self.motors_enabled == True
            if rotation_duration > 0.0:
                rotation_duration -= loopRateSec
            else:
                line = self.laneDetectSub
                print(type(line))
                if self.motors_enabled == False:
                    rate.sleep()
                elif 0.9>self.laneDetectSub > 0.1:
                    angSpeed = self.angular_rate
                    linSpeed = self.linear_rate
                    rotation_duration = (3.14/(abs(self.laneDetectSub*10)))/self.angular_rate
                elif -0.9<self.laneDetectSub < -0.1:
                    angSpeed = -1.0 * self.angular_rate
                    linSpeed = self.linear_rate
                    rotation_duration = (3.14/(abs(self.laneDetectSub*10)))/self.angular_rate
                else:
                    angSpeed = 0.0
                    linSpeed = self.linear_rate
                    rotation_duration = 0.0
            if self.motors_enabled == True:
                print("Speeds:lin %f ang %f"%(linSpeed,angSpeed))
            zeroSpeed = (angSpeed == 0 and linSpeed == 0)
            if not zeroSpeed:
                self.suppressCmd = False
            if not self.suppressCmd:
                twist = Twist()
                twist.angular.z = angSpeed
                twist.linear.x = linSpeed
                self.cmdPub.publish(twist)
                if zeroSpeed:
                    self.suppressCmd = True
            rate.sleep()
                
       


if __name__ == '__main__':
    node = motor_control()
    node.run()
    