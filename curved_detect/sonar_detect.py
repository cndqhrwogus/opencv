#!/usr/bin/env python
import rospy
from geometry_msgs.msg import TransformStamped, Twist
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, String, Float32
import time
class motor:
    def sonar_callback(self,msg):
        self.sonar0 = msg.range
    def sonar_callback1(self,msg):
        self.sonar1 = msg.range
    def sonar_callback2(self,msg):
        self.sonar2 = msg.range
    def sonar_callback3(self,msg):
        self.sonar3 = msg.range
    def sonar_callback4(self,msg):
        self.sonar4 = msg.range            

    def run(self):
        rospy.init_node("sonar_detect")
        self.sonar0 = 2.0
        self.sonar1 = 2.0
        self.sonar2 = 2.0
        self.sonar3 = 2.0
        self.sonar4 = 2.0
        self.sonarSub0 = rospy.Subscriber("/pi_sonar/sonar_0",Range,self.sonar_callback)
        self.sonarSub0 = rospy.Subscriber("/pi_sonar/sonar_1",Range,self.sonar_callback1)
        self.sonarSub0 = rospy.Subscriber("/pi_sonar/sonar_2",Range,self.sonar_callback2)
        self.sonarSub0 = rospy.Subscriber("/pi_sonar/sonar_3",Range,self.sonar_callback3)
        self.sonarSub0 = rospy.Subscriber("/pi_sonar/sonar_4",Range,self.sonar_callback4)
        Pub = rospy.Publisher("/sonar_range",String,queue_size=3)
        while not rospy.is_shutdown():
            if (self.sonar0 <= 0.3) or (self.sonar2 <= 0.3):
                self.warning = 'RIGHT DETECT'
            elif (self.sonar1<=0.3) or (self.sonar4<=0.3):
                self.warning = 'LEFT DETECT'
            elif self.sonar3 <= 0.3:
                self.warning = 'FRONT DETECT'
            else:
                self.warning = 'SAFY'
            Pub.publish(self.warning)
            time.sleep(0.1)
        rospy.spin()

if __name__ == '__main__':
    node = motor()
    node.run()