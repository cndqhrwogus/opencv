#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from std_msgs.msg import Int64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# img = cv2.imread("images/example.jpg")
bridge = CvBridge()
rospy.init_node("image_pub")
opencv = rospy.Publisher("opencv",Image,queue_size=1)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
rate = rospy.Rate(10)
while True:
   _, frame = cap.read()
   #cv2.imwrite("images/screen_shot.jpg",frame)
   gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   #print(frame.shape)
   canny = cv2.Canny(gray_img,200,250)
   blur = cv2.GaussianBlur(canny, (5,5),0)
   vertices = np.array([[(0,frame.shape[0]), (150,220), (500,220),(frame.shape[1],frame.shape[0])]])
   mask=np.zeros_like(gray_img)
   if len(frame.shape)>2:
      channel_count = frame.shape[2]
      mask_color = (255,) * channel_count
   else:
      mask_color = 255
   cv2.fillPoly(mask,vertices,mask_color)
   mask_img = cv2.bitwise_and(canny,mask)
   lines = cv2.HoughLinesP(mask_img,1,np.pi/180,10,minLineLength=20
   , maxLineGap=10)
   #print(lines)
   line_img = np.zeros((frame.shape[0],frame.shape[1],3),dtype=np.uint8)
   if lines is not None:
      for line in lines:
         x1, y1, x2, y2 = line[0]
         cv2.line(line_img,(x1,y1),(x2,y2),[255,255,255],5)
   result_img = cv2.addWeighted(line_img,0.8, frame, 1. , 0)
   pub_img = bridge.cv2_to_imgmsg(line_img,"bgr8")
   opencv.publish(pub_img)
   #print(lines[0].reshape(4))
   #print(lines[1].reshape(4))
   #text = open("video/text.txt","w")
   cv2.imshow("line",line_img)
   cv2.imshow("result",result_img)
   rate.sleep()
   if cv2.waitKey(27) & 0xFF == ord('a'):
      exit()

cap.release()
cv2.destroyAllWindows()
