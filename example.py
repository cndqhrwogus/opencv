#!/usr/bin/env python
import cv2
import numpy as np

# img = cv2.imread("images/example.jpg")
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
while True:
   _, frame = cap.read()
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
         cv2.line(line_img,(x1,y1),(x2,y2),[0,0,255],5)
   result_img = cv2.addWeighted(line_img,0.8, frame, 1. , 0. )
   cv2.imshow("img",line_img)
   cv2.imshow("grey",result_img)
   if cv2.waitKey(25) & 0xFF == ord('a'):
      exit()

cap.release()
cv2.destroyAllWindows()