#!/usr/bin/env python

import  cv2
import numpy as np
from std_msgs.msg import String, Float32

def birdview(frame):#(14,200) (300,200)
    img_size = (frame.shape[1], frame.shape[0])
    src = np.float32([[14, 200],[300, 200],[0, 240],[320, 240]])
    # Window to be shown
    dst = np.float32([[0, 0],
                      [320, 0],
                      [0, 240],
                      [320, 240]])

    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(src, dst)
    # Inverse matrix to unwarp the image for final window
    minv = cv2.getPerspectiveTransform(dst, src)
    birdseye = cv2.warpPerspective(frame, matrix, img_size)
    return birdseye
def mask(frame):
    hls = cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
    cv2.imshow("c",hls)
    # result1 = cv2.inRange(hls,(80,190,80),(100,230,255))#(80,200,100) (100,230,150)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    result1 = cv2.inRange(hls,(10,150,10),(100,230,60))
    mask = cv2.bitwise_and(gray,result1)
    ret, thresh = cv2.threshold(mask,170,255,cv2.THRESH_BINARY)
    cv2.imshow("b",mask)
    cv2.imshow("e",thresh)
    return result1
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame,(320,240),cv2.INTER_AREA)
    cv2.imshow("frame",frame)
    birdseye = birdview(frame)
    mask_img = mask(birdseye)
    cv2.imshow("a",mask_img)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
cap.release()
cv2.destroyAllWindow()