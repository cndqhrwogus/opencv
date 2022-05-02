#!/usr/bin/env python
import cv2
import numpy as np
import argparse

def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 3
    blur = cv2.GaussianBlur(img,(kernel, kernel),0)
    canny = cv2.Canny(blur, 40, 120)
    #mask = cv2.inRange(v,70,150)
    return canny

def color_filter(frame):
    #img = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    #h,l,s = cv2.split(img)
    #result = cv2.inRange(h,110,255)
    result = cv2.inRange(frame,(0,0,150),(179,255,255))
    #img_sobel = cv2.Sobel(result,cv2.CV_64F,1,0,5)
    return result

def rectangle_line(img):
    rectangle_img = cv2.rectangle(img,(450,410),(760,480),(255,255,255),3)
    return rectangle_img


def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    # rectangle = np.array([[ 
    # (290, 540),
    # (557, 390),
    # (650, 390),
    # (860, 540),]], np.int32)
    rectangle = np.array([[ #test
    (420, 480),
    (580, 360),
    (640, 360),
    (860, 480),]], np.int32)
    masked = cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(canny, masked)
    return masked_image

def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 1, np.pi/180, 30, 
        np.array([]),minLineLength=10, maxLineGap=200)


def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
 
def display_lines(img,lines):
    print(lines)
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            try:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1,y1),(x2,y2),(0,0,255),2)
            except OverflowError:
                print("error")
            except TypeError:
                print("error")
    return line_image
            

def make_points(image, line):
    slope, intercept = line
    try:
        y1 = 480
        y2 = 360
        # y1 = 490
        # y2 = 470      
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return [[x1, y1, x2, y2]]
    except OverflowError:
        print("don't found line")
 
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        fit = np.polyfit((x1,x2), (y1,y2), 1)
        slope = fit[0]
        intercept = fit[1]
        if not(np.abs(slope)*180<95):
            if slope < 0: 
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if left_fit:
        left_fit_average  = np.average(left_fit, axis=0)
        left_line  = make_points(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
    if left_fit and right_fit:
        averaged_lines = np.array([left_line, right_line])
        return averaged_lines
    else:
        return None

def center_line(image, lines):
    try:
        x1_1, y1_1, x2_1, y2_1 = lines[0].reshape(4)
        x1_2, y1_2, x2_2, y2_2 = lines[1].reshape(4)
        #print(lines)
        # ym1 = y1_2
        # ym2 = y2_2
        ym1 = 480 
        ym2 = 410
    
        xm1 = int((x1_1 + x1_2)/2)
        xm2 = int((x2_1 + x2_2)/2)
        cv2.line(image,
                 (xm1,ym1),
                 (xm2,ym2),
                 color = (255,0,0),
                 thickness = 3)
    except OverflowError:
        print("don't find line")
    except AttributeError:
        print("error")
    except TypeError:
        print("error")
    return image

def warped_image(image):
    # pts =  np.array([[ #top_left,top_right,bottom_left,bottom_right
    # (545,390),
    # (680,390),
    # (290,540),
    # (850,540),]],dtype="float32")
    pts =  np.array([[ #top_left,top_right,bottom_left,bottom_right
    (530,400),#680,400 530,400
    (680,400),
    (300,540),
    (890,540),]],dtype="float32")
    point = np.array([[0,0],[1280,0],[0,720],[1280,720]],dtype="float32")
    M = cv2.getPerspectiveTransform(pts,point)
    warped_img = cv2.warpPerspective(image,M,(1280,720))
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    
    return warped_img

cap = cv2.VideoCapture("test3.mp4")
#cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
while True:
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    lines = houghLines(cropped_canny)
    #print(lines)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    rectangle_img = rectangle_line(line_image)
    center_image = center_line(line_image, averaged_lines)
    combo_image = addWeighted(frame, center_image)
    
    warped_img = warped_image(frame)
    canny_image1 = color_filter(warped_img)
    cv2.imshow("result", canny_image1)
    cv2.imshow("12",warped_img)
    # cv2.imshow("mask",frame)
    #cv2.imshow("img",line_image)
    #cv2.imshow("crop",canny_image)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()

