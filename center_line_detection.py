#!/usr/bin/env python
import cv2
import numpy as np

def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny

def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    # triangle = np.array([[
    # (200, height),
    # (800, 350),
    # (1200, height),]], np.int32)
    triangle = np.array([[
    (50, height),
    (320, 110),
    (500, height),]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 1, np.pi/180, 100, 
        np.array([]), minLineLength=40, maxLineGap=5)

def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
 
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            try:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1,y1),(x2,y2),(0,0,255),10)
            except OverflowError:
                print("error")
            except TypeError:
                print("error")
    return line_image
            

def make_points(image, line):
    slope, intercept = line
    #print(line)
    try:
        y1 = int(image.shape[0])
        y2 = int(y1*3.0/5)      
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return [[x1, y1, x2, y2]]
    except OverflowError:
        print("don't found line")
 
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
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
    if lines is None:
        return None
    x1_1, y1_1, x2_1, y2_1 = lines[0].reshape(4)
    x1_2, y1_2, x2_2, y2_2 = lines[1].reshape(4)
    
    ym1 = y1_2
    ym2 = y2_2
    
    xm1 = int((x1_1 + x1_2)/2)
    xm2 = int((x2_1 + x2_2)/2)
    try:
        cv2.line(image,
                 (xm1,ym1),
                 (xm2,ym2),
                 color = (255,255,255),
                 thickness = 10)
    except OverflowError:
        print("don't find line")
    return image

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
while True:
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    lines = houghLines(cropped_canny)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    center_image = center_line(line_image, averaged_lines)
    combo_image = addWeighted(frame, line_image)
    cv2.imshow("result", center_image)
    cv2.imshow("1",line_image)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()

