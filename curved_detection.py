#!/usr/bin/env python
import numpy as np
import cv2
import os 
from matplotlib import pyplot as plt , cm, colors
cap = cv2.VideoCapture('test3.mp4')
def color_thresh(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        #abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        #scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        thresh_min = 10
        thresh_max = 200
        #sxbinary = np.zeros_like(scaled_sobel)
        #sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        s_binary = np.zeros_like(hsv)
        s_binary = cv2.inRange(hsv,(0,0,100),(179,255,255))
        mask = cv2.bitwise_and(frame,frame,mask=s_binary)
        gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
        #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        #combined_binary = np.zeros_like(sxbinary)
        #combined_binary[(s_binary == 255) | (sxbinary == 1)] = 255
        blur = cv2.GaussianBlur(thresh,(3,3),0)
        canny = cv2.Canny(blur,30,90)
        cv2.imshow("zxvzxv",canny)
        return canny
def warped_image(frame):
    img_size = (frame.shape[1],frame.shape[0])
    src =  np.array([[ #top_left,top_right,bottom_left,bottom_right
    (560,380),#380,650 380,560
    (650,380),
    (300,540),
    (890,540),]],dtype="float32")
    dst = np.array([[0,0],
                    [1280,0],
                    [0,720],
                    [1280,720]],dtype="float32")
    matrix = cv2.getPerspectiveTransform(src,dst)
    minv = cv2.getPerspectiveTransform(dst,src)
    birdeye = cv2.warpPerspective(frame,matrix,img_size)
    height,width = birdeye.shape[:2]
    birdeyeleft = birdeye[0:height,0:width//2]
    birdeyeright = birdeye[0:height,width//2:width]
    return birdeye,birdeyeleft,birdeyeright,minv
    
    
def plotHistogram(image):
    histogram = np.sum(image[image.shape[0]//2:,:],axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    left = np.argmax(histogram[:midpoint])
    right = np.argmax(histogram[midpoint:])+midpoint
    plt.xlim(0,1280)
    plt.ylim(0,720)
    plt.xlabel("Image X coordinates")
    plt.ylabel("number of white pixels")
    # plt.plot(histogram)
    # plt.show()
    return histogram
def slide_window(img,histogram):
    out_img = np.dstack((img,img,img))*255
    midpoint = np.int(histogram.shape[0]//2)
    left = np.argmax(histogram[:midpoint]) #~640
    right = np.argmax(histogram[midpoint:])+midpoint # 640~
    nwindows = 7
    window_height = np.int(img.shape[0]/nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_current = left
    right_current = right
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    for window in range(nwindows):
        win_y_low = img.shape[0]-(window+1)*window_height
        win_y_high = img.shape[0]-window*window_height
        win_left_low = left_current - margin #70
        win_left_high = left_current + margin #270
        win_right_low = right_current - margin #1001
        win_right_high = right_current + margin #1201
        cv2.rectangle(out_img,(win_left_low,win_y_low),(win_left_high,win_y_high),(0,255,0),2) 
        cv2.rectangle(out_img,(win_right_low,win_y_low),(win_right_high,win_y_high),(0,255,0),2)
        left_inds = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_left_low)&(nonzerox<win_right_high)).nonzero()[0]
        right_inds = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_right_low)&(nonzerox<win_right_high)).nonzero()[0]
        left_lane.append(left_inds)
        right_lane.append(right_inds)
        if len(left_inds)>minpix:
            left_current = np.int(np.mean(nonzerox[left_inds]))
        if len(right_inds)>minpix:
            right_current = np.int(np.mean(nonzerox[left_inds]))
    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)
    leftx = nonzerox[left_lane]
    lefty = nonzeroy[left_lane]
    rightx = nonzerox[right_lane]
    righty = nonzeroy[right_lane]
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    ploty = np.linspace(0,img.shape[0]-1,img.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty+left_fit[2]
        right_fitx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
    except TypeError:
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    ltx = np.trunc(left_fitx)
    rtx = np.trunc(right_fitx)
    out_img[lefty,leftx]=[255,0,0]
    out_img[righty,rightx]=[0,0,255]
    plt.plot(left_fitx,ploty,color = 'yellow')
    plt.plot(right_fitx,ploty,color='yellow')
    plt.xlim(0,1280)
    plt.ylim(720,0)
    plt.show()
    plt.imshow(out_img)
    return ploty,left_fit,right_fit,ltx,rtx
def general_search(img,left_fit,right_fit):
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img = np.dstack((img,img,img))*255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    #plt.plot(left_fitx,  ploty, color = 'yellow')
    #plt.plot(right_fitx, ploty, color = 'yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)

    ret = {}
    ret['leftx'] = leftx
    ret['rightx'] = rightx
    ret['left_fitx'] = left_fitx
    ret['right_fitx'] = right_fitx
    ret['ploty'] = ploty

    return ret
def measure_lane_curvature(ploty, leftx, rightx):
    ym_per_pix = 30.0/720
    xm_per_pix = 3.7/700
    leftx = leftx[::-1]  
    rightx = rightx[::-1]  
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    if leftx[0] - leftx[-1] > 60:
        curve_direction = 'Left Curve'
    elif leftx[-1] - leftx[0] > 60:
        curve_direction = 'Right Curve'
    else:
        curve_direction = 'Straight'
    return (left_curverad + right_curverad) / 2.0, curve_direction
def draw_lane_lines(original_image, warped_image, Minv, draw_info):

    leftx = draw_info['leftx']
    rightx = draw_info['rightx']
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 255, 255))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    return pts_mean, result
def offCenter(meanPts, inpFrame):
    ym_per_pix = 30.0/720
    xm_per_pix = 3.7/720
    mpts = meanPts[-1][-1][-2].astype(int)
    pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
    deviation = pixelDeviation * xm_per_pix
    direction = "left" if deviation < 0 else "right"

    return deviation, direction
def addText(img, radius, direction, deviation, devDirection):
    font = cv2.FONT_HERSHEY_TRIPLEX

    if (direction != 'Straight'):
        text = 'Radius of Curvature: ' + '{:04.0f}'.format(radius) + 'm'
        text1 = 'Curve Direction: ' + (direction)

    else:
        text = 'Radius of Curvature: ' + 'N/A'
        text1 = 'Curve Direction: ' + (direction)

    cv2.putText(img, text , (50,100), font, 0.8, (0,100, 200), 2, cv2.LINE_AA)
    cv2.putText(img, text1, (50,150), font, 0.8, (0,100, 200), 2, cv2.LINE_AA)

    deviation_text = 'Off Center: ' + str(round(abs(deviation), 3)) + 'm' + ' to the ' + devDirection
    cv2.putText(img, deviation_text, (50, 200), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,100, 200), 2, cv2.LINE_AA)

    return img


def main():
    while True:
        _,frame = cap.read()
        birdview, birdviewL, birdviewR, minverse = warped_image(frame)
        thresh = color_thresh(birdview)
        # hlsL,grayL,threshL,blurL,cannyL = color_thresh(birdviewL)
        # hlsR,grayR,threshR,blurR,cannyR = color_thresh(birdviewR)
        hist = plotHistogram(thresh)
        ploty,left_fit,right_fit,left_fitx,right_fitx=slide_window(thresh,hist)
        draw = general_search(thresh,left_fit,right_fit)
        curveRad, curveDir = measure_lane_curvature(ploty,left_fitx,right_fitx)
        meanPts, result = draw_lane_lines(frame,thresh,minverse,draw)
        deviation, directionDev = offCenter(meanPts, frame)
        result_img = addText(result,curveRad,curveDir,deviation,directionDev)
        #plt.plot(hist)
        #cv2.imshow("result",result_img)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

if __name__ == '__main__':
    main()