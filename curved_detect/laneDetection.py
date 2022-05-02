#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from matplotlib import pyplot as plt , cm, colors
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
def processImage(inpImage):
    # mask = cv2.inRange(inpImage,(210,210,190),(240,240,225))
    # hsv = cv2.cvtColor(inpImage,cv2.COLOR_BGR2HSV)
    # h,s,v = cv2.split(hsv)
    
    # yuv = cv2.cvtColor(inpImage,cv2.COLOR_BGR2CMY)
    # y,u,v1 = cv2.split(yuv)
    # result = cv2.bitwise_and(v,v,mask=mask)
    hls = cv2.cvtColor(inpImage,cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(inpImage,cv2.COLOR_BGR2GRAY)
    result = cv2.inRange(hls,(70,150,30),(110,230,130))
    mask = cv2.bitwise_and(result,gray)
    ret, thresh = cv2.threshold(mask,160,255,cv2.THRESH_BINARY)
    # thresh = cv2.inRange(result,210,240)
    blur = cv2.GaussianBlur(thresh,(21, 21), 0)
    canny = cv2.Canny(blur, 40, 60)
    # cv2.imshow("1",result)
    # cv2.imshow("1",mask)
    cv2.imshow("2",thresh)
    # cv2.imshow("mask",thresh)
    # cv2.imshow("2",inpImage)
    # Display the processed images

    return thresh, blur, canny
#### END - FUNCTION TO PROCESS IMAGE ###########################################
################################################################################



################################################################################
#### START - FUNCTION TO APPLY PERSPECTIVE WARP ################################
def perspectiveWarp(inpImage):

    # Get image size
    img_size = (inpImage.shape[1], inpImage.shape[0])

    # Perspective points to be warped (35,170) (280,170) (200,300) (200,10)
    src = np.float32([[10,200],
                      [300,200],
                      [0, 240],
                      [320, 240]])

    # Window to be shown
    dst = np.float32([[0, 0],
                      [320, 0],
                      [0, 240],
                      [320, 240]])

    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(src, dst)
    # Inverse matrix to unwarp the image for final window
    minv = cv2.getPerspectiveTransform(dst, src)
    birdseye = cv2.warpPerspective(inpImage, matrix, img_size)

    # Get the birdseye window dimensions
    height, width = birdseye.shape[:2]

    # Divide the birdseye view into 2 halves to separate left & right lanes
    birdseyeLeft  = birdseye[0:height, 0:width // 2]
    birdseyeRight = birdseye[0:height, width // 2:width]

    # Display birdseye view image
    # cv2.imshow("Birdseye" , birdseye)
    # cv2.imshow("Birdseye Left" , birdseyeLeft)
    # cv2.imshow("Birdseye Right", birdseyeRight)

    return birdseye, birdseyeLeft, birdseyeRight, minv
#### END - FUNCTION TO APPLY PERSPECTIVE WARP ##################################
################################################################################



################################################################################
#### START - FUNCTION TO PLOT THE HISTOGRAM OF WARPED IMAGE ####################
def plotHistogram(inpImage):

    histogram = np.sum(inpImage[inpImage.shape[0] // 2:, :], axis = 0)

    midpoint = np.int(histogram.shape[0] / 2)
    leftxBase = np.argmax(histogram[:midpoint])
    rightxBase = np.argmax(histogram[midpoint:]) + midpoint
    # plt.xlabel("Image X Coordinates")
    # plt.ylabel("Number of White Pixels")
    # plt.plot(histogram)
    # plt.show()
    #Return histogram and x-coordinates of left & right lanes to calculate
    #lane width in pixels
    return histogram, leftxBase, rightxBase
#### END - FUNCTION TO PLOT THE HISTOGRAM OF WARPED IMAGE ######################
################################################################################



################################################################################
#### START - APPLY SLIDING WINDOW METHOD TO DETECT CURVES ######################
def slide_window_search(binary_warped, histogram):
    try:
    # Find the start of left and right lane lines using histogram info
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # A total of 9 windows will be used
        nwindows = 9
        window_height = np.int(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 20
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []
        # print("1",np.sum(histogram[:72]))
        # print("2",np.sum(histogram[248:]))
        # print("3",np.sum(histogram[85:205]))
        if (np.sum(histogram[85:205]) < (np.sum(histogram[:72])+np.sum(histogram[248:]))) and 3000000>(np.sum(histogram[:72])+np.sum(histogram[248:]))>1000000 :
        #### START - Loop to iterate through windows and search for lane lines #####
            for window in range(nwindows):
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                (0,255,0), 2)
                cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high),
                (0,255,0), 2)
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            #### END - Loop to iterate through windows and search for lane lines #######

            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Apply 2nd degree polynomial fit to fit curves
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)


            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            ltx = np.trunc(left_fitx)
            rtx = np.trunc(right_fitx)
        # plt.plot(right_fitx)
            # plt.show()
            # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            # plt.imshow(out_img)
            # plt.plot(left_fitx,  ploty, color = 'yellow')
            # plt.plot(right_fitx, ploty, color = 'yellow')
            # plt.xlim(0, 320)
            # plt.ylim(240, 0)
            # plt.imshow(out_img)
            # plt.show()
            return ploty, left_fit, right_fit, ltx, rtx
    except TypeError:
        return None
#### END - APPLY SLIDING WINDOW METHOD TO DETECT CURVES ########################
################################################################################



################################################################################
#### START - APPLY GENERAL SEARCH METHOD TO DETECT CURVES ######################
def general_search(binary_warped, left_fit, right_fit):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 20
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
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    ## VISUALIZATION ###########################################################

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # plt.imshow(result)
    # plt.plot(left_fitx,  ploty, color = 'yellow')
    # plt.plot(right_fitx, ploty, color = 'yellow')
    # plt.xlim(0, 320)
    # plt.ylim(240, 0)

    ret = {}
    ret['leftx'] = leftx
    ret['rightx'] = rightx
    ret['left_fitx'] = left_fitx
    ret['right_fitx'] = right_fitx
    ret['ploty'] = ploty

    return ret
#### END - APPLY GENERAL SEARCH METHOD TO DETECT CURVES ########################
################################################################################



################################################################################
#### START - FUNCTION TO MEASURE CURVE RADIUS ##################################
def measure_lane_curvature(ploty, leftx, rightx):
    ym_per_pix = 30.0 / 240
    xm_per_pix = 3.7 / 240

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit new polynomials to x, y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    # Decide if it is a left or a right curve
    if leftx[0] - leftx[-1] > 60:
        curve_direction = 'Left Curve'
    elif leftx[-1] - leftx[0] > 60:
        curve_direction = 'Right Curve'
    else:
        curve_direction = 'Straight'
    
    return (left_curverad + right_curverad) / 2.0, curve_direction
#### END - FUNCTION TO MEASURE CURVE RADIUS ####################################
################################################################################



################################################################################
#### START - FUNCTION TO VISUALLY SHOW DETECTED LANES AREA #####################
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
    # cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 0, 255))
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

    return pts_mean, result
#### END - FUNCTION TO VISUALLY SHOW DETECTED LANES AREA #######################
################################################################################


#### START - FUNCTION TO CALCULATE DEVIATION FROM LANE CENTER ##################
################################################################################
def offCenter(meanPts, inpFrame):
    ym_per_pix = 30.0 / 240
    xm_per_pix = 3.7 / 240
    # Calculating deviation in meters
    mpts = meanPts[-1][-1][-2].astype(int)
    pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
    deviation = pixelDeviation * xm_per_pix
    direction = "left" if deviation < 0 else "right"
    return deviation, direction
################################################################################
#### END - FUNCTION TO CALCULATE DEVIATION FROM LANE CENTER ####################



################################################################################
#### START - FUNCTION TO ADD INFO TEXT TO FINAL IMAGE ##########################
def addText(img,direction):

    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(img,direction,(4,20),font,0.5,(0,255,0),1,cv2.LINE_AA)

    return img
#### END - FUNCTION TO ADD INFO TEXT TO FINAL IMAGE ############################
################################################################################

################################################################################
######## END - FUNCTIONS TO PERFORM IMAGE PROCESSING ###########################
################################################################################

################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
######## START - MAIN FUNCTION #################################################
################################################################################

# Read the input image
image = cv2.VideoCapture(0)
image.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
# image.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
# image.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
bridge = CvBridge()
pub = rospy.Publisher('lane_detect',String,queue_size=10)
rospy.init_node('cam_detect')
################################################################################
#### START - LOOP TO PLAY THE INPUT IMAGE ######################################
while True:
    
    _, frame = image.read()
    frame = cv2.resize(frame,(320,240),cv2.INTER_AREA)
    # Apply perspective warping by calling the "perspectiveWarp()" function
    # Then assign it to the variable called (birdView)
    # Provide this function with:
    # 1- an image to apply perspective warping (frame)
    birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)

    # Apply image processing by calling the "processImage()" function
    # Then assign their respective variables (img, hls, grayscale, thresh, blur, canny)
    # Provide this function with:
    # 1- an already perspective warped image to process (birdView)
    thresh, blur, canny = processImage(birdView)
    #imgL, hlsL, grayscaleL, threshL, blurL, cannyL = processImage(birdViewL)
    #imgR, hlsR, grayscaleR, threshR, blurR, cannyR = processImage(birdViewR)

    # cv2.imshow("asd",thresh)
    # Plot and display the histogram by calling the "get_histogram()" function
    # Provide this function with:
    # 1- an image to calculate histogram on (thresh)

    hist, leftBase, rightBase = plotHistogram(thresh)
    # print(rightBase - leftBase)
    #plt.plot(hist)
    # plt.show()
    try:
        ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(thresh, hist)
        # plt.plot(left_fit)
        # plt.show()

        draw_info = general_search(canny, left_fit, right_fit)
        
        # plt.show()
        curveRad, curveDir = measure_lane_curvature(ploty, left_fitx, right_fitx)


        # Filling the area of detected lanes with green
        meanPts, result = draw_lane_lines(frame, thresh, minverse, draw_info)

        
        deviation, directionDev = offCenter(meanPts, frame)
        pub.publish(curveDir)
        # print(deviation)
        # Adding text to our final image
        finalImg = addText(result,curveDir)
        # Displaying final image
        cv2.imshow("Final", finalImg)
    except TypeError:
        if (np.sum(hist[:160]) > 3*np.sum(hist[160:])) and (800000<np.sum(hist)<2000000):
            curveDir = 'only left'
        elif (3*np.sum(hist[:160]) <  np.sum(hist[160:])) and (800000<np.sum(hist)<2000000):
            curveDir = 'only right'
        else:
            curveDir = 'no line'
        finalImg = addText(frame,curveDir)
        cv2.imshow("Final",finalImg)
        pub.publish(curveDir)
    if cv2.waitKey(30) & 0xFF == ord('c'):
            break

    # Wait for the ENTER key to be pressed to stop playback

#### END - LOOP TO PLAY THE INPUT IMAGE ########################################
################################################################################

# Cleanup
image.release()
cv2.destroyAllWindows()

################################################################################
######## END - MAIN FUNCTION ###################################################
################################################################################


































##
