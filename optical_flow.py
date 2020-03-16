import cv2
import numpy as np
from features import *
from datetime import time, timedelta
import datetime as dt


def get_farneback_flow(img1, img2):
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def get_LK_flow(f1,f2):
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Variable for color to draw optical flow track
    color = (0, 255, 0)
    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    # ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(f1)

    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    # Calculates sparse optical flow by Lucas-Kanade method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    # Selects good feature points for previous position
    good_old = prev[status == 1]
    # Selects good feature points for next position
    good_new = next[status == 1]
    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        # Draws line between new and old position with green color and 2 thickness
        mask = cv2.line(mask, (a, b), (c, d), color, 2)
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        frame = cv2.circle(f2, (a, b), 3, color, -1)
    # Overlays the optical flow tracks on the original frame
    output = cv2.add(frame, mask)
    # Updates previous frame
    prev_gray = gray.copy()
    # Updates previous good feature points
    prev = good_new.reshape(-1, 1, 2)
    # Opens a new window and displays the output frame
    # cv2.imshow("sparse optical flow", output)
    # show_img(output)
    return output

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def generate_next_img_FB(frame1, frame2):
    frame1_g = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_g = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = get_farneback_flow(frame1_g, frame2_g)
    # flow = get_lk_flow(frame1_g, frame2_g)
    return warp_flow(frame2, flow), draw_hsv(flow)

def generate_next_img_LK(frame1, frame2):
    return get_LK_flow(frame1, frame2)


def generate_img_for_cnn(month, day, hour, minute, second, pred_horizon, model='fb'):
        frame_2 = get_full_image_by_date_time(month, day, hour, minute, second)
        a = time(hour=hour, minute=minute,second=second)
        b = (dt.datetime.combine(dt.date(1, 1, 1), a) - dt.timedelta(minutes=pred_horizon)).time()
        frame_1 = get_full_image_by_date_time(month, day, int(b.hour), int(b.minute), int(b.second))

        if len(frame_2) < 5 or len(frame_1) < 5:
            print('Error: ')
            print(month, day, hour, minute)
            return 0
        else:
            if model == 'fb':
                gen3, hsv = generate_next_img_FB(frame_1, frame_2)
            else:
                gen3 = generate_next_img_LK(frame_1, frame_2)
            return pre_process_img(gen3,400)

# f1 = cv2.imread('asi_16124/20190821/20190821120115_11.jpg')
# f2 = cv2.imread('asi_16124/20190821/20190821120130_11.jpg')
# f3 = cv2.imread('asi_16124/20190821/20190821120145_11.jpg')
# #
# #preprocess
# # a = Features()
# opt = OpticalFlow()
#
# frame1 = get_full_image_by_date_time(9,9,12,0,0)
# frame2 = get_full_image_by_date_time(9,9,12,5,0)
# frame3 = get_full_image_by_date_time(9,9,12,10,0)
# print(frame1, frame2)
# gen3, hsv = generate_next_img(frame1, frame2)
# #
# frame1 = pre_process_img(frame1,400)
# frame2 = pre_process_img(frame2,400)
# frame3 = pre_process_img(frame3,400)
# gen3 = pre_process_img(gen3,400)
# hsv = pre_process_img(hsv,400)
#
# show_img(frame1)
# show_img(frame2)
# show_img(frame3)
# show_img(gen3)

# get_LK_flow(frame1, frame2)

# cv2.imshow("orig 1", frame1)
# cv2.imshow("orig 2", frame2)
# cv2.imshow("generated 3", gen3)
# cv2.imshow("orig 3", frame3)
# cv2.imshow("hsv", hsv)
#
# k = cv2.waitKey(0) & 0xff
# 
# cv2.destroyAllWindows()



