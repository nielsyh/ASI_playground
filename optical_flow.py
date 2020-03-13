import cv2
import numpy as np
from features import *
from datetime import time, timedelta
import datetime as dt


def get_farneback_flow(img1, img2):
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

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

def generate_next_img(frame1, frame2):
    frame1_g = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_g = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = get_farneback_flow(frame1_g, frame2_g)
    return warp_flow(frame2, flow), draw_hsv(flow)

def generate_img_for_cnn(month, day, hour, minute, second, pred_horizon):
    try:
        frame_2 = get_full_image_by_date_time(month, day, hour, minute, second)

        a = time(hour=hour, minute=minute,second=second)
        b = (dt.datetime.combine(dt.date(1, 1, 1), a) - dt.timedelta(minutes=pred_horizon)).time()
        frame_1 = get_full_image_by_date_time(month, day, int(b.hour), int(b.minute), int(b.second))

        gen3, hsv = generate_next_img(frame_1, frame_2)
        return pre_process_img(gen3,400)
    except:
        minute = minute + 2
        frame_2 = get_full_image_by_date_time(month, day, hour, minute, second)

        a = time(hour=hour, minute=minute, second=second)
        b = (dt.datetime.combine(dt.date(1, 1, 1), a) - dt.timedelta(minutes=pred_horizon)).time()
        frame_1 = get_full_image_by_date_time(month, day, int(b.hour), int(b.minute), int(b.second))

        gen3, hsv = generate_next_img(frame_1, frame_2)
        return pre_process_img(gen3, 400)


# f1 = cv2.imread('asi_16124/20190821/20190821120115_11.jpg')
# f2 = cv2.imread('asi_16124/20190821/20190821120130_11.jpg')
# f3 = cv2.imread('asi_16124/20190821/20190821120145_11.jpg')
#
# #preprocess
# # a = Features()
# opt = OpticalFlow()
#
# frame1 = get_full_image_by_date_time(9,9,12,0,0)
# frame2 = get_full_image_by_date_time(9,9,12,5,0)
# frame3 = get_full_image_by_date_time(9,9,12,10,0)
# print(frame1, frame2)
# gen3, hsv = opt.generate_next_img(frame1, frame2)
#
#
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

# cv2.imshow("orig 1", frame1)
# cv2.imshow("orig 2", frame2)
# cv2.imshow("generated 3", gen3)
# cv2.imshow("orig 3", frame3)
# cv2.imshow("hsv", hsv)
#
# k = cv2.waitKey(0) & 0xff
# 
# cv2.destroyAllWindows()



