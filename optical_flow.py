import cv2
import numpy as np
from features import *

#https://github.com/opencv/opencv/issues/11068
class OpticalFlow:

    def __init__(self):
        pass

    def get_farneback_flow(self, img1, img2):
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def draw_hsv(self, flow):
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

    def warp_flow(self, img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res

    def generate_next_img(self, frame1, frame2):
        frame1_g = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_g = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = self.get_farneback_flow(frame1_g, frame2_g)
        return self.warp_flow(frame2, flow), self.draw_hsv(flow)


f1 = cv2.imread('asi_16124/20190821/20190821074245_11.jpg')
f2 = cv2.imread('asi_16124/20190821/20190821074330_11.jpg')

#preprocess
a = Features()
opt = OpticalFlow()

frame1 = a.pre_process_img(f1, 400)
frame2 = a.pre_process_img(f2, 400)
gen3, hsv = opt.generate_next_img(frame1, frame2)

cv2.imshow("orig 1", frame1)
cv2.imshow("orig 2", frame2)
cv2.imshow("generated 3", gen3)
cv2.imshow("hsv", hsv)

k = cv2.waitKey(0) & 0xff

cv2.destroyAllWindows()



