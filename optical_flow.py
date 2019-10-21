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


f1 = cv2.imread('asi_16124/20190821/20190821074245_11.jpg')
f2 = cv2.imread('asi_16124/20190821/20190821074330_11.jpg')

#preprocess
a = Features()
opt = OpticalFlow()

frame1 = a.pre_process_img(f1, 400)
frame2 = a.pre_process_img(f2, 400)

frame1_g = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2_g = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

flow = opt.get_farneback_flow(frame1_g, frame2_g)
hsv = opt.draw_hsv(flow)
gen2 = opt.warp_flow(frame1, flow)
gen3 = opt.warp_flow(frame2, flow)

cv2.imshow("orig 1", frame1)
cv2.imshow("orig 2", frame2)
cv2.imshow("generated 2", gen2)
cv2.imshow("generated 3", gen3)
cv2.imshow("hsv", hsv)

k = cv2.waitKey(0) & 0xff
# if k == ord('s'): # Change here
#     cv2.imwrite('opticalflow_horz.pgm', horz)
#     cv2.imwrite('opticalflow_vert.pgm', vert)

cv2.destroyAllWindows()



# frame1 = cv2.imread('asi_16124/20190821/20190821074245_11.jpg')
# frame2 = cv2.imread('asi_16124/20190821/20190821074330_11.jpg')
#
# frame1 = a.pre_process_img(frame1, 400)
# frame2 = a.pre_process_img(frame2, 400)
#
# mask = np.zeros_like(frame1)
# mask[..., 1] = 255
#
# prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
# flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)







# cv2.imwrite("/tmp/flow.jpg",hsv)
# cv2.imwrite("/tmp/im1.jpg", im1*255)
# cv2.imwrite("/tmp/im2.jpg", im2*255)
# cv2.imwrite("/tmp/im2w.jpg", im2w)

# # Change here
# horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
# vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
# horz = horz.astype('uint8')
# vert = vert.astype('uint8')
#
# # Change here too
# # cv2.imshow('Horizontal Component', horz)
# # cv2.imshow('Vertical Component', vert)
#
# #new
# # Computes the magnitude and angle of the 2D vectors
# magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
# print(angle)
# mask[..., 0] = angle * 180 / np.pi / 2
# # Sets image value according to the optical flow magnitude (normalized)
# mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
# # Converts HSV to RGB (BGR) color representation
# rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
# # Opens a new window and displays the output frame
# cv2.imshow("dense optical flow", rgb)
#
#
#
#
#
