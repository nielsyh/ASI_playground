import cv2
import numpy as np
from PIL import Image

class Features:
    def __init__(self):
        pass

    def pre_process_img(self,img, dim):
        y  = 26
        x = 26
        h = 1486
        w = 1486
        img = img[y:y + h, x:x + w]
        img = self.resize_image(img, dim)

        return img

    def resize_image(self,img, dim):
        # setting dim of the resize
        return cv2.resize(img, (dim, dim), interpolation=cv2.INTER_LINEAR)

    def number_of_cloud_pixels(self, img, threshold = 0.8):
        img = self.pre_process_img(img, 400)
        img2 = 0
        img2 = cv2.normalize(img,img2,-1,1,cv2.NORM_MINMAX)
        height, width, channels = img2.shape

        amount_of_cloud_pixels = 0
        r = img2[:,:, 0] #red
        g = img2[:,:, 1] #green
        b = img2[:,:, 2] #blue

        for h in range(0, height):
            for w in range(0, width):
                r = img2[h,w,0]
                g = img2[h,w,1]
                b = img2[h,w,2]

                # if b == 0:
                #     continue
                # rbr = 1 + ((r-b) /b)

                rbr = r/b
                # nrbr = (r - b) / (r+b)
                # rb = r - b
                # brgb = (b/r) + (b/g)

                # if brgb < 500:
                # if nrbr > 0.15 and nrbr < 1:
                if rbr > 0.67 :
                    amount_of_cloud_pixels += 1
                    img[h, w, 0] = 124
                    img[h, w, 1] = 252
                    img[h, w, 2] = 0


        print(amount_of_cloud_pixels)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return amount_of_cloud_pixels

# a = Features()
# # url = 'image/20190723152730_11.jpg'
# url = 'image/20190723151830_11.jpg'
# # url = 'asi_16124/20190813/20190813153015_11.jpg'
# img = cv2.imread(url)
# a.number_of_cloud_pixels(img)