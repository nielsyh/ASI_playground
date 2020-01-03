import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import os.path


def int_to_str(i):
    s = str(i)
    if(len(s)) == 2:
        return s
    else:
        return '0' + s


def get_image_by_date_time(year, month, day, hour, minute, seconds):
    if (year == 19):
        year = '2019'
    elif (year == 20):
        year = '2020'

    image_not_found = True
    while(image_not_found):
        base_url = 'asi_16124/'
        tmp_url = year + int_to_str(month) + int_to_str(day)
        folder_url = tmp_url + '/'
        img_url = tmp_url + int_to_str(hour) + int_to_str(minute) + int_to_str(seconds) + '_11.jpg'
        total_url = base_url + folder_url + img_url
        if os.path.isfile(total_url):
            image = cv2.imread(total_url)
            image_not_found=False
        else:
            seconds += 15

    return pre_process_img(image, 400)


def extract_features(img):  # get df from image
    features = np.zeros(4)
    features[0] = intensity(img)
    features[1] = number_of_cloud_pixels(img)
    features[2] = harris_corner_detector(img)
    features[3] = edge_detector(img)

    return features


def resize_image(img, dim):
    return cv2.resize(img, (dim, dim), interpolation=cv2.INTER_LINEAR)


def pre_process_img(img, dim):
    y = 26
    x = 26
    h = 1486
    w = 1486
    img = img[y:y + h, x:x + w]
    img = resize_image(img, dim + 20)
    img = circular_crop(img)

    return img

def circular_crop(img):
    x = 10
    y = 10
    r = 200
    # crop image as a square
    img = img[y:y + r * 2, x:x + r * 2]
    # create a mask
    mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)
    # create circle mask, center, radius, fill color, size of the border
    cv2.circle(mask, (r, r), r, (255, 255, 255), -1)
    # get only the inside pixels
    fg = cv2.bitwise_or(img, img, mask=mask)

    mask = cv2.bitwise_not(mask)
    background = np.full(img.shape, 0, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)
    final = cv2.bitwise_or(fg, bk)

    return final

def intensity(img):
    return np.mean(img)

def gauss_denoise(img):
    return ndimage.gaussian_filter(img, 1)


def harris_corner_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    return np.count_nonzero(dst)
    # print(corner_count)

    # cv2.imshow('dst', img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()


def edge_detector(img):
    edges = cv2.Canny(img, img.shape[0], img.shape[1])
    return np.count_nonzero(edges)
    # print(edge_count)
    # cv2.imshow('edges', edges)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()


def number_of_cloud_pixels(img, threshold = 0.8):
    height, width, channels = img.shape

    amount_of_cloud_pixels = 0

    for h in range(0, height):
        for w in range(0, width):
            b = img[h,w,0]
            g = img[h,w,1]
            r = img[h,w,2]

            # if b == 0:
            #     continue
            # rbr = 1 + ((r-b) /b)
            np.seterr(divide='ignore', invalid='ignore')
            rbr = r/b

            # nrbr = (r - b) / (r+b)
            # rb = r - b
            # brgb = (b/r) + (b/g)
            # if brgb < 500:
            # if nrbr > 0.15 and nrbr < 1:

            if rbr > 0.67 and rbr < 1:
                amount_of_cloud_pixels += 1
                img[h, w, 0] = 124
                img[h, w, 1] = 252
                img[h, w, 2] = 0

    # print(amount_of_cloud_pixels)
    # cv2.imshow('image', img)
    # # cv2.imshow('orig', orig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return amount_of_cloud_pixels


class Features:
    def __init__(self):
        pass


# #
# a = Features()
# url = 'asi_16124/20190813/20190813153030_11.jpg'
# # # url = 'test.PNG'
# img = cv2.imread(url)
# img = pre_process_img(img, 400)
#
# cv2.imshow('res', img)
#
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
#
# number_of_cloud_pixels(img)
#
# # cv2.imshow('res', img)
# #
# # if cv2.waitKey(0) & 0xff == 27:
# #     cv2.destroyAllWindows()
#
# # a.number_of_cloud_pixels(img)
# # a.edge_detector(img)
# # a.harris_corner_detector(img)
# # print(intensity(img))