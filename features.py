import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import os.path
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


def getColor_racket(N, idx):
    import matplotlib as mpl
    c = 'jet'
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))


def int_to_str(i):
    s = str(i)
    if(len(s)) == 2:
        return s
    else:
        return '0' + s

def cloud_pixel_feature(img, plot=False):
    p1, p2, p3, p4, p5, p6, p7 = 0, 0, 0, 0, 0, 0, 0 # distance 20, 40, 60 ,80, 100 etc
    loc = get_sun_cor_by_img(img)
    th = 0.8
    copy = np.copy(img)
    height, width, channels = img.shape
    for h in range(0, height):
        for w in range(0, width):
            r = copy[h, w, 0]
            b = copy[h, w, 1]
            g = copy[h, w, 2]
            np.seterr(divide='ignore', invalid='ignore')
            rbr = r / b
            if rbr > th and rbr < 1:
                d = get_distance(loc[0], w, loc[1], h)

                if d < 20:
                    p1 += 1
                    copy[h, w, 0] = 214
                    copy[h, w, 1] = 70
                    copy[h, w, 2] = 17
                elif d < 40:
                    p2 += 1
                    copy[h, w, 0] = 214
                    copy[h, w, 1] = 175
                    copy[h, w, 2] = 17
                elif d < 60:
                    p3 += 1
                    copy[h, w, 0] = 161
                    copy[h, w, 1] = 214
                    copy[h, w, 2] = 17
                elif d < 80:
                    p4 += 1
                    copy[h, w, 0] = 17
                    copy[h, w, 1] = 214
                    copy[h, w, 2] = 89
                elif d < 100:
                    p5 += 1
                    copy[h, w, 0] = 125
                    copy[h, w, 1] = 17
                    copy[h, w, 2] = 214
                elif d < 120:
                    p6 += 1
                    copy[h, w, 0] = 98
                    copy[h, w, 1] = 140
                    copy[h, w, 2] = 99
                elif d < 140:
                    p7 += 1
                    copy[h, w, 0] = 95
                    copy[h, w, 1] = 105
                    copy[h, w, 2] = 70
    if plot:
        clr = (255,255,255)
        copy = cv2.circle(copy, loc, 20, clr, 0)
        copy = cv2.circle(copy, loc, 40, clr, 0)
        copy = cv2.circle(copy, loc, 60, clr, 0)
        copy = cv2.circle(copy, loc, 80, clr, 0)
        copy = cv2.circle(copy, loc, 100, clr, 0)
        copy = cv2.circle(copy, loc, 120, clr, 0)
        copy = cv2.circle(copy, loc, 140, clr, 0)
        show_img(copy)
    return p1, p2, p3, p4, p5, p6, p7

def get_full_image_by_date_time(month, day, hour, minute, seconds):
    # seconds_list = ['0', '15', '30', '45']
    seconds_list = [seconds]

    base_url = 'asi_16124/'
    tmp_url = '2019' + int_to_str(month) + int_to_str(day)
    base_url = base_url + tmp_url + '/'  # folder
    found_img = False
    while not found_img:
        print('Trying: ')
        print(month, day, hour, minute)
        for s in seconds_list:
            img_url = tmp_url + int_to_str(hour) + int_to_str(minute) + int_to_str(s) + '_11.jpg'
            total_url = base_url + img_url
            # print(total_url)
            if os.path.isfile(total_url):
                image = cv2.imread(total_url)
                if (image is None):
                    continue
                else:
                    print('found..')
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        minute = minute + 1
        if minute >= 60:
            minute = 0
            hour = hour + 1
        if hour >= 19:
            return 0

def get_features_by_day_rebuild(month, day, start, end):
    p1,p2,p3,p4,p5,p6,p7 = ([] for i in range(7))

    hours = list(range(start, end))
    minutes = list(range(0,60))

    for h in tqdm(hours, total=len(hours), unit='Hours progress'):
        for m in minutes:
            try:
                img = get_image_by_date_time(month, day, start, end)
                tmp = cloud_pixel_feature(img)
                p1.append(tmp[0])
                p2.append(tmp[1])
                p3.append(tmp[2])
                p4.append(tmp[3])
                p5.append(tmp[4])
                p6.append(tmp[5])
                p7.append(tmp[6])

            except:
                print('CANT FIND IMG')
                print(month, day, h, m)
                p1.append(0)
                p2.append(0)
                p3.append(0)
                p4.append(0)
                p5.append(0)
                p6.append(0)
                p7.append(0)
    return np.array([p1, p2, p3, p4, p5, p6, p7])


def get_features_by_day(month, day, start, end):
    intensity = []
    number_of_cloud_pixels = []
    harris_corner_detector = []
    edge_detector = []

    hours = list(range(start, end))
    minutes = list(range(0,60))

    for h in tqdm(hours, total=len(hours), unit='Hours progress'):
        for m in minutes:
            try:
                img = get_image_by_date_time(month, day, h, m)
                tmp = extract_features(img)
                intensity.append(tmp[0])
                number_of_cloud_pixels.append(tmp[1])
                harris_corner_detector.append(tmp[2])
                edge_detector.append((tmp[3]))
            except:
                print('CANT FIND IMG')
                print(month, day, h, m)
                intensity.append(0)
                number_of_cloud_pixels.append(0)
                harris_corner_detector.append(0)
                edge_detector.append(0)

    return intensity, number_of_cloud_pixels, harris_corner_detector, edge_detector

def get_image_by_date_time(month, day, hour, minute):
    year = '2019'
    base_url = 'asi_16124/'
    tmp_url = year + int_to_str(month) + int_to_str(day)
    base_url = base_url + tmp_url + '/'  # folder

    seconds_list = ['0', '15', '30', '45']
    for s in seconds_list:
        img_url = tmp_url + int_to_str(hour) + int_to_str(minute) + int_to_str(s) + '_11.jpg'
        total_url = base_url+ img_url
        print(total_url)
        if os.path.isfile(total_url):
            image = cv2.imread(total_url)
            if (image is None):
                continue
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return pre_process_img(image, 400)

    print('CANT FIND IMAGE: ')
    print(month, day, hour, minute)
    return None


def extract_features(img):  # get df from image
    features = np.zeros(4)
    features[0] = intensity(img)
    features[1] = number_of_cloud_pixels_RBR(img)
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


def edge_detector(img):
    edges = cv2.Canny(img, img.shape[0], img.shape[1])
    return np.count_nonzero(edges)
    # print(edge_count)


def number_of_cloud_pixels_RBR(img, th = 0.8):
    copy = np.copy(img)
    height, width, channels = img.shape
    amount_of_cloud_pixels = 0

    for h in range(0, height):
        for w in range(0, width):
            r = copy[h,w,0]
            b = copy[h,w,1]
            g = copy[h,w,2]
            np.seterr(divide='ignore', invalid='ignore')
            rbr = r/b
            if rbr > th and rbr < 1:
                amount_of_cloud_pixels += 1
                copy[h, w, 0] = 124
                copy[h, w, 1] = 252
                copy[h, w, 2] = 0
    show_img(copy)
    return amount_of_cloud_pixels




def number_of_cloud_pixels_BRBG(img, th = 2):
    copy = np.copy(img)
    height, width, channels = img.shape
    amount_of_cloud_pixels = 0
    for h in range(0, height):
        for w in range(0, width):
            r = copy[h,w,0]
            b = copy[h,w,1]
            g = copy[h,w,2]
            np.seterr(divide='ignore', invalid='ignore')
            brbg = (b/r) + (b/g)
            # print(brbg)
            if brbg < th:
                amount_of_cloud_pixels += 1
                copy[h, w, 0] = 124
                copy[h, w, 1] = 252
                copy[h, w, 2] = 0

    show_img(copy)
    return amount_of_cloud_pixels

def get_sun_cor_by_img(img, plot=False):
    # img = features.get_image_by_date_time(8, 21, 17, 0, 0)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)
    if plot:
        img = cv2.circle(orig, maxLoc, 10, (0, 0, 255), -1)
        show_img(img)
    return maxLoc

def number_of_cloud_pixels_NRBR(img, th = 0.8):
    copy = np.copy(img)
    height, width, channels = img.shape
    amount_of_cloud_pixels = 0

    for h in range(0, height):
        for w in range(0, width):
            r = copy[h,w,0]
            b = copy[h,w,1]
            g = copy[h,w,2]
            np.seterr(divide='ignore', invalid='ignore')
            nrbr = (r-b) / (r+b)
            if nrbr > th and nrbr < 1:
                amount_of_cloud_pixels += 1
                copy[h, w, 0] = 124
                copy[h, w, 1] = 252
                copy[h, w, 2] = 0
    show_img(copy)
    return amount_of_cloud_pixels


def get_distance(x1, x2, y1, y2):
    return math.sqrt(math.pow((x2-x1), 2) + math.pow((y2-y1),2))


def show_img(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()
#
# img = get_image_by_date_time(8,21,11,0)
# cloud_pixel_feature(img, plot=True)
#
#
#


