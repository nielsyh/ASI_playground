import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import os.path
import matplotlib.pyplot as plt
from tqdm import tqdm

def int_to_str(i):
    s = str(i)
    if(len(s)) == 2:
        return s
    else:
        return '0' + s

def get_full_image_by_date_time(month, day, hour, minute, seconds):
    seconds_list = ['0', '15', '30', '45']

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
                img = get_image_by_date_time(19, month, day, h, m, 0)
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

    return intensity, number_of_cloud_pixels, harris_corner_detector,edge_detector

def get_image_by_date_time(year, month, day, hour, minute, seconds):
    if (year == 19):
        year = '2019'
    elif (year == 20):
        year = '2020'

    base_url = 'asi_16124/'
    tmp_url = year + int_to_str(month) + int_to_str(day)
    base_url = base_url + tmp_url + '/'  # folder

    seconds_list = ['0', '15', '30', '45']
    for s in seconds_list:
        img_url = tmp_url + int_to_str(hour) + int_to_str(minute) + int_to_str(s) + '_11.jpg'
        total_url = base_url+ img_url
        # print(total_url)
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



def edge_detector(img):
    edges = cv2.Canny(img, img.shape[0], img.shape[1])
    return np.count_nonzero(edges)
    # print(edge_count)


def number_of_cloud_pixels(img, threshold = 0.8):
    height, width, channels = img.shape
    amount_of_cloud_pixels = 0

    for h in range(0, height):
        for w in range(0, width):
            r = img[h,w,0]
            b = img[h,w,1]
            np.seterr(divide='ignore', invalid='ignore')
            rbr = r/b
            if rbr > 0.67 and rbr < 1:
                amount_of_cloud_pixels += 1
                img[h, w, 0] = 124
                img[h, w, 1] = 252
                img[h, w, 2] = 0
    # plt.imshow(img)
    # plt.show()
    return amount_of_cloud_pixels

def show_img(img):
    plt.imshow(img)
    plt.show()


# #
# a = Features()
# img = get_image_by_date_time(19,9,1,12,0,0)
# number_of_cloud_pixels(img)


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
# cv2.imshow('res', img)
#
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
#
# # a.number_of_cloud_pixels(img)
# # a.edge_detector(img)
# # # a.harris_corner_detector(img)
# # # print(intensity(img))