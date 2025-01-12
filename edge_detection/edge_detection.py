import os.path
import time
import sys
from os import listdir
from os.path import join, isfile

import cv2 as cv
import numpy as np
from scipy import ndimage


def prewitt_filter(image):
    image = image.astype('float64')
    image /= 255.0
    prewitt_h = ndimage.prewitt(image, axis=0)
    prewitt_v = ndimage.prewitt(image, axis=1)
    magnitude = np.sqrt(prewitt_h ** 2 + prewitt_v ** 2)
    magnitude *= 255 / np.max(magnitude)  # Normalization
    return magnitude

def roberts_filter(image):
    g_x = np.array([[1, 0], [0, -1]])
    g_y = np.array([[0, 1], [-1, 0]])
    gradient_x = cv.filter2D(image, cv.CV_64F, g_x)
    gradient_y = cv.filter2D(image, cv.CV_64F, g_y)
    roberts = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return roberts

def robinson_filter(image):
    masks =  np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
             [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
             [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
             [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
             [[0, -1, -2], [1, 0, -1], [2, 1, 0]],
             [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]])
    edges = []
    for mask in masks:
        edges.append(np.abs(cv.filter2D(image, cv.CV_64F, mask)))
    robinson = np.maximum.reduce(edges)
    robinson *= 255 / np.max(robinson)  # Normalization
    return robinson

def save_after_filter(path, img, name, time, performance_data_file):
    cv.imwrite(path, img)
    info = f"{name}: execution time = {time:.8f}s"
    print(info)
    performance_data_file.write(f"{info}\n")


input_dirs = ["forest", "mountain"]
for input_dir in input_dirs:
    input_path = os.path.join(".", "input", input_dir)
    images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    for image in images:
        img_name = image

        paths = {
            "img": os.path.join(input_path, img_name),
            "roberts": os.path.join(".", "output", input_dir, "roberts", img_name),
            "prewitt": os.path.join(".", "output", input_dir, "prewitt", img_name),
            "sobel": os.path.join(".", "output", input_dir, "sobel", img_name),
            "robinson": os.path.join(".", "output", input_dir, "robinson", img_name),
            "laplacian": os.path.join(".", "output", input_dir, "laplacian", img_name),
            "canny": os.path.join(".", "output", input_dir, "canny", img_name),
            "performance_data": os.path.join(".", "output", input_dir, "performance", img_name + ".txt")
        }
        for key in paths:
            os.makedirs(os.path.dirname(paths[key]), exist_ok=True)

        img = cv.imread(paths["img"], cv.IMREAD_GRAYSCALE)
        performance_data_file = open(paths["performance_data"], "w")
        img_info = f"{img_name}, resolution: {img.shape[1]}x{img.shape[0]}, execution time of specific methods are listed below."
        performance_data_file.write(f"{img_info}\n")
        print(img_info)

        # Roberts
        start = time.monotonic()
        roberts_img = roberts_filter(image=img)
        end = time.monotonic()
        roberts_time = end - start
        save_after_filter(paths["roberts"], roberts_img, "Roberts", roberts_time, performance_data_file)

        # Prewitt
        start = time.monotonic()
        prewitt_img = prewitt_filter(image=img)
        end = time.monotonic()
        prewitt_time = end - start
        save_after_filter(paths["prewitt"], prewitt_img, "Prewitt", prewitt_time, performance_data_file)

        # Sobel
        start = time.monotonic()
        sobel_img = cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
        end = time.monotonic()
        sobel_time = end - start
        save_after_filter(paths["sobel"], sobel_img, "Sobel", sobel_time, performance_data_file)

        # Robinson
        start = time.monotonic()
        robinson_img = robinson_filter(image=img)
        end = time.monotonic()
        robinson_time = end - start
        save_after_filter(paths["robinson"], robinson_img, "Robinson", robinson_time, performance_data_file)

        # Laplacian
        start = time.monotonic()
        laplacian_img = cv.Laplacian(src=img, ddepth=cv.CV_16S, ksize=3)
        end = time.monotonic()
        laplacian_time = end - start
        save_after_filter(paths["laplacian"], laplacian_img, "Laplacian", laplacian_time, performance_data_file)

        # Canny
        start = time.monotonic()
        canny_img = cv.Canny(image=img, threshold1=100, threshold2=200)
        end = time.monotonic()
        canny_time = end - start
        save_after_filter(paths["canny"], canny_img, "Canny", canny_time, performance_data_file)