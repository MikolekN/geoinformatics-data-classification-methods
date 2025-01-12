import numpy as np
from scipy import ndimage
import cv2 as cv

FILTERS = {
    "roberts": lambda img: roberts_filter(image=img),
    "prewitt": lambda img: prewitt_filter(image=img),
    "sobel": lambda img: cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5),
    "robinson": lambda img: robinson_filter(image=img),
    "laplacian": lambda img: cv.Laplacian(src=img, ddepth=cv.CV_16S, ksize=3),
    "canny": lambda img: cv.Canny(image=img, threshold1=100, threshold2=200),
}

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