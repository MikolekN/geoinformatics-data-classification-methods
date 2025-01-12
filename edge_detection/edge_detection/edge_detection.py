import os.path
import time
from os import listdir
from os.path import join, isfile
from . import filters
import cv2 as cv


def save_after_filter(path, img, name, time, performance_data_file):
    cv.imwrite(path, img)
    info = f"{name}: execution time = {time:.8f}s"
    print(info)
    performance_data_file.write(f"{info}\n")

def apply_filter(filter_name, filter_func, img, paths, performance_file):
    start = time.monotonic()
    filtered_img = filter_func(img)
    end = time.monotonic()
    execution_time = end - start
    save_after_filter(paths[filter_name], filtered_img, filter_name, execution_time, performance_file)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run():
    input_dirs = ["forest", "mountain"]
    for input_dir in input_dirs:
        input_path = os.path.join(BASE_DIR, "input", input_dir)
        images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        for image in images:
            img_name = image

            paths = {
                "img": os.path.join(input_path, img_name),
                **{key: os.path.join("..", "output", input_dir, key, img_name) for key in filters.FILTERS.keys()},
                "performance_data": os.path.join("..", "output", input_dir, "performance", img_name + ".txt")
            }

            for key in paths:
                os.makedirs(os.path.dirname(paths[key]), exist_ok=True)

            img = cv.imread(paths["img"], cv.IMREAD_GRAYSCALE)

            with open(paths["performance_data"], "w") as performance_data_file:
                img_info = f"{img_name}, resolution: {img.shape[1]}x{img.shape[0]}, execution time of specific methods are listed below."
                performance_data_file.write(f"{img_info}\n")
                print(img_info)

                # Apply each filter (defined in filters.py)
                for filter_name, filter_func in filters.FILTERS.items():
                    apply_filter(filter_name, filter_func, img, paths, performance_data_file)
