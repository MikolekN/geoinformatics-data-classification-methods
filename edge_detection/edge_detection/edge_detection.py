import os.path
import time
from os import listdir
from os.path import join, isfile
from . import filters
from . import image_result
import cv2 as cv
import json


def save_after_filter(path, img, name, time):
    cv.imwrite(path, img)
    info = f"{name}: execution time = {time:.8f}s"
    # print(info)


def apply_filter(filter_name, filter_func, img, paths):
    start = time.monotonic()
    filtered_img = filter_func(img)
    end = time.monotonic()
    execution_time = end - start
    save_after_filter(paths[filter_name], filtered_img, filter_name, execution_time)
    return execution_time


def run():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    image_id = 0
    results = []
    input_dirs = ["forest", "mountain"]
    for input_dir in input_dirs:
        input_path = os.path.join(BASE_DIR, "input", input_dir)
        images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        for image in images:

            image_id += 1
            img_original_path = os.path.join(input_path, image)
            img_name = image

            paths = {"img": os.path.join(input_path, img_name),
                     **{key: os.path.join("..", "output", input_dir, key, img_name) for key in
                        filters.FILTERS.keys()},
                     "json_dump": os.path.join("..", "output", input_dir)}

            for key in paths:
                os.makedirs(os.path.dirname(paths[key]), exist_ok=True)

            img = cv.imread(paths["img"], cv.IMREAD_GRAYSCALE)
            height, width = img.shape

            execution_times = {}
            # Apply each filter (defined in filters.py)
            for filter_name, filter_func in filters.FILTERS.items():
                execution_times[filter_name] = apply_filter(filter_name, filter_func, img, paths)

            results.append(image_result.ImageResult(
                id=image_id,
                original_path=img_original_path,
                is_high_resolution="high_res" in img_name,
                is_ai_generated = "ai" in img_name,
                is_gauss_noise = "gauss" in img_name,
                is_salt_and_pepper_noise = "snp" in img_name,
                roberts_path=paths["roberts"],
                prewitt_path=paths["prewitt"],
                sobel_path=paths["sobel"],
                robinson_path=paths["robinson"],
                laplace_path=paths["laplacian"],
                canny_path=paths["canny"],
                time_roberts=execution_times["roberts"],
                time_prewitt=execution_times["prewitt"],
                time_sobel=execution_times["sobel"],
                time_robinson=execution_times["robinson"],
                time_laplace=execution_times["laplacian"],
                time_canny=execution_times["canny"],
                width=width,
                height=height
            ))

            output_json_path = os.path.join(BASE_DIR, paths["json_dump"], "results.json")
            with open(output_json_path, "w") as json_file:
                json.dump([result.to_dict() for result in results], json_file, indent=4)
