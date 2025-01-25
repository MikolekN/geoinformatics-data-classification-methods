import os

from edge_detection import edge_detection
from image_preprocess.process_images import process_images

folders = ["mountain", "forest"]
subfolder = "image_preprocess/photos/"
output_base_folder = "input"

if __name__ == "__main__":

    for folder in folders:
        process_images(os.path.join(subfolder, folder), os.path.join(output_base_folder, folder))

    edge_detection.run()