import os
import cv2
import numpy as np

def apply_salt_and_pepper_noise(image, salt_vs_pepper=0.5, amount=0.02):
    row, col, _ = image.shape
    num_salt = np.ceil(amount * image.size * salt_vs_pepper).astype(int)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper)).astype(int)

    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]

    noisy_image = image.copy()
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

def apply_gaussian_noise(image, mean=0, var=1):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(image, gauss)
    return noisy_image

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            file_id = os.path.splitext(filename)[0]
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Could not load image {image_path}. Skipping...")
                continue

            # Save the high-resolution original
            cv2.imwrite(os.path.join(output_folder, f"{file_id}_high_res_original.jpg"), image)

            # Create and save low-resolution version
            low_res_image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
            cv2.imwrite(os.path.join(output_folder, f"{file_id}_low_res_original.jpg"), low_res_image)

            # Add salt-and-pepper noise and save
            snp_image = apply_salt_and_pepper_noise(image)
            cv2.imwrite(os.path.join(output_folder, f"{file_id}_high_res_snp.jpg"), snp_image)

            low_res_snp_image = apply_salt_and_pepper_noise(low_res_image)
            cv2.imwrite(os.path.join(output_folder, f"{file_id}_low_res_snp.jpg"), low_res_snp_image)

            # Add Gaussian noise and save
            gauss_image = apply_gaussian_noise(image)
            cv2.imwrite(os.path.join(output_folder, f"{file_id}_high_res_gauss.jpg"), gauss_image)

            low_res_gauss_image = apply_gaussian_noise(low_res_image)
            cv2.imwrite(os.path.join(output_folder, f"{file_id}_low_res_gauss.jpg"), low_res_gauss_image)

