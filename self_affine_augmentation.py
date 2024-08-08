from datetime import datetime
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from utils.transforms import elastic_transform, perspective_transform
from utils.file_io import make_sure_paths_exist, read_lines_to_list
from utils.utils import generate_random_transform_matrix

generate_nums = 150
transform_times = 6
max_layers = 5
start_idx = 0

foreground_list_path = r'data/data_list/20240801_generate_foreground_list_2.txt'
save_img_path = r'\\192.168.3.155\高光谱测试样本库\原油检测\00大庆现场测试\03标注数据以及模型文件\Generate\20240801\generate_foreground\2'
make_sure_paths_exist(save_img_path, save_img_path)

foreground_path_list = read_lines_to_list(foreground_list_path)


def random_flip_and_rotate(image):
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random vertical flip
    if random.random() > 0.5:
        image = cv2.flip(image, 0)

    # Random 90-degree rotation
    num_rotations = random.randint(0, 3)  # 0, 1, 2, or 3 times 90-degree rotation
    if num_rotations > 0:
        image = np.rot90(image, num_rotations)

    return image


for i, foreground_path in enumerate(tqdm(foreground_path_list)):

    foreground_img = cv2.imdecode(np.fromfile(foreground_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    foreground_base_name = os.path.splitext(os.path.basename(foreground_path))[0]
    foreground_height, foreground_width, _ = foreground_img.shape

    for j in range(generate_nums):
        # Create a larger canvas for the augmented image to accommodate transformations

        num_layers = random.randint(1, max_layers)  # Random number of layers to add
        layer_imgs = []
        for k in range(num_layers):
            layer_img = foreground_img.copy()
            # layer_img = random_flip_and_rotate(layer_img)
            # 随机多次变换
            for t in range(transform_times):
                if random.random() > 0.8:
                    if layer_img.shape[0] > 80 or layer_img.shape[1] > 80:
                        min_scale = 0.8
                        max_sale = 1.0
                    elif layer_img.shape[0] < 20 or layer_img.shape[1] < 20:
                        min_scale = 1
                        max_sale = 1.2
                    else:
                        min_scale = 0.8
                        max_sale = 1.2
                    transform_matrix = generate_random_transform_matrix(layer_img.shape,
                                                                        scale_range=(min_scale, max_sale),
                                                                        # Reduced range
                                                                        rotation_range=(-15, 15),  # Reduced range
                                                                        translation_range=(-0.1, 0.1),  # Reduced range
                                                                        shear_range=(-10, 10))  # Reduced range
                    layer_img = perspective_transform(layer_img, transform_matrix)

                    alpha_ratio = random.uniform(0.2, 0.4)
                    sigma_ratio = random.uniform(0.05, 0.1)

                    layer_img = elastic_transform(image=layer_img, alpha=foreground_height * alpha_ratio,
                                                  sigma=foreground_height * sigma_ratio)
            layer_imgs.append(layer_img)

        max_height = max([layer_img.shape[0] for layer_img in layer_imgs])
        max_width = max([layer_img.shape[1] for layer_img in layer_imgs])

        augmented_img = np.zeros((max_height, max_width, 4), dtype=np.uint8)
        for layer_img in layer_imgs:
            h, w = layer_img.shape[:2]
            y = random.randint(0, max_height - h)
            x = random.randint(0, max_width - w)
            alpha_s = layer_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            # Apply Gaussian blur to the alpha channel to smooth edges
            # layer_img[:, :, 3] = cv2.GaussianBlur(layer_img[:, :, 3], (5, 5), 0)

            for c in range(0, 3):
                augmented_img[y:y + layer_img.shape[0], x:x + layer_img.shape[1], c] = (
                        alpha_s * layer_img[:, :, c] + alpha_l * augmented_img[y:y + layer_img.shape[0],
                                                                 x:x + layer_img.shape[1], c]
                )

            augmented_img[y:y + layer_img.shape[0], x:x + layer_img.shape[1], 3] = np.maximum(
                augmented_img[y:y + layer_img.shape[0], x:x + layer_img.shape[1], 3], layer_img[:, :, 3]
            )

        # Randomly flip and rotate the final augmented image
        augmented_img = random_flip_and_rotate(augmented_img)

        if augmented_img.shape[0] > 120 or augmented_img.shape[1] > 120:
            # 随机生成目标尺寸（在70到90之间）
            target_size = random.randint(80, 100)
            # 计算缩放比例
            height, width = augmented_img.shape[:2]
            scale = 0.8
            # 计算新的尺寸
            new_size = (int(width * scale), int(height * scale))
            # 缩放图像
            augmented_img = cv2.resize(augmented_img, new_size, interpolation=cv2.INTER_AREA)

        save_path = os.path.join(save_img_path, f'{foreground_base_name}_{start_idx + j}.png')

        image_type = '.png'
        success, img_encoded = cv2.imencode(image_type, augmented_img)
        if not success:
            raise ValueError(f"Could not encode image to format: {image_type}")
        img_encoded.tofile(save_path)
