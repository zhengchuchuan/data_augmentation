from datetime import datetime
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from utils.transforms import elastic_transform, perspective_transform
from utils.file_io import make_sure_paths_exist, read_lines_to_list
from utils.utils import generate_random_transform_matrix

generate_nums = 20
transform_times = 2

foreground_list_path = 'data/data_list/source_foreground_list.txt'
save_img_path = 'exp/cutmix_result/augmentation_foreground'
make_sure_paths_exist(save_img_path, save_img_path)

foreground_path_list = read_lines_to_list(foreground_list_path)


def alpha_blend(img1, img2, x, y):
    """Blend img2 onto img1 at position (x, y) using alpha blending."""
    alpha_s = img2[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img1[y:y + img2.shape[0], x:x + img2.shape[1], c] = (
                alpha_s * img2[:, :, c] + alpha_l * img1[y:y + img2.shape[0], x:x + img2.shape[1], c]
        )

    img1[y:y + img2.shape[0], x:x + img2.shape[1], 3] = np.maximum(
        img1[y:y + img2.shape[0], x:x + img2.shape[1], 3], img2[:, :, 3]
    )


for i, foreground_path in enumerate(tqdm(foreground_path_list)):
    foreground_img = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)  # Read image with alpha channel

    foreground_base_name = os.path.splitext(os.path.basename(foreground_path))[0]
    foreground_height, foreground_width, _ = foreground_img.shape

    for j in range(generate_nums):
        # Create a larger canvas for the augmented image to accommodate transformations
        # 最后增强叠加的图像
        augmented_img = np.zeros((foreground_height, foreground_width, 4), dtype=np.uint8)

        num_layers = random.randint(1, 5)  # Random number of layers to add
        layer_imgs = []
        for k in range(num_layers):
            # 每次用原始图像进行比变换
            layer_img = foreground_img.copy()
            # 多次随机变换
            for t in range(transform_times):
                if random.random() > 0.5:
                    transform_matrix = generate_random_transform_matrix(layer_img.shape,
                                                                        scale_range=(0.8, 1.2),
                                                                        rotation_range=(-15, 15),
                                                                        translation_range=(-0.2, 0.2),
                                                                        shear_range=(-10, 10))
                    # Adjust the warpPerspective function to allow larger output size
                    layer_img = perspective_transform(layer_img, transform_matrix)
                    layer_img = elastic_transform(layer_img, alpha=foreground_height * 0.2,
                                                  sigma=foreground_height * 0.1,
                                                  alpha_affine=foreground_height * 0.1)
            layer_imgs.append(layer_img)

        max_height = max([layer_img.shape[0] for layer_img in layer_imgs])
        max_width = max([layer_img.shape[1] for layer_img in layer_imgs])

        augmented_img = np.zeros((max_height, max_width, 4), dtype=np.uint8)
        for layer_img in layer_imgs:
            h, w = layer_img.shape[:2]
            y = random.randint(0, max_height - h)
            x = random.randint(0, max_width - w)
            alpha_blend(augmented_img, layer_img, x, y)

        save_path = os.path.join(save_img_path, f'{foreground_base_name}_{j}.png')
        cv2.imwrite(save_path, augmented_img)
