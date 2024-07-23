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


def get_bounding_box(image):
    """Get the bounding box of the non-zero regions in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0, 0, image.shape[1], image.shape[0]
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h


for i, foreground_path in enumerate(tqdm(foreground_path_list)):
    foreground_img = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)  # Read image with alpha channel

    foreground_base_name = os.path.splitext(os.path.basename(foreground_path))[0]
    foreground_height, foreground_width, _ = foreground_img.shape

    for j in range(generate_nums):
        # Create a larger canvas for the augmented image to accommodate transformations
        # 最后增强叠加的图像
        augmented_img = foreground_img.copy()

        num_layers = random.randint(2, 5)  # Random number of layers to add
        layer_imgs = []
        for k in range(num_layers):
            # 每次用原始图像进行比变换
            layer_img = foreground_img.copy()
            # 多次随机变换
            for t in range(transform_times):
                if random.random() > 0.5:
                    transform_matrix = generate_random_transform_matrix(layer_img.shape,
                                                                        scale_range=(0.8, 1.2),
                                                                        rotation_range=(-20, 20),
                                                                        translation_range=(-0.2, 0.2),
                                                                        shear_range=(-20, 20))
                    # Adjust the warpPerspective function to allow larger output size
                    layer_img = perspective_transform(layer_img, transform_matrix)
                    layer_img = elastic_transform(layer_img, alpha=foreground_height * 0.1,
                                                  sigma=foreground_height * 0.05,
                                                  alpha_affine=foreground_height * 0.03)
            layer_imgs.append(layer_img)

        max_height = max([layer_img.shape[0] for layer_img in layer_imgs])
        max_width = max([layer_img.shape[1] for layer_img in layer_imgs])

        augmented_img = np.zeros((max_height, max_width, 4), dtype=np.uint8)
        mask = None
        for layer_img in layer_imgs:
            h, w = layer_img.shape[:2]
            # 中心店有问题
            center_x = random.randint(w // 2, max_width - w // 2)
            center_y = random.randint(h // 2, max_height - h // 2)

            x_min = center_x - w // 2
            y_min = center_y - h // 2
            x_max = center_x + w // 2
            y_max = center_y + h // 2

            if mask is None:
                mask = np.ones((max_height, max_width), dtype=np.uint8) * 255
            else:
                mask = augmented_img[:, :, 3]

            augmented_img[y_min:y_max, x_min:x_max, :3] = cv2.seamlessClone(layer_img, augmented_img[y_min:y_max, x_min:x_max, :3],
                                                                    mask, (center_y,center_x), cv2.NORMAL_CLONE)

            # alpha_s = layer_img[:, :, 3] / 255.0
            # alpha_l = 1.0 - alpha_s

            # for c in range(0, 3):
            #     augmented_img[y:y + h, x:x + w, c] = (
            #             alpha_s * layer_img[:, :, c] + alpha_l * augmented_img[y:y + h, x:x + w, c])
            # 更新alpha通道
            augmented_img[y_min:y_max, x_min:x_max, :3] = np.maximum(
                augmented_img[y_min:y_max, x_min:x_max, :3], layer_img[:, :, 3])

        # Get the bounding box of the augmented image to make the edges tight
        # x, y, w, h = get_bounding_box(augmented_img)
        # tight_augmented_img = augmented_img[y:y + h, x:x + w]

        save_path = os.path.join(save_img_path, f'{foreground_base_name}_{j}.png')
        cv2.imwrite(save_path, augmented_img)
