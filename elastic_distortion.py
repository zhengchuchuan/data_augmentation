import os
import random

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from tqdm import tqdm

from utils.utils import generate_random_transform_matrix
from utils.transforms import elastic_transform, perspective_transform
from utils.data_process import center_bbox_to_corner_bbox, random_sub_label
from utils.file_io import read_yolo_labels






jpg_path = None
label_path = r'C:\Users\zcc\project\wayho\oil_detection\oil_identification\image_expansion\data\labels\MAX_20240608_MAX_0001_Color_D.txt'
jpg_path = label_path.replace('labels', 'imgs').replace('.txt', '.png')
result_path = 'experiment/copy'

file_name = os.path.basename(jpg_path)
file_name_no_ext = os.path.splitext(file_name)[0]
if not os.path.exists(result_path):
    os.makedirs(result_path)

# 读取图像
jpg = cv2.imread(jpg_path)
jpg_height, jpg_weight, _ = jpg.shape
jpg_center = (jpg_weight // 2, jpg_height // 2)
# 读取标签
yolo_labels = read_yolo_labels(label_path)
bbox_labels = center_bbox_to_corner_bbox(yolo_labels, jpg_height, jpg_weight)

cut_imgs = []

random_times = 200
for i in tqdm(range(random_times)):
    result = jpg
    for _, bbox_label in enumerate(bbox_labels):
        class_id, x1, y1, x2, y2 = bbox_label
        width = x2 - x1
        height = y2 - y1
        label_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        labeled_patch_img = jpg[y1:y2, x1:x2, :]

        # 弹性变换
        elastic_transformed_img = elastic_transform(labeled_patch_img, alpha=height*0.1, sigma=height*0.02, alpha_affine=height*0.02)
        # 生成随机变换矩阵
        transform_matrix = generate_random_transform_matrix(elastic_transformed_img.shape, scale_range=(0.8, 1.2),
                                                            rotation_range=(-10, 10),
                                                            translation_range=(0.1, 0.3), shear_range=(-10, 10))
        # 应用透视变换, 返回变换后的图像、更新后的检测框列表和掩码

        perspective_transformed_img, transformed_bbox,mask = perspective_transform(elastic_transformed_img, bbox_label, transform_matrix)

        # mask = np.ones((height, width, 3), dtype=np.uint8) * 255

        d_rate = 0.2
        dx = np.random.randint(- width*d_rate, width*d_rate)
        dy = np.random.randint(- height*d_rate, height*d_rate)
        random_cneter = (label_center[0] + dx,
                         label_center[1] + dy)

        transformed_width = transformed_bbox[3] - transformed_bbox[1]
        transformed_height = transformed_bbox[4] - transformed_bbox[2]
        if random_cneter[0] - transformed_width // 2 < 0 or random_cneter[0] + transformed_height // 2 < 0:
            continue

        if random_cneter[1] - transformed_height // 2 < 0 or random_cneter[1] + transformed_height // 2 < 0:
            continue

        """
        -NORMAL_CLONE: 不保留dst 图像的texture细节。目标区域的梯度只由源图像决定。
        -MIXED_CLONE: 保留dest图像的texture 细节。目标区域的梯度是由原图像和目的图像的组合计算出来(计算dominat gradient)。
        -MONOCHROME_TRANSFER: 不保留src图像的颜色细节，只有src图像的质地，颜色和目标图像一样，可以用来进行皮肤质地填充。

        """
        result = cv2.seamlessClone(perspective_transformed_img, result, mask, random_cneter, cv2.MIXED_CLONE)

    cv2.imwrite(f'{result_path}/{file_name_no_ext}_{i}.png', result)
# cv2.imwrite(f'{result_path}/mask.png', mask)
# cv2.imwrite(f'{result_path}/labeled_patch_img.png', labeled_patch_img)
# cv2.imwrite(f'{result_path}/perspective_transformed_img.png', perspective_transformed_img)

