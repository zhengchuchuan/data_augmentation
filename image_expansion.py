import os
import random

import cv2
import numpy as np

from image_expansion.utils.utils import generate_random_transform_matrix
from utils.data_process import center_bbox_to_corner_bbox, random_sub_label
from utils.file_io import read_yolo_labels





jpg_path = None
label_path = r'C:\Users\zcc\project\wayho\oil_detection\oil_identification\image_expansion\data\labels\MAX_20240519_MAX_0076_Color_D.txt'
jpg_path = label_path.replace('labels', 'imgs').replace('.txt', '.png')
result_path = 'experiment/copy'

file_name = os.path.basename(jpg_path)
file_name_no_ext = os.path.splitext(file_name)[0]
if not os.path.exists(result_path):
    os.makedirs(result_path)


# 读取图像
jpg = cv2.imread(jpg_path)
jpg_height, jpg_weight, c = jpg.shape
jpg_center = (jpg_weight // 2, jpg_height // 2)
# 读取标签
yolo_labels = read_yolo_labels(label_path)
bbox_labels = center_bbox_to_corner_bbox(yolo_labels, jpg_height, jpg_weight)

cut_imgs = []
for _, bbox_label in enumerate(bbox_labels):
    class_id, x1, y1, x2, y2 = bbox_label
    width = x2 - x1
    height = y2 - y1
    label_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    labeled_patch_img = jpg[y1:y2, x1:x2, :]

    random_times = 50
    for j in range(random_times):

        # dx = np.random.randint(- width//4, width//4)
        # dy = np.random.randint(- height//4, height//4)
        # random_cneter = (label_center[0] + dx, label_center[1] + dy)
        # 标注图像内随机选取一个子图像
        sub_labeled_patch_img_label = random_sub_label(bbox_label,width_ratio=0.4, height_ratio=0.4)
        _, sub_x1, sub_y1, sub_x2, sub_y2 = sub_labeled_patch_img_label
        sub_labeled_patch_img = jpg[sub_y1:sub_y2, sub_x1:sub_x2, :]

        sub_width = sub_x2 - sub_x1
        sub_height = sub_y2 - sub_y1
        sub_label_center = ((sub_x1 + sub_x2) // 2, (sub_y1 + sub_y2) // 2)

        d_ratio = 0.4
        dx = np.random.randint(- sub_width*d_ratio, sub_width*d_ratio)
        dy = np.random.randint(- sub_height*d_ratio, sub_height*d_ratio)
        random_cneter = (sub_label_center[0] + dx, sub_label_center[1] + dy)


        # 生成随机变换矩阵
        transform_matrix = generate_random_transform_matrix(sub_labeled_patch_img.shape, scale_range=(0.8, 1.2), rotation_range=(-10, 10),
                                     translation_range=(0.1, 0.3), shear_range=(-10, 10))
        # 应用透视变换, 返回变换后的图像、更新后的检测框列表和掩码
        # transformed_cut_image, transformed_boxes,mask = apply_perspective_transform(sub_labeled_patch_img, sub_labeled_patch_img_label, transform_matrix)
        # transformed_cut_image, transformed_boxes,mask = apply_perspective_transform(labeled_patch_img, bbox_label, transform_matrix)

        """
        -NORMAL_CLONE: 不保留dst 图像的texture细节。目标区域的梯度只由源图像决定。
        -MIXED_CLONE: 保留dest图像的texture 细节。目标区域的梯度是由原图像和目的图像的组合计算出来(计算dominat gradient)。
        -MONOCHROME_TRANSFER: 不保留src图像的颜色细节，只有src图像的质地，颜色和目标图像一样，可以用来进行皮肤质地填充。
    
        """
        jpg = cv2.seamlessClone(transformed_cut_image, jpg, mask, random_cneter, cv2.MIXED_CLONE)
        # jpg[sub_y1:sub_y2, sub_x1:sub_x2, :] = sub_labeled_patch_img

        cv2.imwrite(f'{result_path}/{file_name_no_ext}_{j}.png', jpg)
    cv2.imwrite(f'{result_path}/mask.png', mask)
    cv2.imwrite(f'{result_path}/sub_labeled_patch_img.png', sub_labeled_patch_img)
    cv2.imwrite(f'{result_path}/transformed_cut_image.png', transformed_cut_image)

