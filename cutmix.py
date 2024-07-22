import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from utils.utils import generate_random_transform_matrix, rectangle_union, validate_polygon, get_bounding_box
from utils.transforms import elastic_transform, perspective_transform
from utils.data_process import xywh_rect_to_x1y1x2y2
from utils.file_io import make_sure_paths_exist, read_lines_to_list
from utils.img_label_utils import read_yolo_labels, read_labelme_json, yolo_to_labelme_json, save_labelme_json, \
    labelme_json_to_yolo, save_yolo_labels, points_to_yolo_label

per_background_nums = 2
paste_nums = 5
# 标间类别索引
class_idx = 2

background_list_path = 'data/data_list/background_list.txt'
foreground_list_path = 'data/data_list/foreground_list.txt'
save_img_path = 'exp/cutmix_result/imgs'
save_label_path = 'exp/cutmix_result/labels'

make_sure_paths_exist(save_img_path, save_label_path)

background_list = read_lines_to_list(background_list_path)
foreground_list = read_lines_to_list(foreground_list_path)

for i, background_path in enumerate(tqdm(background_list)):

    background_img = cv2.imread(background_path)
    save_name = os.path.splitext(os.path.basename(background_path))[0]
    background_height, background_width, _ = background_img.shape

    # 随机选择n个前景粘贴
    if paste_nums > len(foreground_list):
        paste_nums = len(foreground_list)
        print(f'Warning: paste_nums is larger than the number of foreground images, set paste_nums to {paste_nums}')
    random_foreground_list_idx = random.sample(range(len(foreground_list)), paste_nums)

    for j in range(per_background_nums):
        fusion_img = background_img.copy()
        labels = []

        for k in random_foreground_list_idx:
            # 读取前景图像，png不忽略透明度
            foreground_img = cv2.imread(foreground_list[k], cv2.IMREAD_UNCHANGED)
            foreground_height, foreground_width, foreground_channels = foreground_img.shape

            if foreground_channels != 4:
                raise ValueError(f"Foreground image {foreground_list[k]} does not have alpha channels")

            # 提取前景图像的 alpha 通道，并归一化到 [0, 1] 范围
            alpha_foreground = foreground_img[:, :, 3] / 255.0
            alpha_background = 1.0 - alpha_foreground

            # 随机选择粘贴位置
            paste_x = random.randint(0, background_width - foreground_width)
            paste_y = random.randint(0, background_height - foreground_height)

            # 确保前景图像不会超出背景图像的边界
            if paste_x + foreground_width > background_width or paste_y + foreground_height > background_height:
                raise ValueError("Foreground image exceeds background image boundaries")

            x_end = paste_x + foreground_width
            y_end = paste_y + foreground_height

            points = ((paste_x, paste_y), (x_end,y_end))

            # 混合图像
            for c in range(0, 3):
                fusion_img[paste_y:y_end, paste_x:x_end, c] = \
                    (alpha_foreground * foreground_img[:, :, c] + alpha_background * fusion_img[paste_y:y_end,paste_x:x_end, c])

            # 保存标签
            label = points_to_yolo_label(points, class_idx,fusion_img.shape)
            labels.append(label)

        # 保存结果图像
        result_img_name = f'{save_name}_{j}.png'
        result_img_path = os.path.join(save_img_path, result_img_name)

        # 保存标签
        result_label_name = f'{save_name}_{j}.txt'
        result_label_path = os.path.join(save_label_path, result_label_name)


        cv2.imwrite(result_img_path, fusion_img)
        save_yolo_labels(labels, result_label_path)

