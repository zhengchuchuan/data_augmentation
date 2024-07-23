import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from utils.utils import generate_random_transform_matrix, rectangle_union, validate_polygon, get_bounding_box, \
    is_overlap
from utils.transforms import elastic_transform, perspective_transform
from utils.data_process import xywh_rect_to_x1y1x2y2
from utils.file_io import make_sure_paths_exist, read_lines_to_list
from utils.img_label_utils import read_yolo_labels, read_labelme_json, yolo_to_labelme_json, save_labelme_json, \
    labelme_json_to_yolo, save_yolo_labels, points_to_yolo_label

from utils.utils import generate_random_transform_matrix

per_background_nums = 2
paste_nums = 5
# 标间类别索引
class_idx = 2

background_list_path = 'data/data_list/background_list.txt'
background_label_path = None
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


    # 待完善,当背景图像也有标签时
    if background_label_path is not None:
        pass



    for j in range(per_background_nums):
        fusion_img = background_img.copy()

        # 背景弹性变形
        fusion_img = elastic_transform(fusion_img, alpha=background_height * 0.2,
                              sigma=background_height * 0.02, alpha_affine=background_height * 0.02)
        labels = []

        # 随机选择n个前景粘贴
        # if paste_nums > len(foreground_list):
        #     paste_nums = len(foreground_list)
        #     print(f'Warning: paste_nums is larger than the number of foreground images, set paste_nums to {paste_nums}')
        # random_foreground_list_idx = random.sample(range(len(foreground_list)), paste_nums)

        # 随机选择n个前景粘贴
        if paste_nums > len(foreground_list):
            print(
                f'Warning: paste_nums is larger than the number of foreground images, some images will be reused to match paste_nums')
        extended_foreground_list = foreground_list * (paste_nums // len(foreground_list)) + random.sample(
            foreground_list, paste_nums % len(foreground_list))
        random_foreground_list = random.sample(extended_foreground_list, paste_nums)

        for foreground_path in random_foreground_list:
            # 读取前景图像，png不忽略透明度
            foreground_img = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
            foreground_height, foreground_width, foreground_channels = foreground_img.shape

            if foreground_channels != 4:
                raise ValueError(f"Foreground image {foreground_path} does not have alpha channels")

            # 前景弹性变形
            elastic_transformed_foreground_img = elastic_transform(foreground_img, alpha=foreground_height * 0.1,
                              sigma=foreground_height * 0.02, alpha_affine=foreground_height * 0.02)



            # 生成随机透视变换矩阵
            transform_matrix = generate_random_transform_matrix(elastic_transformed_foreground_img.shape,
                                                                scale_range=(0.8, 1.2),
                                                                rotation_range=(-15, 15),
                                                                translation_range=(0.1, 0.3),
                                                                shear_range=(-10, 10))

            transformed_foreground_img = perspective_transform(elastic_transformed_foreground_img, transform_matrix)
            # transformed_foreground_img = foreground_img
            transed_foreground_height, transed_foreground_width, _ = transformed_foreground_img.shape

            # 提取前景图像的 alpha 通道，并归一化到 [0, 1] 范围
            alpha_foreground = transformed_foreground_img[:, :, 3] / 255.0
            alpha_background = 1.0 - alpha_foreground

            # 不满足条件则重新选择位置,最多尝试10次
            max_attempts = 10
            for attempt in range(max_attempts):
                # 随机选择粘贴位置
                paste_x = random.randint(0, background_width - transed_foreground_width)
                paste_y = random.randint(0, background_height - transed_foreground_height)

                # 确保前景图像不会超出背景图像的边界
                if paste_x + transed_foreground_width > background_width or paste_y + transed_foreground_height > background_height:
                    continue

                x_end = paste_x + transed_foreground_width
                y_end = paste_y + transed_foreground_height

                points = ((paste_x, paste_y), (x_end, y_end))

                # 计算背景区域的平均亮度
                background_roi = fusion_img[paste_y:y_end, paste_x:x_end]
                background_brightness = np.mean(cv2.cvtColor(background_roi, cv2.COLOR_BGR2GRAY))
                foreground_brightness = np.mean(cv2.cvtColor(transformed_foreground_img[:, :, :3], cv2.COLOR_BGR2GRAY))

                # 如果背景区域的平均亮度低于前景图像的平均亮度，则放弃此轮粘贴
                if background_brightness < foreground_brightness:
                    continue

                # 检查标签是否重叠
                overlap = False
                for label in labels:
                    _, center_x, center_y, width, height = label
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2
                    label_points = ((x1, y1), (x2, y2))
                    if is_overlap(points, label_points):
                        overlap = True
                        break

                if overlap:
                    continue

                # 混合图像
                fusion_img_patch = fusion_img[paste_y:y_end, paste_x:x_end]
                for c in range(0, 3):
                    fusion_img[paste_y:y_end, paste_x:x_end, c] = \
                        (alpha_foreground * transformed_foreground_img[:, :, c] + alpha_background * fusion_img_patch[:, :, c])

                # 保存标签
                label = points_to_yolo_label(points, class_idx, fusion_img.shape)
                labels.append(label)
                break

        # 保存结果图像
        result_img_name = f'{save_name}_{j}.png'
        result_img_path = os.path.join(save_img_path, result_img_name)

        # 保存标签
        result_label_name = f'{save_name}_{j}.txt'
        result_label_path = os.path.join(save_label_path, result_label_name)

        cv2.imwrite(result_img_path, fusion_img)
        save_yolo_labels(labels, result_label_path)
