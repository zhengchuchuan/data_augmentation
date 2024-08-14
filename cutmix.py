from datetime import datetime
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from utils.utils import generate_random_transform_matrix, rectangle_intersection, validate_polygon, get_bounding_box
from utils.transforms import elastic_transform, perspective_transform
from utils.data_process import xywh_rect_to_x1y1x2y2
from utils.file_io import make_sure_paths_exist, read_lines_to_list
from utils.img_label_utils import read_yolo_labels, read_labelme_json, yolo_to_labelme_json, save_labelme_json, \
    labelme_json_to_yolo, save_yolo_labels, points_to_yolo_label

per_background_nums = 1
min_paste_nums = 2
max_paste_nums = 4


classes_path = 'data/labels/classes.txt'
save_images_path = r'\\192.168.3.155\高光谱测试样本库\原油检测\00大庆现场测试\03标注数据以及模型文件\00数据和标签\dataset_20240806_one_label\generate_images\20240813_2\images'

foreground_list_path = r'data/data_list/20240813_generated_samples_list.txt'
background_list_path = r'data/data_list/20240813_background_list.txt'

save_label_path = save_images_path.replace('\images', '\labels')
make_sure_paths_exist(save_images_path, save_label_path)

background_list = read_lines_to_list(background_list_path)
foreground_list = read_lines_to_list(foreground_list_path)




for i, background_path in enumerate(tqdm(background_list)):
    background_img = cv2.imdecode(np.fromfile(background_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    background_base_name = os.path.splitext(os.path.basename(background_path))[0]
    background_height, background_width, _ = background_img.shape

    background_path_without_suffix = os.path.splitext(background_path.replace('images', 'labels'))[0]

    if os.path.exists(background_path_without_suffix + '.txt'):
        background_label_path = background_path_without_suffix + '.txt'
    elif os.path.exists(background_path_without_suffix + '.json'):
        background_label_path = background_path_without_suffix + '.json'
    else:
        background_label_path = None

    if background_label_path is not None:
        if background_label_path.endswith('.txt'):
            background_labels = read_yolo_labels(background_label_path)
        elif background_label_path.endswith('.json'):
            background_labelme_data = read_labelme_json(background_label_path)
            classes = read_lines_to_list(classes_path)
            background_labels = labelme_json_to_yolo(background_labelme_data, classes)
        else:
            raise ValueError(f'Unsupported label file format: {background_label_path}')
    else:
        background_labels = []
    for j in range(per_background_nums):
        fusion_img = background_img.copy()
        fusion_labels = background_labels.copy()

        alpha_ratio = random.uniform(0.2, 0.3)
        sigma_ratio = random.uniform(0.02, 0.06)
        # 背景弹性变换
        # fusion_img = elastic_transform(fusion_img, alpha=background_height * alpha_ratio,
        #                                sigma=background_height * sigma_ratio)


        paste_nums = random.randint(min_paste_nums, max_paste_nums)
        random_foreground_list = random.sample(foreground_list, paste_nums)

        for foreground_path in random_foreground_list:
            # 根据文件夹名获取类别索引
            foreground_dir_path, _ = os.path.split(foreground_path)
            norm_path = os.path.normpath(foreground_dir_path)
            class_idx = norm_path.split(os.sep)[-1]

            foreground_img = cv2.imdecode(np.fromfile(foreground_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            foreground_height, foreground_width, foreground_channels = foreground_img.shape

            if foreground_channels != 4:
                raise ValueError(f"Foreground image {foreground_path} does not have alpha channels")

            alpha_ratio = random.uniform(0.2, 0.3)
            sigma_ratio = random.uniform(0.02, 0.06)
            transformed_foreground_img = elastic_transform(foreground_img, alpha=background_height * alpha_ratio,
                                           sigma=background_height * sigma_ratio)
            transed_foreground_height, transed_foreground_width, _ = transformed_foreground_img.shape

            alpha_foreground = transformed_foreground_img[:, :, 3] / 255.0
            alpha_background = 1.0 - alpha_foreground

            max_attempts = 10
            for attempt in range(max_attempts):
                paste_x = random.randint(0, background_width - transed_foreground_width)
                paste_y = random.randint(0, background_height - transed_foreground_height)

                if paste_x + transed_foreground_width > background_width or paste_y + transed_foreground_height > background_height:
                    continue

                x_end = paste_x + transed_foreground_width
                y_end = paste_y + transed_foreground_height

                points = ((paste_x, paste_y), (x_end, y_end))

                background_roi = fusion_img[paste_y:y_end, paste_x:x_end]
                background_brightness = np.mean(cv2.cvtColor(background_roi, cv2.COLOR_BGR2GRAY))
                foreground_brightness = np.mean(cv2.cvtColor(transformed_foreground_img[:, :, :3], cv2.COLOR_BGR2GRAY))

                if background_brightness < foreground_brightness or background_brightness < 50:
                    continue

                overlap = False
                for label in fusion_labels:
                    _, center_x, center_y, width, height = label
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2
                    label_points = ((x1, y1), (x2, y2))
                    if rectangle_intersection(points, label_points):
                        overlap = True
                        break

                if overlap:
                    continue

                fusion_img_patch = fusion_img[paste_y:y_end, paste_x:x_end]
                for c in range(0, 3):
                    fusion_img[paste_y:y_end, paste_x:x_end, c] = \
                        (alpha_foreground * transformed_foreground_img[:, :, c] + alpha_background * fusion_img_patch[:,
                                                                                                     :, c])

                label = points_to_yolo_label(points, class_idx, fusion_img.shape)
                fusion_labels.append(label)
                break

        now = datetime.now()
        formatted_date = now.strftime("%Y%m%d")

        result_img_name = f'{formatted_date}_{background_base_name}_{j}.png'
        result_img_path = os.path.join(save_images_path, result_img_name)
        image_type = '.png'
        success, img_encoded = cv2.imencode(image_type, fusion_img)
        if not success:
            raise ValueError(f"Could not encode image to format: {image_type}")
        img_encoded.tofile(result_img_path)

        result_label_name = f'{formatted_date}_{background_base_name}_{j}.txt'
        result_label_path = os.path.join(save_label_path, result_label_name)
        save_yolo_labels(fusion_labels, result_label_path)
