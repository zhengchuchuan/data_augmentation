import os
import random

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from tqdm import tqdm

from utils.utils import generate_random_transform_matrix, rectangle_union
from utils.transforms import elastic_transform, perspective_transform
from utils.data_process import center_bbox_to_corner_bbox, xywh_rect_to_x1y1x2y2
from utils.file_io import make_sure_paths_exist, read_lines_to_list
from utils.img_label_utils import read_yolo_labels, read_labelme_json, yolo_to_labelme_json

img_path = None
label_path_list = r'data/data_list/label_list_0717.txt'
calsses_path = r'data/labels/classes.txt'
result_path = 'exp'
classes = read_lines_to_list(calsses_path)

label_paths = read_lines_to_list(label_path_list)
result_imgs_path = f'{result_path}/imgs'
result_labels_path = f'{result_path}/labels'
make_sure_paths_exist(result_imgs_path, result_labels_path)

if len(label_paths) == 0:
    raise ValueError('标签文件列表为空')
for label_path in label_paths:
    if not os.path.exists(label_path):
        raise ValueError(f'标签文件不存在：{label_path}')


    label_name = os.path.basename(label_path)
    file_name_without_suffix, label_suffix = os.path.splitext(label_name)
    img_path = label_path.replace('labels', 'imgs').replace(label_suffix, '.png')
    if not os.path.exists(img_path):
        img_path = img_path.replace('.png', '.jpg')
    if not os.path.exists(img_path):
        raise ValueError(f'图像文件不存在：{img_path}')



    # 读取图像
    img = cv2.imread(img_path)
    img_height, img_weight, _ = img.shape
    img_center = (img_weight // 2, img_height // 2)
    # 读取标签
    if label_suffix == '.txt':
        yolo_labels = read_yolo_labels(label_path)
        labelme_data = yolo_to_labelme_json(yolo_labels, classes, img.shape)
    elif label_suffix == '.json':
        labelme_data = read_labelme_json(label_path)

    labelme_labels = labelme_data['shapes']


    new_labels = []

    random_times = 20
    for i in tqdm(range(random_times)):
        result = img
        for _, label in enumerate(labelme_labels):
            class_name = label['label']
            points = label['points']

            points = np.array(points,dtype=np.float32)
            if len(points) > 1:
            # 两个点也会判断成矩形
                xywh_rect = cv2.boundingRect(points)
            else:
                raise ValueError(f'标签点数小于2{label_path}')
            label_width = xywh_rect[2]
            label_height = xywh_rect[3]
            x1y1x2y2_rect = xywh_rect_to_x1y1x2y2(xywh_rect)
            x1, y1, x2, y2 = x1y1x2y2_rect
            label_center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

            labeled_patch_img = img[y1:y2, x1:x2, :]
            # 弹性变换
            elastic_transformed_img = elastic_transform(labeled_patch_img, alpha=label_height*0.2, sigma=label_height*0.02, alpha_affine=label_height*0.02)
            # 生成随机变换矩阵
            transform_matrix = generate_random_transform_matrix(elastic_transformed_img.shape, scale_range=(0.8, 1.2),
                                                                rotation_range=(-10, 10),
                                                                translation_range=(0.1, 0.3), shear_range=(-10, 10))
            # 应用透视变换, 返回变换后的图像、更新后的检测框列表和掩码
            mask = np.ones(labeled_patch_img.shape, dtype=np.uint8) * 255

            perspective_transformed_img, transformed_bbox, transformed_mask = perspective_transform(elastic_transformed_img, mask, transform_matrix, x1y1x2y2_rect )


            trans_x1, trans_y1, trans_x2, trans_y2 = transformed_bbox
            d_rate = 0.2
            dx = np.random.randint(- label_width*d_rate, label_width*d_rate)
            dy = np.random.randint(- label_height*d_rate, label_height*d_rate)
            random_cneter = (label_center[0] + dx,
                             label_center[1] + dy)

            transformed_width = transformed_bbox[2] - transformed_bbox[0]
            transformed_height = transformed_bbox[3] - transformed_bbox[1]
            ### 待优化
            if random_cneter[0] - transformed_width // 2 < 0 or random_cneter[0] + transformed_height // 2 < 0:
                continue

            if random_cneter[1] - transformed_height // 2 < 0 or random_cneter[1] + transformed_height // 2 < 0:
                continue

            """
            -NORMAL_CLONE: 不保留dst 图像的texture细节。目标区域的梯度只由源图像决定。
            -MIXED_CLONE: 保留dest图像的texture 细节。目标区域的梯度是由原图像和目的图像的组合计算出来(计算dominat gradient)。
            -MONOCHROME_TRANSFER: 不保留src图像的颜色细节，只有src图像的质地，颜色和目标图像一样，可以用来进行皮肤质地填充。
    
            """
            rect1 = ((x1, y1), (x2, y2))
            rect2 = ((trans_x1,trans_y1),(trans_x2,trans_y2))
            new_rect = rectangle_union(rect1, rect2)
            new_label = (label, new_rect[0][0], new_rect[0][1], new_rect[1][0], new_rect[1][1])
            new_labels.append(new_label)

            result = cv2.seamlessClone(perspective_transformed_img, result, transformed_mask, random_cneter, cv2.MIXED_CLONE)

        cv2.imwrite(f'{result_imgs_path}/{file_name_without_suffix}_{i}.png', result)

        with open(f'{result_labels_path}/{file_name_without_suffix}_{i}.txt', 'w') as f:
            for label in new_labels:
                f.write(f'{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n')

    cv2.imwrite(f'{result_imgs_path}/mask.png', transformed_mask)
    cv2.imwrite(f'{result_imgs_path}/labeled_patch_img.png', labeled_patch_img)
    cv2.imwrite(f'{result_imgs_path}/perspective_transformed_img.png', perspective_transformed_img)

