import os
import random

import cv2
import numpy as np
from shapely import MultiPolygon
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils.utils import generate_random_transform_matrix, validate_polygon, get_bounding_box
from utils.transforms import elastic_transform, perspective_transform_with_mask
from utils.data_process import xywh_rect_to_x1y1x2y2
from utils.file_io import make_sure_paths_exist, read_lines_to_list
from utils.img_label_utils import read_yolo_labels, read_labelme_json, yolo_to_labelme_json, save_labelme_json, \
    labelme_json_to_yolo, save_yolo_labels

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
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    img_dir = os.path.dirname(label_path).replace('labels', 'imgs')

    for ext in image_extensions:
        img_path_temp = os.path.join(img_dir, file_name_without_suffix + ext)
        if os.path.exists(img_path_temp):
            img_path = img_path_temp

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

    generate_nums = 200
    random_times = 2

    for i in tqdm(range(generate_nums)):
        augmented_img = img

        new_shaps = []
        status = None
        for j, label in enumerate(labelme_labels):
            status = True
            class_name = label['label']
            label_points = label['points']

            # 对一个标签进行多次随机变换
            for k in range(random_times):
                label_points = np.array(label_points, dtype=np.int32)
                if len(label_points) > 1:
                    # 两个点也会判断成矩形
                    xywh_rect = cv2.boundingRect(label_points)
                else:
                    raise ValueError(f'标签点数小于2{label_path}')
                label_width = xywh_rect[2]
                label_height = xywh_rect[3]
                rect_label_points = xywh_rect_to_x1y1x2y2(xywh_rect)
                # 获取标签区域的左上角和右下角坐标
                x1 = rect_label_points[0][0]
                y1 = rect_label_points[0][1]
                x2 = rect_label_points[1][0]
                y2 = rect_label_points[1][1]
                label_center = (x1 + label_width // 2, y1 + label_height // 2)
                # 标签区域
                labeled_img = img[y1:y2, x1:x2, :]
                # 标签区域的掩码
                mask = np.zeros(labeled_img.shape, dtype=np.uint8)
                relative_label_points = label_points - np.array([x1, y1])
                if len(label_points) >= 3:

                    cv2.fillPoly(mask, [relative_label_points], (255, 255, 255))
                else:
                    cv2.rectangle(mask, relative_label_points[0],relative_label_points[1], (255, 255, 255), thickness=-1)
                # mask = np.ones(labeled_img.shape, dtype=np.uint8) * 255

                # 弹性变换
                elastic_transformed_img = elastic_transform(labeled_img, alpha=label_height * 0.1,
                                                            sigma=label_height * 0.02, alpha_affine=label_height * 0.02)

                # 生成随机透视变换矩阵
                transform_matrix = generate_random_transform_matrix(elastic_transformed_img.shape,
                                                                    scale_range=(0.8, 1.2),
                                                                    rotation_range=(-15, 15),
                                                                    translation_range=(0.1, 0.3),
                                                                    shear_range=(-10, 10))
                # 应用透视变换, 返回变换后的图像、更新后的检测框列表和掩码
                perspective_transformed_img, transformed_mask, transformed_label_points = perspective_transform_with_mask(
                    elastic_transformed_img, mask, transform_matrix, relative_label_points)

                transformed_label_points = transformed_label_points + np.array([x1, y1])
                d_ratio = 0.4
                dx = np.random.randint(-label_width * d_ratio, label_width * d_ratio)
                dy = np.random.randint(- label_height * d_ratio, label_height * d_ratio)



                random_cneter = (label_center[0] + dx,
                                 label_center[1] + dy)

                transformed_label_points_bias = transformed_label_points + np.array([dx, dy])

                transformed_width = transformed_label_points_bias[1][0] - transformed_label_points_bias[0][0]
                transformed_height = transformed_label_points_bias[1][1] - transformed_label_points_bias[0][1]

                ### 待优化
                if (transformed_label_points_bias[0][0] < 0 or transformed_label_points_bias[1][0] > img_weight
                        or transformed_label_points_bias[0][1] < 0 or transformed_label_points_bias[1][1] > img_height):
                    status = False
                    continue


                # 更新标签未完善
                """
                -NORMAL_CLONE: 不保留dst 图像的texture细节。目标区域的梯度只由源图像决定。
                -MIXED_CLONE: 保留dest图像的texture 细节。目标区域的梯度是由原图像和目的图像的组合计算出来(计算dominat gradient)。
                -MONOCHROME_TRANSFER: 不保留src图像的颜色细节，只有src图像的质地，颜色和目标图像一样，可以用来进行皮肤质地填充。

                """
                try:
                    augmented_img = cv2.seamlessClone(perspective_transformed_img, augmented_img, transformed_mask,random_cneter, cv2.MIXED_CLONE)
                except Exception as e:
                    print(e)
                    status = False
                    continue


                # 定义两个多边形并检查有效性
                union_rect = get_bounding_box(rect_label_points, transformed_label_points_bias)

                # 将结果转换为numpy数组，并移除最后一个重复的点
                label_points = np.array(union_rect, dtype=np.float32)

            new_shape = {
                'label': class_name,
                'points': label_points.tolist(),
                'shape_type': 'rectangle'
            }
            new_shaps.append(new_shape)

        if status:
            cv2.imwrite(f'{result_imgs_path}/{file_name_without_suffix}_{i}.png', augmented_img)
            labelme_data['version'] = '5.2.0'
            labelme_data['shapes'] = new_shaps
            labelme_data['imagePath'] = f'../imgs/{file_name_without_suffix}_{i}.png'
            labelme_data['imageData'] = None
            save_labelme_json(labelme_data, f'{result_labels_path}/{file_name_without_suffix}_{i}.json')

            yolo_data = labelme_json_to_yolo(labelme_data, classes)
            save_yolo_labels(yolo_data, f'{result_labels_path}/{file_name_without_suffix}_{i}.txt')


cv2.imwrite(f'{result_imgs_path}/mask.png', transformed_mask)
cv2.imwrite(f'{result_imgs_path}/labeled_img.png', labeled_img)
cv2.imwrite(f'{result_imgs_path}/perspective_transformed_img.png', perspective_transformed_img)
