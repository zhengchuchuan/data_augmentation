import random

import numpy as np
import cv2
import torch

max_uint16 = np.uint16(-1)


def center_bbox_to_corner_bbox(labels, image_height, image_width):
    box_corner_labels = []

    for label in labels:
        class_id, x_center, y_center, w, h = label

        x_center = int(x_center * image_width)
        y_center = int(y_center * image_height)
        w = int(w * image_width)
        h = int(h * image_height)

        x_min = x_center - w // 2
        x_max = x_center + w // 2
        y_min = y_center - h // 2
        y_max = y_center + h // 2

        # 确保坐标在图像范围内
        x_min = max(0, x_min)
        x_max = min(image_width, x_max)
        y_min = max(0, y_min)
        y_max = min(image_height, y_max)

        box_corner_labels.append([class_id, x_min, y_min, x_max, y_max])

    return box_corner_labels




def center_crop(image, patch_size):
    """
    将图像的中心裁剪为给定的patch_size。
    如果图像的长或宽小于patch_size，则填充到patch_size的大小，再中心裁剪。

    :param image: 输入图像，类型为NumPy数组。
    :param patch_size: 裁剪大小，元组形式(高度, 宽度)。
    :return: 中心裁剪后的图像，类型为NumPy数组。
    """
    image_height, image_width = image.shape[:2]  # 获取图像的高度和宽度
    crop_height, crop_width = patch_size  # 获取裁剪区域的高度和宽度

    if image_height >= crop_height and image_width >= crop_width:
        # 如果图像的长和宽都大于等于patch_size，则直接中心裁剪
        start_x = (image_width - crop_width) // 2
        start_y = (image_height - crop_height) // 2
        cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    else:
        # 计算需要填充的大小
        pad_height = max(crop_height - image_height, 0)
        pad_width = max(crop_width - image_width, 0)

        # 根据图像的数据类型确定填充值
        if np.issubdtype(image.dtype, np.integer):
            fill_value = np.iinfo(image.dtype).max
        else:
            fill_value = np.finfo(image.dtype).max

        # 初始化填充后的图像
        padded_image = np.full((image_height + pad_height, image_width + pad_width, image.shape[2]), fill_value, dtype=image.dtype)

        # 计算填充后的图像中心起始点
        start_x = (padded_image.shape[1] - image_width) // 2
        start_y = (padded_image.shape[0] - image_height) // 2

        # 将原图像放置在填充后的图像中心
        padded_image[start_y:start_y + image_height, start_x:start_x + image_width] = image

        # 进行中心裁剪
        start_x_crop = (padded_image.shape[1] - crop_width) // 2
        start_y_crop = (padded_image.shape[0] - crop_height) // 2
        cropped_image = padded_image[start_y_crop:start_y_crop + crop_height, start_x_crop:start_x_crop + crop_width]

    return cropped_image

def remove_black_borders(img):
    """
    去除图像的黑边，规则为单个波段的边缘的一行或者一列的值全为0，则整个图像上要去除以上边。

    参数:
        img: np.ndarray, hwc格式的图像，通道数大于3

    返回:
        np.ndarray, 去除黑边后的图像
    """
    h, w, c = img.shape

    # 找到非零的边界
    top, bottom, left, right = 0, h, 0, w

    # 检查顶部边缘
    while top < bottom:
        if any(np.all(img[top, :, i] == 0) for i in range(c)):
            top += 1
        else:
            break

    # 检查底部边缘
    while bottom > top:
        if any(np.all(img[bottom - 1, :, i] == 0) for i in range(c)):
            bottom -= 1
        else:
            break

    # 检查左侧边缘
    while left < right:
        if any(np.all(img[:, left, i] == 0) for i in range(c)):
            left += 1
        else:
            break

    # 检查右侧边缘
    while right > left:
        if any(np.all(img[:, right - 1, i] == 0) for i in range(c)):
            right -= 1
        else:
            break

    # 裁剪图像
    cropped_img = img[top:bottom, left:right, :]

    return cropped_img


def bands_align(img, config_params):
    """
    # JPG,450,550,650,720,750,800,850
    -3,-2,
    4,-9,
    -9,-9,
    6,-1,
    0,0,
    -4,1,
    11,0,
    -1,11,
    """
    # 将图像列表转换为numpy数组
    img = np.array(img)
    h, w, c = img.shape
    aligned_img = np.zeros_like(img)
    # Separate handling for RGB channels (last 3 channels)
    for i, offset in enumerate(config_params):
        tx, ty = offset
        # 处理jpg
        if c == 10:  # Process the first 7 bands
            if i == 0:
                band = img[:, :, -3:]
                for channel in range(3):
                    aligned_img[:, :, -3 + channel] = cv2.warpAffine(band[:, :, channel],
                                                                 np.float32([[1, 0, -tx], [0, 1, -ty]]), dsize=(w, h),
                                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                if c == 10:
                    band = img[:, :, i]
                    aligned_img[:, :, i] = cv2.warpAffine(band, np.float32([[1, 0, -tx], [0, 1, -ty]]), dsize=(w, h),
                                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif c == 7:
            # 跳过jpg
            if i == 0:
                continue
            band = img[:, :, i-1]
            aligned_img[:, :, i-1] = cv2.warpAffine(band, np.float32([[1, 0, -tx], [0, 1, -ty]]), dsize=(w, h),
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # 不能直接去除黑边,会影响标签的坐标
    # cropped_img = remove_black_borders(aligned_img)
    return aligned_img


def random_sub_label(bbox_label, width_ratio, height_ratio):
    """
    根据给定的宽度和高度比例，在 bbox_label 中随机选择一个子矩形标签。

    :param bbox_label: 检测框标签，[classid, x1, y1, x2, y2]
    :param width_ratio: 子矩形宽度相对原矩形宽度的比例
    :param height_ratio: 子矩形高度相对原矩形高度的比例

    :return: 新的更小矩形的坐标 (new_x1, new_y1, new_x2, new_y2)
    """
    class_id, x1, y1, x2, y2 = bbox_label

    # 确保 x1 < x2 且 y1 < y2
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid rectangle coordinates")

    # 计算原矩形的宽度和高度
    original_width = x2 - x1
    original_height = y2 - y1

    # 计算子矩形的宽度和高度
    sub_width = int(original_width * width_ratio)
    sub_height = int(original_height * height_ratio)

    # 确保子矩形的宽度和高度不超过原矩形
    if sub_width > original_width or sub_height > original_height:
        raise ValueError("Sub-rectangle size exceeds original rectangle")

    # 随机选择子矩形的左上角坐标
    new_x1 = random.randint(x1, x2 - sub_width)
    new_y1 = random.randint(y1, y2 - sub_height)

    # 计算子矩形的右下角坐标
    new_x2 = new_x1 + sub_width
    new_y2 = new_y1 + sub_height

    sub_bbox_label = (class_id, new_x1, new_y1, new_x2, new_y2)

    return sub_bbox_label

def xywh_rect_to_x1y1x2y2(xywh_rect):
    """
    将 (x, y, w, h) 格式的矩形坐标转换为 (x1, y1, x2, y2) 格式的坐标。

    :param xywh_rect: (x, y, w, h) 格式的矩形坐标
    :return: (x1, y1, x2, y2) 格式的矩形坐标
    """
    x, y, w, h = xywh_rect
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    return ((x1, y1), (x2, y2))


