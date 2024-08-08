import cv2
import numpy as np
from shapely.validation import make_valid

def generate_random_transform_matrix(image_shape, scale_range=(0.8, 1.2), rotation_range=(-10, 10),
                                     translation_range=(0.1, 0.3), shear_range=(-10, 10)):
    """
    生成一个随机的3x3变换矩阵，变换包括缩放、旋转、平移和剪切。

    :param image_shape: 图像的形状 (height, width)
    :param scale_range: 缩放范围 (min_scale, max_scale)
    :param rotation_range: 旋转角度范围 (min_angle, max_angle)，单位为度
    :param translation_range: 平移范围 (min_translation_ratio, max_translation_ratio) 相对于图像尺寸的比例
    :param shear_range: 剪切角度范围 (min_shear, max_shear)，单位为度
    :return: 3x3 变换矩阵
    """
    h, w, _ = image_shape

    # 随机生成缩放因子
    scale_x = np.random.uniform(scale_range[0], scale_range[1])
    scale_y = np.random.uniform(scale_range[0], scale_range[1])
    scale_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])

    # 随机生成旋转角度
    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    theta = np.radians(angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    # 随机生成平移量，按图像尺寸的比例计算
    translate_x = np.random.uniform(translation_range[0], translation_range[1]) * w
    translate_y = np.random.uniform(translation_range[0], translation_range[1]) * h
    translation_matrix = np.array([
        [1, 0, translate_x],
        [0, 1, translate_y],
        [0, 0, 1]
    ])

    # 随机生成剪切角度
    shear_x = np.random.uniform(shear_range[0], shear_range[1])
    shear_y = np.random.uniform(shear_range[0], shear_range[1])
    shear_x = np.radians(shear_x)
    shear_y = np.radians(shear_y)
    shear_matrix = np.array([
        [1, np.tan(shear_x), 0],
        [np.tan(shear_y), 1, 0],
        [0, 0, 1]
    ])

    # 组合变换矩阵
    transform_matrix = scale_matrix @ rotation_matrix @ translation_matrix @ shear_matrix

    return transform_matrix

def rectangle_intersection(rect1, rect2):
    """
    判断两个矩形是否相交。
    :param rect1: 第一个矩形的坐标，格式为 ((x1, y1), (x2, y2))
    :param rect2: 第二个矩形的坐标，格式为 ((x1, y1), (x2, y2))
    :return: 如果矩形相交，返回 True；否则返回 False
    """
    (x_min_a, y_min_a), (x_max_a, y_max_a) = rect1
    (x_min_b, y_min_b), (x_max_b, y_max_b) = rect2

    cross_x_min = max(x_min_a, x_min_b)  # 交叉部分的左上角 x 坐标
    cross_y_min = max(y_min_a, y_min_b)  # 交叉部分的左上角 y 坐标

    cross_x_max = min(x_max_a, x_max_b)  # 交叉部分的右下角 x 坐标
    cross_y_max = min(y_max_a, y_max_b)  # 交叉部分的右下角 y 坐标

    if cross_x_min >= cross_x_max or cross_y_min >= cross_y_max:
        return False

    return True

def validate_polygon(polygon):
    if not polygon.is_valid:
        polygon = make_valid(polygon)
    return polygon


def get_bounding_box(rect1, rect2):
    # rect1 和 rect2 都是两个顶点坐标的列表，表示矩形的左上角和右下角
    # 例如: rect1 = [(x1_min, y1_min), (x1_max, y1_max)]

    # 提取两个矩形的顶点坐标
    x1_min, y1_min = rect1[0]
    x1_max, y1_max = rect1[1]

    x2_min, y2_min = rect2[0]
    x2_max, y2_max = rect2[1]

    # 计算并集的最小外接矩形
    x_min = min(x1_min, x2_min)
    y_min = min(y1_min, y2_min)
    x_max = max(x1_max, x2_max)
    y_max = max(y1_max, y2_max)

    # 返回最小外接矩形的顶点坐标
    bounding_box = [(x_min, y_min), (x_max, y_max)]
    return bounding_box


def is_overlap(rect1, rect2):
    """
    判断两个矩形是否相交。
    :param rect1: 第一个矩形的坐标，格式为 ((x1, y1), (x2, y2))
    :param rect2: 第二个矩形的坐标，格式为 ((x1, y1), (x2, y2))
    :return: 如果矩形相交，返回 True；否则返回 False
    """
    (x1_a, y1_a), (x2_a, y2_a) = rect1
    (x1_b, y1_b), (x2_b, y2_b) = rect2

    # 确保 (x1, y1) 是左下角，(x2, y2) 是右上角
    x1_a, x2_a = min(x1_a, x2_a), max(x1_a, x2_a)
    y1_a, y2_a = min(y1_a, y2_a), max(y1_a, y2_a)
    x1_b, x2_b = min(x1_b, x2_b), max(x1_b, x2_b)
    y1_b, y2_b = min(y1_b, y2_b), max(y1_b, y2_b)

    # 判断矩形是否相交
    if x1_a > x2_b or x1_b > x2_a:
        return False
    if y1_a > y2_b or y1_b > y2_a:
        return False

    return True
