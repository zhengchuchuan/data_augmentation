import numpy as np


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