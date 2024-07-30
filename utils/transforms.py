import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def elastic_transform(image, alpha, sigma, random_state=None):
    """
    对图像进行弹性变形。

    :param image: 输入图像
    :param alpha: 变形强度，控制变形的程度
    :param sigma: 高斯滤波标准差，控制变形的平滑度
    :param random_state: 随机数生成器的种子，默认为 None
    :return: 变形后的图像
    """
    # 设置随机数生成器
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Generate random displacement fields
    dx = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha

    # Generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Apply displacement fields to each channel separately
    transformed_image = np.zeros_like(image)
    # 各个通道分别进行变形
    for i in range(shape[2]):
        transformed_image[..., i] = map_coordinates(image[..., i], indices, order=3, mode='reflect').reshape(shape_size)

    return transformed_image


def perspective_transform(image,M):
    # 获取图像尺寸
    h, w = image.shape[:2]

    # 计算透视变换后的图像边界
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(np.array([corners], dtype=np.float32), M).squeeze()

    # 计算新的图像尺寸
    x_min, y_min = np.min(transformed_corners, axis=0).astype(int)
    x_max, y_max = np.max(transformed_corners, axis=0).astype(int)

    # 计算输出图像尺寸
    new_w = x_max - x_min
    new_h = y_max - y_min

    # 调整透视变换矩阵
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    new_M = np.dot(translation_matrix, M)

    # 应用透视变换
    transformed_image = cv2.warpPerspective(image, new_M, (new_w, new_h), flags=cv2.INTER_CUBIC)


    return transformed_image


def perspective_transform_with_mask(image, mask, M, points):
    # 获取图像尺寸
    h, w = image.shape[:2]

    # 计算透视变换后的图像边界
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(np.array([corners], dtype=np.float32), M).squeeze()

    # 计算新的图像尺寸
    x_min, y_min = np.min(transformed_corners, axis=0).astype(int)
    x_max, y_max = np.max(transformed_corners, axis=0).astype(int)

    # 计算输出图像尺寸
    new_w = x_max - x_min
    new_h = y_max - y_min

    # 调整透视变换矩阵
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    new_M = np.dot(translation_matrix, M)

    # 应用透视变换
    transformed_image = cv2.warpPerspective(image, new_M, (new_w, new_h))
    trans_mask = cv2.warpPerspective(mask, new_M, (new_w, new_h))

    # 变换检测框顶点
    transformed_points = cv2.perspectiveTransform(np.array([points], dtype=np.float32), new_M).squeeze()

    return transformed_image, trans_mask, transformed_points


