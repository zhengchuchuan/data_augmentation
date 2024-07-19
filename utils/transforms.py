import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """
    对图像进行弹性变形。

    :param image: 输入图像
    :param alpha: 变形强度，控制变形的程度
    :param sigma: 高斯滤波标准差，控制变形的平滑度
    :param alpha_affine: 仿射变形强度
    :param random_state: 随机数生成器的种子，默认为 None
    :return: 变形后的图像
    """
    # 设置随机数生成器
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # 非共线的三对对应点确定一个唯一的仿射变换
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])

    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # 计算仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 应用仿射变换
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # Generate random displacement fields
    dx = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha

    # Generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Apply displacement fields to each channel separately
    transformed_image = np.zeros_like(image)
    for i in range(shape[2]):
        transformed_image[..., i] = map_coordinates(imageB[..., i], indices, order=1, mode='reflect').reshape(
            shape_size)

    return transformed_image

def perspective_transform(image, mask, M, points):

    # 获取图像尺寸
    h, w = image.shape[:2]

    # 应用透视变换
    transformed_image = cv2.warpPerspective(image, M, (w, h))

    trans_mask = cv2.warpPerspective(mask, M, (w, h))

    transformed_points = cv2.perspectiveTransform(np.array([points], dtype=np.float32), M).squeeze()

    return transformed_image, trans_mask, transformed_points