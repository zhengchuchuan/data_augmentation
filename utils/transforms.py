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
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
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

    # 计算透视变换后的图像边界
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32)
    transformed_corners = np.dot(M, corners.T).T
    transformed_corners /= transformed_corners[:, 2, None]  # 归一化齐次坐标
    transformed_corners = transformed_corners[:, :2]  # 去除齐次坐标
    min_x = np.min(transformed_corners[:, 0])
    max_x = np.max(transformed_corners[:, 0])
    min_y = np.min(transformed_corners[:, 1])
    max_y = np.max(transformed_corners[:, 1])

    # 计算输出图像尺寸
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))

    # 调整透视变换矩阵
    M[0, 2] -= min_x
    M[1, 2] -= min_y

    # 应用透视变换
    transformed_image = cv2.warpPerspective(image, M, (new_w, new_h))

    # 生成掩码
    # mask = np.ones((h, w), dtype=np.uint8)
    # mask = (mask > 0).astype(np.uint8) * 255
    trans_mask = cv2.warpPerspective(mask, M, (new_w, new_h))
 # 将掩码转换为0和255的二值图像

    # 更新检测框
    x1 = points[0][0]
    y1 = points[0][1]
    x2 = points[2][0]
    y2 = points[2][1]
    # 定义检测框的四个顶点
    points = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32)
    # 变换检测框顶点
    transformed_points = np.dot(M, points.T).T
    transformed_points /= transformed_points[:, 2, None]  # 归一化齐次坐标
    transformed_points = transformed_points[:, :2]  # 去除齐次坐标
    # 获取新的检测框的边界
    new_x1, new_y1 = np.min(transformed_points, axis=0)
    new_x2, new_y2 = np.max(transformed_points, axis=0)
    transformed_labels = ((new_x1, new_y1), (new_x2, new_y2))

    return transformed_image, transformed_labels, trans_mask