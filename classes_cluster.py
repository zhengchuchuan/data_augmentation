import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils.file_io import read_path_list
from utils.label_utils import read_yolo_labels, read_labelme_json, yolo_to_labelme_json, save_labelme_json


def save_images(labeled_imgs, labels, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for label in np.unique(labels):
        label_dir = os.path.join(save_dir, f'cluster_{label}')
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

    for idx, (img, label) in enumerate(zip(labeled_imgs, labels)):
        img_path = os.path.join(save_dir, f'cluster_{label}', f'{idx}.png')
        # cv2.imwrite(img_path, img)
        image_type = '.png'
        success, img_encoded = cv2.imencode(image_type, img)
        if not success:
            raise ValueError(f"Could not encode image to format: {image_type}")
        img_encoded.tofile(img_path)


if __name__ == '__main__':

    classes_path = r'D:\wayho\oil_detection\yolov5\dataset\oil\have_labels\labels\classes.txt'
    img_path_list = r'data/data_list/labeled_image_path_list.txt'
    save_dir = r'exp/clustered_images'  # 保存图像的目录

    img_paths = read_path_list(img_path_list)
    img_paths.sort()
    print(f"image length:{len(img_paths)}")
    label_paths = [path.replace('images', 'labels').replace(os.path.splitext(path)[1], '.txt') for path in img_paths]
    print(f"label length:{len(label_paths)}")
    classes = read_path_list(classes_path)
    print(classes)

    labeled_imgs = []
    for i in tqdm(range(len(img_paths))):
        try:
            img = cv2.imdecode(np.fromfile(img_paths[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(e)
            continue

        try:
            yolo_labels = read_yolo_labels(label_paths[i])
        except Exception as e:
            print(e)
            continue

        for label in yolo_labels:
            class_id, x_center, y_center, width, height = label
            x1 = int((x_center - width / 2) * img.shape[1])
            y1 = int((y_center - height / 2) * img.shape[0])
            x2 = int((x_center + width / 2) * img.shape[1])
            y2 = int((y_center + height / 2) * img.shape[0])
            labeled_img = img[y1:y2, x1:x2, :]
            resized_img = cv2.resize(labeled_img, (128, 128))
            labeled_imgs.append(resized_img)

    print(f"Number of labeled images: {len(labeled_imgs)}")

    # 将图像展平
    flat_imgs = [img.flatten() for img in labeled_imgs]
    flat_imgs = np.array(flat_imgs)

    # 主成分分析 (PCA)
    pca = PCA(n_components=50)  # 降维到 50 维
    pca_result = pca.fit_transform(flat_imgs)

    # K-means 聚类
    kmeans = KMeans(n_clusters=3)  # 假设我们要分成 5 类
    kmeans.fit(pca_result)
    labels = kmeans.labels_

    print(f"Cluster labels: {labels}")

    # 保存图像到对应的类别文件夹
    save_images(labeled_imgs, labels, save_dir)
