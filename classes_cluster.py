import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils.img_label_utils import read_yolo_labels, save_yolo_labels



def save_clustered_images(labeled_imgs, clustered_labels, clustered_images_save_dir):
    if not os.path.exists(clustered_images_save_dir):
        os.makedirs(clustered_images_save_dir)

    for label in np.unique(clustered_labels):
        label_dir = os.path.join(clustered_images_save_dir, f'cluster_{label}')
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

    for idx, (img, label) in enumerate(zip(labeled_imgs, clustered_labels)):
        img_path = os.path.join(clustered_images_save_dir, f'cluster_{label}', f'{idx}.png')
        image_type = '.png'
        success, img_encoded = cv2.imencode(image_type, img)
        if not success:
            raise ValueError(f"Could not encode image to format: {image_type}")
        img_encoded.tofile(img_path)

if __name__ == '__main__':

    classes_path = r'data/classes.txt'
    img_path_list = r'C:\Users\zcc\project\python_project\my_utils\exp\data_list_all_20240805.txt'
    clustered_images_save_dir = r'exp/clustered_images'  # 保存图像的目录

    with open(img_path_list, 'r') as fin:
        img_paths = [line.strip() for line in fin]

    img_paths.sort()
    print(f"image length: {len(img_paths)}")
    label_paths = [path.replace('images', 'labels').replace(os.path.splitext(path)[1], '.txt') for path in img_paths]
    print(f"label length: {len(label_paths)}")

    with open(classes_path, 'r') as fin:
        classes = [line.strip() for line in fin]
    print(classes)

    labeled_imgs = []
    img_labels = []  # 用于保存每个标签的聚类结果
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

        labeled_img_label = []
        for label in yolo_labels:
            class_id, x_center, y_center, width, height = label
            x1 = int((x_center - width / 2) * img.shape[1])
            y1 = int((y_center - height / 2) * img.shape[0])
            x2 = int((x_center + width / 2) * img.shape[1])
            y2 = int((y_center + height / 2) * img.shape[0])
            labeled_img = img[y1:y2, x1:x2, :]
            resized_img = cv2.resize(labeled_img, (128, 128))
            # 标签区域的图像
            labeled_imgs.append(resized_img)
            # 每张图像对应的所有标签
            labeled_img_label.append(class_id)
        img_labels.append(labeled_img_label)  # 记录标签索引


    print(f"Number of labeled images: {len(labeled_imgs)}")

    # 将图像展平
    flat_imgs = [img.flatten() for img in labeled_imgs]
    flat_imgs = np.array(flat_imgs)

    # 主成分分析 (PCA)
    pca = PCA(n_components=50)  # 降维到 50 维
    pca_result = pca.fit_transform(flat_imgs)

    # K-means 聚类
    kmeans = KMeans(n_clusters=3)  # 假设我们要分成 3 类
    kmeans.fit(pca_result)
    clustered_labels = kmeans.labels_

    print(f"Cluster labels: {len(clustered_labels)}")

    clustered_labels_idx = 0
    for label_path in tqdm(label_paths):
        try:
            yolo_labels = read_yolo_labels(label_path)
            new_labels = []
            for i, yolo_label in enumerate(yolo_labels):
                class_id, x_center, y_center, width, height = yolo_label
                new_class_id = img_labels[clustered_labels_idx]  # 使用聚类后的类别
                clustered_labels_idx += 1
                new_labels.append([new_class_id, x_center, y_center, width, height])
            label_save_path = label_path.replace('labels','clustered_labels')
            # 保存新的标签
            save_yolo_labels(new_labels, label_save_path)
        except Exception as e:
            print(f"Error updating label  {label_path}: {e}")



    # 保存图像和更新标签到对应的类别文件夹
    save_clustered_images(labeled_imgs,clustered_labels, clustered_images_save_dir)
