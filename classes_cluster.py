import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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


def determine_optimal_clusters(data, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_k = np.argmax(silhouette_scores) + 2
    return optimal_k, silhouette_scores


if __name__ == '__main__':
    img_path_list = r'C:\Users\zcc\project\python_project\my_utils\exp\data_list_04-07_20240807.txt'
    clustered_images_save_dir = r'exp/clustered_images'  # 保存图像的目录

    with open(img_path_list, 'r') as fin:
        img_paths = [line.strip() for line in fin]

    img_paths.sort()
    print(f"image length: {len(img_paths)}")
    label_paths = [path.replace('images', 'labels').replace(os.path.splitext(path)[1], '.txt') for path in img_paths]
    print(f"label length: {len(label_paths)}")

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

    flat_imgs = [img.flatten() for img in labeled_imgs]
    flat_imgs = np.array(flat_imgs)

    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(flat_imgs)

    optimal_k, silhouette_scores = determine_optimal_clusters(pca_result, max_k=10)
    print(f"Optimal number of clusters: {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k)
    kmeans.fit(pca_result)
    clustered_labels = kmeans.labels_

    print(f"Cluster labels: {len(clustered_labels)}")

    clustered_labels_idx = 0
    for i, label_path in tqdm(enumerate(label_paths)):
        try:
            yolo_labels = read_yolo_labels(label_path)
            new_labels = []
            for j, yolo_label in enumerate(yolo_labels):
                class_id, x_center, y_center, width, height = yolo_label
                new_class_id = clustered_labels[clustered_labels_idx]
                clustered_labels_idx += 1
                new_labels.append([new_class_id, x_center, y_center, width, height])
            label_save_path = label_path.replace('labels', 'clustered_labels')
            if not os.path.exists(os.path.dirname(label_save_path)):
                os.makedirs(os.path.dirname(label_save_path))
            save_yolo_labels(new_labels, label_save_path)
        except Exception as e:
            print(f"Error updating label {label_path}: {e}")

    save_clustered_images(labeled_imgs, clustered_labels, clustered_images_save_dir)
