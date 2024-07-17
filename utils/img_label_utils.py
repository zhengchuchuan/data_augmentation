import json


def read_yolo_labels(file_path):
    labels = []
    with open(file_path, 'r',encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            # 检查行的长度是否符合预期
            if len(parts) != 5:
                continue  # 跳过无效行
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            labels.append((class_id, x_center, y_center, width, height))
    return labels

def read_labelme_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    data['imageData'] = None

    return data


def yolo_to_labelme_json(yolo_data, class_names, img_shape, version='4.5.6'):
    """
    将 YOLO 格式的标注数据转换为 LabelMe 格式的数据

    参数:
    yolo_data (str): YOLO 格式的标注数据
    class_names (list): 类别名称列表
    img_shape (tuple): 图像的形状 (height, width, channels)

    返回:
    dict: LabelMe 格式的数据
    """
    # 获取图像的宽度和高度
    img_height, img_width, _ = img_shape

    # 创建 LabelMe 格式的 JSON 数据结构
    labelme_data = {
        "version": version,  # LabelMe 版本，可以根据实际情况调整
        "flags": {},
        "shapes": [],
        "imagePath": None,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }

    # 处理每一行的 YOLO 数据
    for label in yolo_data:
        # 解析 YOLO 数据
        class_idx, x_center, y_center, width, height = label
        class_name = class_names[int(class_idx)]

        # 将 YOLO 格式转换为 LabelMe 格式
        x_min = (x_center - width / 2.0) * img_width
        y_min = (y_center - height / 2.0) * img_height
        x_max = (x_center + width / 2.0) * img_width
        y_max = (y_center + height / 2.0) * img_height

        shape = {
            "label": class_name,
            "points": [[x_min, y_min], [x_max, y_max]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        labelme_data["shapes"].append(shape)

    return labelme_data