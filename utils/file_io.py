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


# 读取VOC标签文件
def read_labelme_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    annotations = []
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        group_id = shape.get('group_id', None)  # Optional, if group_id exists

        annotations.append({
            'label': label,
            'points': points,
            'group_id': group_id
        })

    return annotations

def read_bbox_labels(file_path):
    """
    读取边界框标签文件，返回一个列表，其中每个元素为一个包含五个整数的元组。
    Args:
        file_path:

    Returns:

    """
    labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            # 检查行的长度是否符合预期
            if len(parts) != 5:
                continue  # 跳过无效行
            class_id = int(parts[0])
            x_min = int(parts[1])
            y_min = int(parts[2])
            x_max = int(parts[3])
            y_max = int(parts[4])
            labels.append((class_id, x_min, y_min, x_max, y_max))
    return labels


def read_lines_to_list(file_path):
    """
    读取文本文件的每一行，并将内容存储到列表中返回。

    Parameters:
    - file_path (str): 文件路径

    Returns:
    - list: 包含每行内容的列表
    """
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())  # 去除每行末尾的换行符并加入到列表中
    return lines


