import glob
import os




def get_data_path(file_paths, suffix=None, prefix=None, file_name_pattern=None):
    filename_list = []

    # 每个文件夹路径
    for file_path in file_paths:
        # 处理文件夹的子目录
        for root, dirs, files in os.walk(file_path):
            dir_paths = []
            root_dir = os.path.basename(root)

            # 处理指定的子文件夹
            if root_dir == 'labels':
                # 获取文件路径列表
                dir_paths = glob.glob(fr"{root}/{file_name_pattern}{suffix}")

                # 仅获取带后缀的文件名,并排除文件名为'classes.txt'的路径
                dir_paths = [path for path in dir_paths if not path.endswith('classes.txt')]

                # 将路径前加上前缀
                if prefix is not None:
                    dir_paths = [os.path.join(prefix, path) for path in dir_paths]

                filename_list.extend(dir_paths)

    return filename_list

data_dirs = [r'labels', ]
save_dir = "data_list"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


prefix = 'data'
# prefix = None
suffix = '.json'
file_name_pattern = '*'


label_list = get_data_path(file_paths=data_dirs, suffix=suffix, prefix=prefix, file_name_pattern=file_name_pattern)

description = '0717'

list_save_path = os.path.join(save_dir, f"label_list_{description}.txt")

with open(list_save_path, "w") as file:
    for string in label_list:
        file.write(string + "\n")