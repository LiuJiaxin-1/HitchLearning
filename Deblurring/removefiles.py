import os
import shutil

# 指定目录路径
dir_path = "/data/liujiaxin/Results/DB"

# 遍历目录下的所有文件和子目录
for root, dirs, files in os.walk(dir_path):
    for file in files:
        # 判断文件是否以png为后缀
        if file.endswith('.pth') and not file.endswith('best.pth') and not file.endswith('latest.pth'):
            # 拼接文件路径
            file_path = os.path.join(root, file)
            # 删除文件
            os.remove(file_path)