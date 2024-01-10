'''
使用F_actin Val集中的SudoER-D4作为sudoSR
每个10个step测试一次
json文件在run-d4中
'''
import os
import utils
from main import dn_main
option_path = './options/training_ours_Golji.yml'
dataname = 'F_actin'
test_dir = "/data/liujiaxin/data/SUFDD/F_actin/9/Val/Raw"
test_paths = utils.get_image_paths(test_dir)
for test_path in test_paths:
    img_name = os.path.splitext(os.path.split(test_path)[-1])[0]
    img_name = img_name.split('_', 1)[-1]
    dn_main(test_dir=test_path, option_path=option_path, dataname=dataname, image_name=img_name)