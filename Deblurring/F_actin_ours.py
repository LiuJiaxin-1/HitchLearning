'''
使用F_actin Val集中的SudoER-D4作为sudoSR
每个10个step测试一次
json文件在run-d4中
'''
import os
import utils
from train import deblur_train
option_path = './options/F-actin/training_ours.yml'
dataname = 'F_actin'
test_dir = "/data/liujiaxin/data/SUFDD/F_actin/9/Val/MBD5A90"
test_paths = utils.get_image_paths(test_dir)
for test_path in test_paths:
    img_name = os.path.splitext(os.path.split(test_path)[-1])[0]
    img_name = img_name.split('_', 1)[-1]
    deblur_train(test_dir=test_path, option_path=option_path, dataname=dataname, img_name=img_name)