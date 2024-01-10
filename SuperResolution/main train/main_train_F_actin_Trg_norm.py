import os
import utils.utils_image as util
from main_trainSR_Norm import main
json_root = './options/F_actin/run-d4-f-norm/train_F-actin_x2_'
image_type = 'F_actin'
# modes = ['AllTrg', 'PartTrg', 'Trg', 'Src', 'Finetune', 'SrcTrg', 'ours']
modes = ['Trg']
test_root_L = "/data/liujiaxin/data/SUFDD/F_actin/9/Val/Noise" 
test_root_H = "/data/liujiaxin/data/SUFDD/F_actin/9/Val/SR/" 
test_paths_L = util.get_image_paths(test_root_L)
test_paths_H = util.get_image_paths(test_root_H)
for i in range(len(test_paths_L)):
    test_L, test_H = test_paths_L[i], test_paths_H[i]
    test_data = {'L': test_L, 'H': test_H}
    for mode in modes:
        if i and mode in ['AllTrg', 'PartTrg', 'Src']:
            continue
        image_name_ext = os.path.basename(test_L)
        image_name, ext = os.path.splitext(image_name_ext)
        image_name = image_name.split('_', 1)[-1]
        if mode in ['AllTrg', 'PartTrg', 'Src']:
            image_name = image_name.split('_')[0] + '_' + image_name.split('_')[1]
        json_path = json_root + mode + '.json'
        main(mode=mode, image_name=image_name, srcname='SR', finetune=False, image_type=image_type, test_dataroot=test_data, test_img_nums=0, train_img_nums=10, dataseed=1, json_path=json_path)