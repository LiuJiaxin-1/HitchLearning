import os
import shutil
# set the source folder and destination folder paths
# source_folder = "/data/liujiaxin/Results/DB/F-actin_Golji"
# destination_folder = '/data/liujiaxin/Results/DB/F-actin_metrics'
# trgfilenames = ['F_actin_AllTrg.csv', 'F_actin_PartTrg.csv', 'F_actin_Trg.csv', 'F_actin_Src.csv', 
#                 'F_actin_SrcTrg.csv', 'F_actin_Finetune.csv', 'F_actin_ours.csv']
source_folder = "/data/liujiaxin/Results/DB/ER_Microtubule/"
destination_folder = '/data/liujiaxin/Results/DB/ER_metrics'
trgfilenames = ['ER_AllTrg.csv', 'ER_PartTrg.csv', 'ER_Trg.csv', 'ER_Src.csv', 
              'ER_SrcTrg.csv', 'ER_Finetune.csv', 'ER_ours.csv']
os.makedirs(destination_folder, exist_ok=True)
for foldername in os.listdir(source_folder):
    folderpath = os.path.join(source_folder, foldername)
    for root, dirs, filenames in os.walk(folderpath):
        if filenames and filenames[0].endswith(".csv") and any(filename in filenames for filename in trgfilenames):
            for filename in trgfilenames:
                if filename in filenames:
                    filepath = os.path.join(root, filename)
                    new_filename = foldername + '_' + filename
                    destination_filepath = os.path.join(destination_folder, new_filename)
                    shutil.copy(filepath, destination_filepath)
                    break