import os
from Datasets.dataset import DataLoaderTrain, DataLoaderSrcTrg, DataLoaderVal, DataLoaderTest

def get_training_data(data_dir, raw, gt, img_options, nums=0):
    assert os.path.exists(data_dir)
    return DataLoaderTrain(data_dir, raw, gt, img_options, nums)

def get_srctrg_data(src_dir, trg_dir, sraw, sgt, traw, tgt, img_options):
    assert os.path.exists(src_dir)
    assert os.path.exists(trg_dir)
    return DataLoaderSrcTrg(src_dir, trg_dir, sraw, sgt, traw, tgt, img_options)

def get_validation_data(data_dir, raw, gt, img_options):
    assert os.path.exists(data_dir)
    return DataLoaderVal(data_dir, raw, gt, img_options)

def get_test_data(data_dir, raw, gt, img_options):
    assert os.path.exists(data_dir)
    return DataLoaderTest(data_dir, raw, gt, img_options)
