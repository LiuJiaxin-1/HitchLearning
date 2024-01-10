import os
from Datasets.dataset import DataLoaderTrain, DataLoaderTrg, DataLoaderVal, DataLoaderTest

def get_training_data(data_dir, raw, gt, img_options):
    assert os.path.exists(data_dir)
    return DataLoaderTrain(data_dir, raw, gt, img_options)

def get_trg_data(data_dir, img_options):
    assert os.path.exists(data_dir)
    return DataLoaderTrg(data_dir, img_options)

def get_validation_data(data_dir, raw, gt, img_options):
    assert os.path.exists(data_dir)
    return DataLoaderVal(data_dir, raw, gt, img_options)

def get_test_data(data_dir, raw, gt, img_options):
    assert os.path.exists(data_dir)
    return DataLoaderTest(data_dir, raw, gt, img_options)
