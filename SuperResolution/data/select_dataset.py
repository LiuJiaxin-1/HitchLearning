

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt, select_img_nums=0, dataseed=0):
    dataset_type = dataset_opt['dataset_type'].lower()
    # -----------------------------------------
    # unsupervised denoising/super-esolution/deblur
    # -----------------------------------------
    if dataset_type in ['l']:
        from data.dataset_l import DatasetL as D
    
    elif dataset_type in ['ls']:
        from data.dataset_ls import DatasetLS as D

    # -----------------------------------------
    # supervised denoising
    # -----------------------------------------
    elif dataset_type in ['dn']:
        from data.dataset_dn import DatasetDN as D

    # -----------------------------------------
    # supervised super-resolution
    # -----------------------------------------
    elif dataset_type in ['sr']:
        from data.dataset_sr import DatasetSR as D

    # -----------------------------------------
    # supervised deblur
    # -----------------------------------------
    elif dataset_type in ['bd']:
        from data.dataset_db import DatasetDB as D

    # -----------------------------------------
    # supervised denoising
    # -----------------------------------------
    elif dataset_type in ['dns']:
        from data.dataset_dns import DatasetDNS as D

    # -----------------------------------------
    # supervised super-resolution
    # -----------------------------------------
    elif dataset_type in ['srs']:
        from data.dataset_srs import DatasetSRS as D

    # -----------------------------------------
    # supervised deblur
    # -----------------------------------------
    elif dataset_type in ['bds']:
        from data.dataset_dbs import DatasetDBS as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt, select_img_nums, dataseed)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset