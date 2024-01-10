
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt, mode):
    model = opt['model']
    if model == 'Nei2Nei':
        if mode in ['AllTrg', 'PartTrg' , 'Trg', 'Src', 'Finetune']:
            model = 'plain1'
        elif mode == 'SrcTrg':
            model = 'plain2'
        elif mode == 'ours':
            model = 'plain3'
    elif model == 'SwinIR':
        if mode in ['AllTrg', 'PartTrg' , 'Trg', 'Src', 'Finetune']:
            model = 'plain4'
        elif mode == 'SrcTrg':
            model = 'plain5'
        elif mode == 'ours':
            model = 'plain6'

    if model == 'plain1':  # one input: L --Nei2Nei
        from models.model_plain1 import ModelPlain1 as M

    elif model == 'plain2':  # two inputs: L, SL --Nei2Nei
        from models.model_plain2 import ModelPlain2 as M

    elif model == 'plain3':  # two inputs: L, DA-SL --Nei2Nei
        from models.model_plain3 import ModelPlain3 as M
    
    elif model == 'plain4':  # two inputs: L, H --supervised
        from models.model_plain4 import ModelPlain4 as M
    
    elif model == 'plain5':  # four inputs: L, H, SL, SH --supervised
        from models.model_plain5 import ModelPlain5 as M
    
    elif model == 'plain6':  # four inputs: L, H, DA-SL, SH --supervised
        from models.model_plain6 import ModelPlain6 as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
