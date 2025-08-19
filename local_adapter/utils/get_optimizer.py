from torch.optim import Adam, SGD, AdamW
from opt import opt
import torch


def get_optimizer(extractor, image_adapter):
    # extractor와 image_adapter의 파라미터를 모두 포함
    params = list(extractor.parameters()) + list(image_adapter.parameters())
    
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif opt.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=5e-4)

    return optimizer