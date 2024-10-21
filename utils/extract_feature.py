import torch


def extract_feature(model, loader):
    features = torch.FloatTensor()

    for (inputs, labels) in loader:

        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f1 = outputs[0].data.cpu()

        # flip
        inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f2 = outputs[0].data.cpu()
        ff = f1 + f2

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features


def extract_single_feature(model, loader):
    features = torch.FloatTensor()
    for (inputs, sizes) in loader:

        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f1 = outputs[0].data.cpu()

        # flip
        inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
        input_img = inputs.to('cuda')
        outputs = model(input_img)
        f2 = outputs[0].data.cpu()
        ff = f1 + f2

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
        size = sizes
        
        p1 = outputs[-3].data.cpu()
        p2 = outputs[-2].data.cpu()
        p3 = outputs[-1].data.cpu()


    return features, size, p1, p2, p3