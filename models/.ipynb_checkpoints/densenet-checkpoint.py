
import torch
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
import thop


def densenet121():
    model = torchvision.models.densenet121().features
    model = IntermediateLayerGetter(model, {'transition2': 'feat1', 'transition3': 'feat2', 'denseblock4': 'feat3'})
    return model


if __name__ == '__main__':
    dumb = torch.randn(1, 3, 320, 320)
    model = torchvision.models.densenet121()
    model = IntermediateLayerGetter(model.features, {'transition2': 'feat1', 'transition3': 'feat2', 'denseblock4': 'feat3'})
    out = model(dumb)
    print(out['feat1'].shape)
    print(out['feat2'].shape)
    print(out['feat3'].shape)
    print(thop.clever_format(thop.profile(model, inputs=(dumb,))))