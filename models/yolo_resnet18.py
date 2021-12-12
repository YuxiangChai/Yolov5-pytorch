
import argparse
import yaml

from models.experimental import *
from models.yolov5 import Detect

from utils.general import make_divisible, check_file, set_logging
from utils.autoanchor import check_anchor_order

from utils.torch_utils import initialize_weights
import math
from torch import nn
import torch

#from nb.torch.utils import device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from models.resnet18 import resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


class Yolo_resnet18(nn.Module):

    def __init__(self, cfg, ch=3):
        super(Yolo_resnet18, self).__init__()

        with open(cfg) as f:
            self.md = yaml.load(f, Loader=yaml.FullLoader)
        self.nc = self.md['nc']
        self.anchors = self.md['anchors']
        self.na = len(self.anchors[0]) // 2  # number of anchors
        
        self.backbone = resnet18()
#         self.backbone = IntermediateLayerGetter(self.backbone, {'stage2': 0, 'stage3': 1, 'stage4': 2})

        # FPN
        in_channels = [128, 256, 512]
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels, out_channels=256)
        self.detect = Detect(self.nc, self.anchors, [256, 256, 256])

        # forward to get Detect lay params dynamically
        s = 512  # 2x min stride
        self.detect.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])    # forward
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        check_anchor_order(self.detect)
        self.stride = self.detect.stride
        self._initialize_biases()
        initialize_weights(self)

        # for compatible, in check_anchors in train.py
        self.model = [self.detect]

    def _initialize_biases(self, cf=None):
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99) ) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def forward(self, x, augment=False):
        # we not using augment at all
        feas = self.backbone(x)
        
        a = self.fpn(feas)
        # for k, v in a.items():
            # print(k, v.shape)
        x_s, x_m, x_l = a['feat1'], a['feat2'], a['feat3']
        x = self.detect([x_s, x_m, x_l])
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    # Create model
    model = YoloV6(opt.cfg).to(device)
    model.train()
    # # print(model)

    import time

    a = torch.randn([1, 3, 1280, 736, ]).to(device)
    all_t = 0
    for i in range(10):
        tic = time.time()
        aa = model(a)
        all_t += (time.time() - tic)
    print('average cost: ', all_t/100)

    m = resnet18()
    a = torch.randn([1, 3, 512, 512])
    c = m(a)
    print(c.shape)
