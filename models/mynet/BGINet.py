import torch
import torch.nn as nn

from .backbones import resnet
from ._blocks import Conv3x3, get_norm_layer
from ._utils import Identity, KaimingInitMixin
from .basicnet import MutualNet


class MGLNet(nn.Module):
    def __init__(self, dropout=0.1, zoom_factor=8, BatchNorm=nn.BatchNorm2d, num_clusters=32): # num_clusters代表图的顶点个数
        super(MGLNet, self).__init__()
        self.gamma = 1.0
        self.dim = 32 # vertex对应的特征维度
        # cascade mutual net
        self.mutualnet0 = MutualNet(BatchNorm, dim=self.dim, num_clusters=num_clusters, dropout=dropout)

    def forward(self, x1, x2):
        x1, x2 = self.mutualnet0(x1, x2)
        return x1, x2


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch): # in:32,out: 2
        super().__init__(
            Conv3x3(in_ch, in_ch, norm=True, act=True),
            Conv3x3(in_ch, out_ch)
        )


class Backbone(nn.Module, KaimingInitMixin):
    def __init__(
            self,
            in_ch, out_ch=32,
            arch='resnet18',
            pretrained=True,
            n_stages=5
    ):
        super().__init__()

        expand = 1
        strides = (2, 1, 2, 1, 1)
        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        else:
            raise ValueError
        self.n_stages = n_stages

        if self.n_stages == 5:
            itm_ch = 512 * expand
        elif self.n_stages == 4:
            itm_ch = 256 * expand   # itm_ch:256
        elif self.n_stages == 3:
            itm_ch = 128 * expand
        else:
            raise ValueError

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_out = Conv3x3(itm_ch, out_ch) # 特征维度降采样到32？

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_ch,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        if not pretrained:
            self._init_weight()

    def forward(self, x):  # 3 256 256
        y1 = self.resnet.conv1(x)  # 64 128 128 stride = 2
        y = self.resnet.bn1(y1)
        y = self.resnet.relu(y)
        y = self.resnet.maxpool(y)  # 64 64 64

        y = self.resnet.layer1(y)  # 64 64 64 stride = 1
        y = self.resnet.layer2(y)  # 128 32 32 stride = 2
        y = self.resnet.layer3(y)  # 256 32 32 stride = 1
        y = self.resnet.layer4(y)  # 256 32 32

        y = self.upsample(y) # 256 64 64

        return self.conv_out(y)

    def _trim_resnet(self):
        if self.n_stages > 5:
            raise ValueError

        if self.n_stages < 5:
            self.resnet.layer4 = Identity() # 最后一层直接用恒等映射

        if self.n_stages <= 3:
            self.resnet.layer3 = Identity()

        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()


class BGINet(nn.Module):
    def __init__(
            self, in_ch, out_ch,
            backbone='resnet18', n_stages=4,
            **backbone_kwargs
    ):
        super().__init__()

        # TODO: reduce hard-coded parameters
        dim = 32 # 32
        chn = dim # 32

        self.backbone = Backbone(in_ch, chn, arch=backbone, n_stages=n_stages, **backbone_kwargs)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear') # 用于对差分图像进行线性插值上采样，（64，,64） -》 （256,256）
        self.classifier = DoubleConv(chn, out_ch)
        self.MGLNet = MGLNet(num_clusters=32) # 图投影只是32？

    def forward(self, t1, t2):
        x1 = self.backbone(t1) # （32,64,64）
        x2 = self.backbone(t2) #
        x1, x2 = self.MGLNet(x1, x2)
        y = torch.abs(x1 - x2)
        y = self.upsample(y)

        pred = self.classifier(y) # 得到的是变化图，不是距离图吧？

        return pred
