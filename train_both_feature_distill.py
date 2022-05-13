import torch
from torch import Tensor
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from typing import Type, Any, Callable, Union, List, Optional
import time
import numpy as np
import shutil
import PIL
import os

import math

from itertools import product

import bdpy
from bdpy.bdata import concat_dataset
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from bdpy.util import makedir_ifnot, get_refdata
from bdpy.dataform import append_dataframe
from bdpy.distcomp import DistComp

import pandas as pd
import cv2

import torch.nn.functional as F

TESTSET_source = ['n01518878_8432', 'n01639765_52902', 'n01645776_9743', 'n01664990_7133', 'n01704323_9812', 'n01726692_9090', 'n01768244_9890', 'n01770393_29944', 'n01772222_16161', 'n01784675_9652', 'n01787835_9197', 'n01833805_9089', 'n01855672_15900', 'n01877134_8213', 'n01944390_21065', 'n01963571_7996', 'n01970164_30734', 'n02005790_9371', 'n02054036_9789', 'n02055803_9893', 'n02068974_7285', 'n02084071_34874', 'n02090827_9359', 'n02131653_7902', 'n02226429_9440', 'n02233338_9484', 'n02236241_8966', 'n02317335_6776', 'n02346627_7809', 'n02374451_19966', 'n02391049_8178', 'n02432983_9121', 'n02439033_9911', 'n02445715_834', 'n02472293_5718', 'n02480855_9990', 'n02481823_8477', 'n02503517_9095', 'n02508213_7987', 'n02692877_7319', 'n02766534_64673', 'n02769748_52896', 'n02799175_9738', 'n02800213_8939', 'n02802215_9364', 'n02808440_5873', 'n02814860_39856', 'n02841315_35046', 'n02843158_9764', 'n02882647_7150', 'n02885462_9275', 'n02943871_9326', 'n02974003_9408', 'n02998003_7158', 'n03038480_7843', 'n03063599_5015', 'n03079230_8270', 'n03085013_24642', 'n03085219_8998', 'n03187595_9726', 'n03209910_8951', 'n03255030_8996', 'n03261776_40091', 'n03335030_54164', 'n03345487_8958', 'n03359137_44522', 'n03394916_47877', 'n03397947_9823', 'n03400231_8108', 'n03425413_22005', 'n03436182_9434', 'n03445777_9793', 'n03467796_862', 'n03472535_8251', 'n03483823_8564', 'n03494278_42246', 'n03496296_9005', 'n03512147_7137', 'n03541923_9436', 'n03543603_9964', 'n03544143_8694', 'n03602883_8804', 'n03607659_8657', 'n03609235_7271', 'n03612010_9076', 'n03623556_9879', 'n03642806_6122', 'n03646296_9175', 'n03649909_42614', 'n03665924_8249', 'n03721384_6760', 'n03743279_8976', 'n03746005_9272', 'n03760671_9669', 'n03790512_41520', 'n03792782_8413', 'n03793489_8932', 'n03815615_8756', 'n03837869_9127', 'n03886762_9466', 'n03918737_8191', 'n03924679_9951', 'n03950228_39297', 'n03982430_9407', 'n04009552_8947', 'n04044716_8619', 'n04070727_9914', 'n04086273_5433', 'n04090263_7624', 'n04113406_5900', 'n04123740_8706', 'n04146614_9889', 'n04154565_35063', 'n04168199_9862', 'n04180888_8500', 'n04197391_8396', 'n04225987_7933', 'n04233124_24091', 'n04254680_43', 'n04255586_878', 'n04272054_61432', 'n04273569_6437', 'n04284002_21913', 'n04313503_25193', 'n04320973_9400', 'n04373894_59446', 'n04376876_7775', 'n04398044_35327', 'n04401680_36641', 'n04409515_8358', 'n04409806_9916', 'n04412416_8901', 'n04419073_8965', 'n04442312_32591', 'n04442441_7248', 'n04477387_9242', 'n04482393_6359', 'n04497801_8739', 'n04555897_7654', 'n04587559_7411', 'n04591713_7167', 'n04612026_7911', 'n07734017_8706', 'n07734744_8647', 'n07756951_9462', 'n07758680_22370', 'n11978233_42157', 'n12582231_44661', 'n12596148_9625', 'n13111881_9170']
TESTSET = []
for item in TESTSET_source:
    test_item_front = int(item[1:].split('_')[0]) + 0.000001* int(item[1:].split('_')[1])
    TESTSET.append([test_item_front])
print(TESTSET)

class config():
  def __init__(self): 
    self.analysis_name = 'GenericObjectDecoding'

    # Data settings
    self.subjects = {#'Subject1' : ['data/Subject1.h5']}
                'Subject2' : ['data/Subject2.h5'],
                'Subject3' : ['data/Subject3.h5'],
                'Subject4' : ['data/Subject4.h5'],
                'Subject5' : ['data/Subject5.h5']}

    self.rois = {#'VC' : 'ROI_VC = 1',
            #'LVC' : 'ROI_LVC = 1',
            #'HVC' : 'ROI_HVC = 1',
            'V1' : 'ROI_V1 = 1',
            'V2' : 'ROI_V2 = 1',
            'V3' : 'ROI_V3 = 1',
            'V4' : 'ROI_V4 = 1',
            'LOC' : 'ROI_LOC = 1',
            'FFA' : 'ROI_FFA = 1',
            'PPA' : 'ROI_PPA = 1'}

    self.num_voxel = {'VC' : 1000,
                'LVC' : 1000,
                'HVC' : 1000,
                'V1' : 500,
                'V2' : 500,
                'V3' : 500,
                'V4' : 500,
                'LOC' : 500,
                'FFA' : 500,
                'PPA' : 500}

    self.image_feature_file = 'data/ImageFeatures.h5'
    self.features = ['cnn5']#['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8', 'hmax1', 'hmax2', 'hmax3', 'gist', 'sift']

    # Results settings
    self.results_dir = os.path.join('results', self.analysis_name)
    self.results_file = os.path.join('results', self.analysis_name + '.pkl')

    # Figure settings
    selfroi_labels = ['V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA', 'LVC', 'HVC', 'VC']


SAVEPATH = '/root/wheemi/GenericObjectDecoding/code/python/models/'
WEIGHTDECAY = 5e-4
MOMENTUM = 0.8
BATCHSIZE = 64#128
LR = 0.01
EPOCHS = 15
PRINTFREQ = 50
BETA = 1.0
CUTMIXPROB = 0.5
RESUME=False
config = config()

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


#ResNet 1D
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 150,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return [x, torch.flatten(x1,1), torch.flatten(x2,1), torch.swapaxes(x3,1,2), torch.flatten(x4,1)]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print("Pretrained!")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        for name, value in state_dict.items():
            print(name)
            print(value.shape)
            if "conv1" in name:
                state_dict[name] = torch.unsqueeze(torch.max(torch.max(value, -1)[0],1)[0],1)
            elif "conv" in name:
                state_dict[name] = torch.max(value, -1)[0]
            elif "downsample" in name:
                state_dict[name] = torch.squeeze(value, -1)

            model.state_dict()[name].copy_(state_dict[name])
        
        for para in model.parameters():
            para.requires_grad = True
        model.fc = nn.Linear(512, 150)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


#ResNet 2D
def conv2d3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv2d1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock2d(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv2d3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv2d3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck2d(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv2d1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv2d3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv2d1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2d(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock2d, Bottleneck2d]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = BATCHSIZE
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck2d):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock2d):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock2d, Bottleneck2d]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv2d1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return [x, torch.flatten(x1,1), torch.flatten(x2,1), torch.swapaxes(torch.flatten(x3,2),1,2), torch.flatten(x4,1)]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resnet2d(
    arch: str,
    block: Type[Union[BasicBlock2d, Bottleneck2d]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet2d:
    model = ResNet2d(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)   
        model.fc = nn.Linear(512 * block.expansion, 150)
    return model


def resnet2d18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2d:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet2d('resnet18', BasicBlock2d, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet2d34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2d:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet2d('resnet34', BasicBlock2d, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet2d50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2d:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet2d('resnet50', Bottleneck2d, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet2d101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2d:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet2d('resnet101', Bottleneck2d, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet2d152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2d:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet2d('resnet152', Bottleneck2d, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

class CustomfmriDataset_train(Dataset):
    def __init__(self, subjects, rois, features, length=100, transform=None):
        self.transform = transform

        x_train, _, y_train, _ = get_data(subjects, rois, features, length)
        x_train = torch.from_numpy(np.expand_dims(x_train, axis=1)).float()
        y_train = torch.from_numpy(np.expand_dims(y_train, axis=1)).float()

        self.signals = x_train
        self.labels = y_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        if self.transform:
            signal = self.transform(signal)
        return signal, label

class CustomfmriDataset_test(Dataset):
    def __init__(self, subjects, rois, features, length=100, transform=None):
        self.transform = transform

        _, x_test, _, y_test = get_data(subjects, rois, features, length)
        x_test = torch.from_numpy(np.expand_dims(x_test, axis=1)).float()
        y_test = torch.from_numpy(np.expand_dims(y_test, axis=1)).float()

        self.signals = x_test
        self.labels = y_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        if self.transform:
            signal = self.transform(signal)
        return signal, label

class CustomImageNetDataset(Dataset):
    def __init__(self, csv_file, root_dir, fmri_data, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.fmri_data = fmri_data

        target_class_frame = pd.read_csv(csv_file, delimiter = '\t+|:')
        data = {'names':[], 'classes':[]}
        for i in range(1200):
          if train:
            if int(target_class_frame.loc[i, 'number']) % 8 != 0:
              data['names'].append(target_class_frame.loc[i, 'name'])
              data['classes'].append(target_class_frame.loc[i, 'class'])
          else:
            if int(target_class_frame.loc[i, 'number']) % 8 == 0:
              data['names'].append(target_class_frame.loc[i, 'name'])
              data['classes'].append(target_class_frame.loc[i, 'class'])
        
        print(data['names'])
        self.target_class_frame = data

    def __len__(self):
        return len(self.target_class_frame['names'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, 'training',
                        self.target_class_frame['names'][idx]+'.JPEG')

        image = PIL.Image.open(img_name).convert("RGB")
        target_class = self.target_class_frame['classes'][idx]
        target_class = np.array([target_class])
        target_class = target_class.astype('int') - 1

        if self.transform:
            image = self.transform(image)
        
        fmri = self.fmri_data[idx]

        return idx, image, fmri, target_class

class AverageMeter(object):
    r"""Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
        
class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    r"""Computes the accuracy over the $k$ top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))

        # faster topk (ref: https://github.com/pytorch/pytorch/issues/22812)
        _, idx = output.sort(descending=True)
        pred = idx[:,:maxk]
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_data(subjects, rois, features, length=100):#length=1000 is the maximum
    # Data settings
    num_voxel = config.num_voxel

    image_feature = config.image_feature_file

    n_iter = 200

    results_dir = config.results_dir

    # Misc settings
    analysis_basename = os.path.basename(__file__)

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_all = {}
    for sbj in subjects:
        if len(subjects[sbj]) == 1:
            data_all[sbj] = bdpy.BData(subjects[sbj][0])
        else:
            # Concatenate data
            suc_cols = ['Run', 'Block']
            data_all[sbj] = concat_dataset([bdpy.BData(f) for f in subjects[sbj]],
                                           successive=suc_cols)

    data_feature = bdpy.BData(image_feature)
    data_feature.show_metadata()

    # Add any additional processing to data here

    # Initialize directories -------------------------------------------
    makedir_ifnot(results_dir)
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for s, r, f in product(subjects, rois, features):
        sbj = s
        roi = r
        feat = f

    print('--------------------')
    print('Subject:    %s' % sbj)
    print('ROI:        %s' % roi)
    print('Num voxels: %d' % num_voxel[roi])
    print('Feature:    %s' % feat)

    # Distributed computation
    analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
    results_file = os.path.join(results_dir, analysis_id + '.pkl')

    # Prepare data
    print('Preparing data')
    dat = data_all[sbj]

    x = dat.select(rois[roi])           # Brain data
    datatype = dat.select('DataType')   # Data type
    labels = dat.select('stimulus_id')  # Image labels in brain data
    print("labels: ", labels)

    y = data_feature.select(feat)             # Image features
    y_label = data_feature.select('ImageID')  # Image labels
    print("y: ", y)
    print("y_label: ", y_label)

    # For quick demo, reduce the number of units from 1000 to 100
    y = y[:, :length]

    y_sorted = get_refdata(y, y_label, labels)  # Image features corresponding to brain data

    # Get training and test dataset
    i_train = (datatype == 1).flatten()    # Index for training
    i_test_pt = (datatype == 2).flatten()  # Index for perception test
    i_test_im = (datatype == 3).flatten()  # Index for imagery test
    i_test = i_test_pt + i_test_im

    x_train = x[i_train, :]
    x_test = x[i_test, :]

    y_train = y_sorted[i_train, :]
    y_test = y_sorted[i_test, :]

    return x_train, x_test, y_train, y_test

def save_ckpt(state, is_best):
  f_path = SAVEPATH+'model_weight_latest.pt'
  torch.save(state, f_path)
  if is_best:
    best_fpath = SAVEPATH+'model_weight_best.pt'
    shutil.copyfile(f_path, best_fpath)

def load_ckp(model, sbj):
  checkpoint_fpath = SAVEPATH+'model1d_'+sbj+'_weight_latest.pt'#SAVEPATH+'model1d_weight_latest.pt'
  checkpoint = torch.load(checkpoint_fpath)
  model.load_state_dict(checkpoint['state_dict'])
  return model

class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss
def softmax(x):
    
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

def main(brain_teacher_target=None):

    # Data settings
    subjects = config.subjects
    rois = config.rois
    num_voxel = config.num_voxel

    image_feature = config.image_feature_file
    features = config.features

    n_iter = 200

    results_dir = config.results_dir

    # Misc settings
    analysis_basename = os.path.basename(__file__)

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_all = {}
    for sbj in subjects:
        if len(subjects[sbj]) == 1:
            data_all[sbj] = bdpy.BData(subjects[sbj][0])
        else:
            # Concatenate data
            suc_cols = ['Run', 'Block']
            data_all[sbj] = concat_dataset([bdpy.BData(f) for f in subjects[sbj]],
                                           successive=suc_cols)

    print("DATA ALL:", data_all)
    data_feature = bdpy.BData(image_feature)
    data_feature.show_metadata()

    # Add any additional processing to data here     

    # Initialize directories -------------------------------------------
    makedir_ifnot(results_dir)
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    
    model = resnet2d18(pretrained=True)
    #adap_layers = [nn.Linear(128 * 93, 128 * 28*28, bias=False).cuda(),
    #        nn.Linear(256 * 47, 256 * 14*14, bias=False).cuda(),
    #        nn.Linear(512 * 24, 512 * 7*7, bias=False).cuda()]
    model = model.cuda()

    #model_fmri = resnet2d34(pretrained=True).cuda()

    teacher_model = None #resnet34(pretrained=True)

    ##### optimizer / learning rate scheduler / criterion #####
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHTDECAY,
                                nesterov=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 75],
    #                                                 gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    feature_kd_criterion = [torch.nn.MSELoss().cuda(),
            torch.nn.MSELoss().cuda(),
            torch.nn.MSELoss().cuda()]
    ###########################################################

    if brain_teacher_target is not None:
        print(brain_teacher_target)
        brain_teacher = np.load(brain_teacher_target)
        print(brain_teacher.dtype)
    else:
        brain_teacher = None
    
    #print(brain_teacher.shape)#(1050, 150)
    #print(brain_teacher)

    noise_teacher = np.random.rand(1050, 150)
    noise_teacher = softmax(noise_teacher/0.01).astype(np.float32)
    print(np.max(noise_teacher,axis=1))
    print(noise_teacher.dtype)

    # Check number of parameters your model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")

    start_epoch = 0
    
    print('==> Load data..')


    last_top1_acc = 0
    acc1_valid = 0
    bests = {}
    best_acc1 = 0
    best_train_acc1 = 0
    is_train_best = False
    is_best = False

    for sbj, roi, feat in product(subjects, rois, features):

        model_fmri = resnet18()
        model_fmri = model_fmri.cuda()
        if model_fmri is not None:
            model_fmri = load_ckp(model_fmri, sbj)
            for param in model_fmri.parameters():
                param.requires_grad = False
                
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % num_voxel[roi])
        print('Feature:    %s' % feat)

        # Distributed computation
        analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
        results_file = os.path.join(results_dir, analysis_id + '.pkl')

        '''
        if os.path.exists(results_file):
            print('%s is already done. Skipped.' % analysis_id)
            continue

        dist = DistComp(lockdir='tmp', comp_id=analysis_id)
        if dist.islocked():
            print('%s is already running. Skipped.' % analysis_id)
            continue
        dist.lock()
        '''

        # Prepare data
        print('Preparing data')
        dat = data_all[sbj]

        x = dat.select(rois[roi])           # Brain data
        datatype = dat.select('DataType')   # Data type
        labels = dat.select('stimulus_id')  # Image labels in brain data
        print("Labels:", labels) 
        print("Labels:", labels.shape) #(3450,1)

        y = data_feature.select(feat)             # Image features
        y_label = data_feature.select('ImageID')  # Image labels
        print("ImID:", y_label)
        print("ImID:", y_label.shape) #(16622,1)

        y = y[:, :1000]

        print(x)
        print(x.shape) #(3450,860)
        feature_length = int(x.shape[-1])
        print(y)
        print(y.shape) #(16622,1000)
        print(y_label)
        print(y_label.shape) #(16622,1)

        train_indexes = []
        test_indexes = []
        for i, test_label in enumerate(y_label[:1200]):
            if test_label in TESTSET:
                test_indexes.append(i)
            else:
                train_indexes.append(i)

        print("train_label: ", train_indexes)
        print("train_label: ", len(train_indexes))#1050

        print("test_label: ", test_indexes)
        print("test_label: ", len(test_indexes))#150

        y_label_not_none = np.isnan(y_label)        
        print(y_label[y_label_not_none==False])
        print(y_label[y_label_not_none==False].shape) #(1250,1)

        x = x[:1200, :]

        x_train = np.expand_dims(x[train_indexes, :], axis=1)
        x_test = np.expand_dims(x[test_indexes, :], axis=1)

        y_train = np.array([i//7 for i in range(1050)])
        y_test = np.array([i for i in range(150)])

        y_label_train = y_label[train_indexes, :]
        y_label_test = y_label[test_indexes, :]

        print("x_train: ", x_train.shape)#1050,740
        print("x_test: ", x_test.shape)#150,740
        print("y_train: ", y_train.shape)#1050,
        print("y_test: ", y_test.shape)#150,
        print("y_label_train: ", y_label_train.shape)#1050,1
        print("y_label_test: ", y_label_test.shape)#150,1

        train_transform = transforms.Compose([
            transforms.RandomCrop(448),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

        train_imagenet_dataset = CustomImageNetDataset(csv_file='/root/wheemi/GenericObjectDecoding/data/stimulus_ImageNetTraining.tsv', root_dir='/root/wheemi/GenericObjectDecoding/code/python/images/', fmri_data=x_train, train=True, transform=train_transform)
        test_imagenet_dataset = CustomImageNetDataset(csv_file='/root/wheemi/GenericObjectDecoding/data/stimulus_ImageNetTraining.tsv', root_dir='/root/wheemi/GenericObjectDecoding/code/python/images/', fmri_data=x_test, train=False, transform=train_transform)

        #dataset_lengths = [int(len(imagenet_dataset)*0.9), int(len(imagenet_dataset)*0.1)]
        #subsetA, subsetB = random_split(imagenet_dataset, dataset_lengths)

        train_loader = DataLoader(train_imagenet_dataset,
                                batch_size=BATCHSIZE, shuffle=True,
                                num_workers=4, pin_memory=True)

        val_loader = DataLoader(test_imagenet_dataset,
                                batch_size=1, shuffle=False,
                                num_workers=1, pin_memory=True)
      
        feature_length_fit = math.ceil(feature_length/16.)
        print(feature_length_fit)
        #adap_layers = nn.Linear(256 * feature_length_fit, 256 * 14*14, bias=False).cuda()
        adap_layers = nn.Conv1d(feature_length_fit, 14*14, kernel_size=1, bias=False).cuda()

        for epoch in range(start_epoch, EPOCHS):
            print("\n----- epoch: {}, lr: {} -----".format(
                epoch, optimizer.param_groups[0]["lr"]))

            # train for one epoch
            start_time = time.time()
            last_top1_acc = train(train_loader, epoch, model, model_fmri, optimizer, criterion, feature_kd_criterion, adap_layers, teacher_model=None, brain_teacher=None)
            elapsed_time = time.time() - start_time
            print('==> {:.2f} seconds to train this epoch\n'.format(
                elapsed_time))

            # validate for one epoch
            start_time = time.time()
            acc1_valid = validate(val_loader, model, model_fmri, criterion, feature_kd_criterion)
            elapsed_time = time.time() - start_time
            print('==> {:.2f} seconds to validate this epoch\n'.format(
                elapsed_time))

            # learning rate scheduling
            scheduler.step()

            is_best = acc1_valid > best_acc1
            best_acc1 = max(acc1_valid, best_acc1)

            is_train_best = last_top1_acc > best_train_acc1
            best_train_acc1 = max(last_top1_acc, best_train_acc1)

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            save_ckpt(checkpoint, is_train_best)

        print(f"Last Top-1 Accuracy: {acc1_valid}")
        print(f"Last Top-1 Train Accuracy: {last_top1_acc}")
        print(f"Best Top-1 Valid Accuracy: {best_acc1}")
        print(f"Number of parameters: {pytorch_total_params}")
        print(brain_teacher_target)
        bests[(sbj, roi)] = best_acc1

        f = open("results_log.txt", 'a')
        f.write(str(best_acc1.item())+"%"+" ")
        f.close()

        best_acc1 = 0
        print(bests)

    return bests


def train(train_loader, epoch, model, model_fmri, optimizer, criterion, feature_kd_criterion, adap_layers, teacher_model=None, brain_teacher=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if teacher_model is None and brain_teacher is None:
        kd_losses = AverageMeter('KD Loss', ':.4e')
        progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, kd_losses,
                              top1, top5, prefix="Epoch: [{}]".format(epoch))
        #progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, 
        #                      top1, top5, prefix="Epoch: [{}]".format(epoch))
    else:
        kd_losses = AverageMeter('KD Loss', ':.4e')
        progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, kd_losses,
                              top1, top5, prefix="Epoch: [{}]".format(epoch))
      
    # switch to train mode
    model.train()
    #for l in adap_layers:
    #    l.train()
    adap_layers.train()
    model_fmri.eval()

    end = time.time()
    for i, (idx, input, fmri, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        fmri = fmri.cuda().float()
        target = target.squeeze(1).cuda()

        r = np.random.rand(1)

        # compute output
        output1 = model(input)
        output2 = model_fmri(fmri)
        output = output1[0]
        loss = criterion(output, target)
        kd_feat_loss = feature_kd_criterion[1](output1[3], adap_layers(output2[3]))
        kd_losses.update(kd_feat_loss, input.size(0))
        loss += kd_feat_loss

        # measure accuracy and record loss, accuracy 
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))

        if teacher_model is not None:
            teacher_output = teacher_model(input)
            kd_criterion = SoftTarget(T=0.1)
            kd_loss = kd_criterion(output, teacher_output)
            kd_losses.update(kd_loss, input.size(0))
            loss += kd_loss

        if brain_teacher is not None:
            soft_target = brain_teacher[idx]
            teacher_output = torch.from_numpy(soft_target).cuda()
            kd_criterion = SoftTarget(T=0.1)
            kd_loss = kd_criterion(output, teacher_output)
            kd_losses.update(kd_loss, input.size(0))
            loss += kd_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if i % PRINTFREQ == 0:
        #    progress.print(i)

    #print('=> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #      .format(top1=top1, top5=top5))
    return top1.avg


def validate(val_loader, model, model_fmri, criterion, feature_kd_criterion):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), batch_time, losses, top1, top5, prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    model_fmri.eval()
    total_loss = 0.0

    with torch.no_grad():
        end = time.time()
        for i, (idx, input, fmri, target) in enumerate(val_loader):

            input = input.cuda()
            fmri = fmri.cuda().float()
            target = target.squeeze(1).cuda()

            # compute output
            output1 = model(input)
            output2 = model_fmri(fmri)
            output = output1[0]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            total_loss += loss.item()

            #if i % PRINTFREQ == 0:
            #    progress.print(i)

            end = time.time()

        #print(
        #    "====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
        #        top1=top1, top5=top5
        #    )
        #)
        total_loss = total_loss / len(val_loader)

    return top1.avg

if __name__ == "__main__":
    result_list = []
    for i in range(5):
        bests = main(brain_teacher_target=None)#'soft_label.npy')
        result_list.append(bests)
        f = open("results_log.txt", 'a')
        f.write("\n")
        f.close()
    print(result_list)

    #best_acc1 = main(brain_teacher_target=None)
    '''    
    brain_teacher_target_dir=SAVEPATH+'soft_labels/'
    brain_teacher_targets=os.listdir(brain_teacher_target_dir)
    print(brain_teacher_targets)
    result_dict = {}
    for file in brain_teacher_targets:
        brain_teacher_target=brain_teacher_target_dir+file
        best_acc1 = main(brain_teacher_target=brain_teacher_target)


    for file in brain_teacher_targets:
        brain_teacher_target=brain_teacher_target_dir+file
        best_acc1 = main(brain_teacher_target=brain_teacher_target)
        result_dict[brain_teacher_target] = best_acc1
    for key, value in result_dict.items():
        print(key, value)
    '''