'''Generic Object Decoding: Feature prediction

Analysis summary
----------------

- Learning method:   Sparse linear regression
- Preprocessing:     Normalization and voxel selection
- Data:              GenericDecoding_demo
- Results format:    Pandas dataframe
'''


from __future__ import print_function

import os
import sys
import pickle
from itertools import product
from time import time

import numpy as np
import pandas as pd
from scipy import stats

import bdpy
from bdpy.bdata import concat_dataset
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from bdpy.util import makedir_ifnot, get_refdata
from bdpy.dataform import append_dataframe
from bdpy.distcomp import DistComp

import god_config as config

from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch.utils.data import TensorDataset, DataLoader
from torch.hub import load_state_dict_from_url

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


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

TESTSET_source = ['n01518878_8432', 'n01639765_52902', 'n01645776_9743', 'n01664990_7133', 'n01704323_9812', 'n01726692_9090', 'n01768244_9890', 'n01770393_29944', 'n01772222_16161', 'n01784675_9652', 'n01787835_9197', 'n01833805_9089', 'n01855672_15900', 'n01877134_8213', 'n01944390_21065', 'n01963571_7996', 'n01970164_30734', 'n02005790_9371', 'n02054036_9789', 'n02055803_9893', 'n02068974_7285', 'n02084071_34874', 'n02090827_9359', 'n02131653_7902', 'n02226429_9440', 'n02233338_9484', 'n02236241_8966', 'n02317335_6776', 'n02346627_7809', 'n02374451_19966', 'n02391049_8178', 'n02432983_9121', 'n02439033_9911', 'n02445715_834', 'n02472293_5718', 'n02480855_9990', 'n02481823_8477', 'n02503517_9095', 'n02508213_7987', 'n02692877_7319', 'n02766534_64673', 'n02769748_52896', 'n02799175_9738', 'n02800213_8939', 'n02802215_9364', 'n02808440_5873', 'n02814860_39856', 'n02841315_35046', 'n02843158_9764', 'n02882647_7150', 'n02885462_9275', 'n02943871_9326', 'n02974003_9408', 'n02998003_7158', 'n03038480_7843', 'n03063599_5015', 'n03079230_8270', 'n03085013_24642', 'n03085219_8998', 'n03187595_9726', 'n03209910_8951', 'n03255030_8996', 'n03261776_40091', 'n03335030_54164', 'n03345487_8958', 'n03359137_44522', 'n03394916_47877', 'n03397947_9823', 'n03400231_8108', 'n03425413_22005', 'n03436182_9434', 'n03445777_9793', 'n03467796_862', 'n03472535_8251', 'n03483823_8564', 'n03494278_42246', 'n03496296_9005', 'n03512147_7137', 'n03541923_9436', 'n03543603_9964', 'n03544143_8694', 'n03602883_8804', 'n03607659_8657', 'n03609235_7271', 'n03612010_9076', 'n03623556_9879', 'n03642806_6122', 'n03646296_9175', 'n03649909_42614', 'n03665924_8249', 'n03721384_6760', 'n03743279_8976', 'n03746005_9272', 'n03760671_9669', 'n03790512_41520', 'n03792782_8413', 'n03793489_8932', 'n03815615_8756', 'n03837869_9127', 'n03886762_9466', 'n03918737_8191', 'n03924679_9951', 'n03950228_39297', 'n03982430_9407', 'n04009552_8947', 'n04044716_8619', 'n04070727_9914', 'n04086273_5433', 'n04090263_7624', 'n04113406_5900', 'n04123740_8706', 'n04146614_9889', 'n04154565_35063', 'n04168199_9862', 'n04180888_8500', 'n04197391_8396', 'n04225987_7933', 'n04233124_24091', 'n04254680_43', 'n04255586_878', 'n04272054_61432', 'n04273569_6437', 'n04284002_21913', 'n04313503_25193', 'n04320973_9400', 'n04373894_59446', 'n04376876_7775', 'n04398044_35327', 'n04401680_36641', 'n04409515_8358', 'n04409806_9916', 'n04412416_8901', 'n04419073_8965', 'n04442312_32591', 'n04442441_7248', 'n04477387_9242', 'n04482393_6359', 'n04497801_8739', 'n04555897_7654', 'n04587559_7411', 'n04591713_7167', 'n04612026_7911', 'n07734017_8706', 'n07734744_8647', 'n07756951_9462', 'n07758680_22370', 'n11978233_42157', 'n12582231_44661', 'n12596148_9625', 'n13111881_9170']
TESTSET = []
for item in TESTSET_source:
    test_item_front = int(item[1:].split('_')[0]) + 0.000001* int(item[1:].split('_')[1])
    TESTSET.append([test_item_front])
print(TESTSET)

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

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


# Main #################################################################

def save_ckpt(state):
  SAVEPATH = '/root/wheemi/GenericObjectDecoding/code/python/models/'
  f_path = SAVEPATH+'model1d_weight_latest.pt'
  torch.save(state, f_path)

def main():
    # Settings ---------------------------------------------------------

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

    for sbj, roi, feat in product(subjects, rois, features):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % num_voxel[roi])
        print('Feature:    %s' % feat)

        # Distributed computation
        analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
        results_file = os.path.join(results_dir, analysis_id + '.pkl')

        if os.path.exists(results_file):
            print('%s is already done. Skipped.' % analysis_id)
            continue

        dist = DistComp(lockdir='tmp', comp_id=analysis_id)
        if dist.islocked():
            print('%s is already running. Skipped.' % analysis_id)
            continue

        dist.lock()

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

        '''
        y_train = y[train_indexes, :]
        y_test = y[test_indexes, :]
        y_label_train = y_label[train_indexes, :]
        y_label_test = y_label[test_indexes, :]

        y_train_sorted, train_labels_sorted = get_refdata(y_train, y_label_train, labels)  # Image features corresponding to brain data
        y_test_sorted, test_labels_sorted = get_refdata(y_test, y_label_test, labels)
        '''
        x = x[:1200, :]

        x_train = x[train_indexes, :]
        x_test = x[test_indexes, :]

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

        # Feature prediction
        pred_y, true_y, test_y_predicted, test_y_true = feature_prediction(x_train, y_train,
                                            x_test, y_test,
                                            n_voxel=num_voxel[roi],
                                            n_iter=n_iter)

        def softmax(x):
            f_x = np.exp(x) / np.sum(np.exp(x))
            return f_x

        print(test_y_predicted.shape)
        print(test_y_predicted)
        print(test_y_true.shape)
        print(test_y_true)
        answers = []
        save_numpy = []
        for i in range(len(test_y_predicted)):
            pred_y = softmax(test_y_predicted[i].squeeze())
            print(np.argmax(pred_y))
            print(test_y_true[i])
            correct = (test_y_true[i] == np.argmax(pred_y))
            answers.append(correct)
            save_numpy.append(pred_y)
            print("RANK: ", 150 - pred_y.argsort().argsort()[int(test_y_true[i])])
        
        print(np.array(save_numpy))
        print(np.array(save_numpy).shape)
        numpy_name = "soft_labels/"+analysis_id+".npy"
        np.save(numpy_name, np.array(save_numpy))
        '''
        for i in range(len(test_y_predicted)):
            test_diff = np.abs(test_y_true - test_y_predicted[i])
            print(test_y_predicted[i].shape)
            print(test_diff.shape)
            closest_label = np.argmin(np.sum(test_diff, axis=1))
            print(closest_label)
            correct = (closest_label == i)
            answers.append(correct)
        '''
        print(answers)
        if True not in answers:
            acc = 0
        else:
            corr = 0
            err = 0
            for answ in answers:
                if answ == True:
                    corr += 1
                else:
                    err += 1
            acc = corr / (corr + err)
        print("Accuracy: ", acc)        


        dist.unlock()


# Functions ############################################################

def feature_prediction(x_train, y_train, x_test, y_test, n_voxel=500, n_iter=200):
    '''Run feature prediction

    Parameters
    ----------
    x_train, y_train : array_like [shape = (n_sample, n_voxel)]
        Brain data and image features for training
    x_test, y_test : array_like [shape = (n_sample, n_unit)]
        Brain data and image features for test
    n_voxel : int
        The number of voxels
    n_iter : int
        The number of iterations

    Returns
    -------
    predicted_label : array_like [shape = (n_sample, n_unit)]
        Predicted features
    ture_label : array_like [shape = (n_sample, n_unit)]
        True features in test data
    '''
    # Feature prediction for each unit
    print('Running feature prediction')

    test_true_list = []
    test_pred_list = []

    model = resnet18().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)

    tensor_x_train = torch.Tensor(x_train) # transform to torch tensor
    tensor_y_train = torch.Tensor(y_train)
    tensor_x_test = torch.Tensor(x_test)
    tensor_y_test = torch.Tensor(y_test)

    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.train()
    epochs = 80
    y_true_list = []
    y_pred_list = []
    for epoch in range(epochs):
        y_true_list = []
        y_pred_list = []
        start_time = time()
        print("Epoch: ", epoch)
        for i, (input, target) in enumerate(train_dataloader):
            target = target.type(torch.LongTensor).cuda()
            input = torch.unsqueeze(input, 1).cuda()
            #print('Unit', (i + 1))

            optimizer.zero_grad()
            y_pred = model(input)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()

            y_true_list.append(target.cpu().detach().numpy())
            y_pred_list.append(y_pred.cpu().detach().numpy())

            print("Loss: ", loss)

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            save_ckpt(checkpoint)

        save_path = "model1d_weight_"+str(epoch)+".pth"
        torch.save(model.state_dict(), save_path)
        print('Time: %.3f sec' % (time() - start_time))
        scheduler.step()
   
    #save_path = "model_weight_20.pth"
    #model.load_state_dict(torch.load(save_path))
    model.eval()
    for i, (input, target) in enumerate(test_dataloader):
        input = input.cuda()
        target = target.cuda()
        input = torch.unsqueeze(input, 0)
        print(input.shape)

        print('Unit', (i + 1))
        start_time = time()

        optimizer.zero_grad()
        y_pred = model(input)

        test_true_list.append(target.cpu().detach().numpy())
        test_pred_list.append(y_pred.cpu().detach().numpy())

        print('Time: %.3f sec' % (time() - start_time))
    
    # Create numpy arrays for return values
    y_predicted = np.array(y_pred_list)
    y_true = np.array(y_true_list)
    
    test_y_predicted = np.array(test_pred_list)
    test_y_true = np.array(test_true_list)

    return y_predicted, y_true, test_y_predicted, test_y_true


def get_averaged_feature(pred_y, true_y, labels):
    '''Return category-averaged features'''

    labels_set = np.unique(labels)

    pred_y_av = np.array([np.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = np.array([np.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


# Run as a scirpt ######################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()
