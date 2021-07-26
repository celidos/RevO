from collections import OrderedDict

from torchvision.models import resnet
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter
import torch
from torch import nn
from models.simclr_resnet import get_resnet
from models import darknet
from utils.utils import freeze_layers


def resnet_backbone(
        name='resnet50',
        pretrained=False,
        trainable_layers=3,
        returned_layer='layer4',
        norm_layer=None,
        map_location=None,
        **kwargs
):
    """
    :param name: resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
    :param pretrained: If True, returns a model with backbone pre-trained on Imagenet
    :param trainable_layers: number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    :param returned_layer: layer of the network to return.
    """
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    elif norm_layer == 'frozen_bn':
        norm_layer = FrozenBatchNorm2d

    if isinstance(pretrained, str):
        backbone = resnet.__dict__[name](pretrained=False, norm_layer=norm_layer)
        state_dict = torch.load(pretrained, map_location=map_location)
        # I WANNA KILL MY FAMILY AND MYSELF
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        backbone.load_state_dict(state_dict, strict=False)
    else:
        backbone = resnet.__dict__[name](pretrained=pretrained, norm_layer=norm_layer)

    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    if trainable_layers == 5:
        layers_to_train.append('bn1')
    else:
        backbone.bn1.track_running_stats = False

    freeze_layers(backbone, layers_to_train)

    return_layer = {returned_layer: 'output'}

    print('return_layer = {}'.format(return_layer))
    print('backbone layers = {}'.format([name for name, _ in backbone.named_children()]))
    return IntermediateLayerGetter(model=backbone, return_layers=return_layer)


def simclr_backbone(pretrained,
                    name='resnet50',
                    trainable_layers=3,
                    returned_layer=4,
                    width_multiplier=1,
                    sk_ration=0,
                    norm_layer=None,
                    map_location=None,
                    **kwargs
):
    """
    :param name: resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
    :param pretrained: If True, returns a model with backbone pre-trained on Imagenet
    :param trainable_layers: number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    :param returned_layer: layer of the network to return.
    """
    assert isinstance(pretrained, str)
    if norm_layer == 'frozen_bn':
        frozen_bn = True
    else:
        frozen_bn = False

    depth = int(''.join(filter(str.isdigit, name)))

    backbone, _ = get_resnet(depth=depth, width_multiplier=width_multiplier, sk_ratio=sk_ration, frozen_bn=frozen_bn)
    state_dict = torch.load(pretrained, map_location=map_location)['resnet']
    backbone.load_state_dict(state_dict, strict=True)

    assert 0 <= trainable_layers <= 5
    layers_to_train = ['net.4', 'net.3', 'net.2', 'net.1', 'net.0'][:trainable_layers]

    if trainable_layers == 5:
        layers_to_train.append('bn1')
    else:
        backbone.net[0][1][0].track_running_stats = False

    freeze_layers(backbone, layers_to_train)

    assert 0 < returned_layer < 5
    return_layer = {f'{returned_layer}': 'output'}

    return IntermediateLayerGetter(model=backbone.net, return_layers=return_layer)


def resnet_backbone_headed(name='resnet50',
                              pretrained=False,
                              trainable_layers=3,
                              returned_layer=4,
                              head_out_dim=512,
                              norm_layer=None,
                              map_location=None,
                              **kwargs):

    def _extract_layers(model, returned_layer):
        cutoff = 6 - returned_layer
        layers = nn.Sequential(OrderedDict([
            (name, child)
            for name, child in list(model.named_children())[:-cutoff]
        ]))
        return layers

    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    elif norm_layer == 'frozen_bn':
        norm_layer = FrozenBatchNorm2d

    if isinstance(pretrained, str):
        backbone = resnet.__dict__[name](pretrained=False, norm_layer=norm_layer)
        state_dict = torch.load(pretrained, map_location=map_location)
        # I WANNA KILL MY FAMILY AND MYSELF
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        backbone.load_state_dict(state_dict, strict=False)
    else:
        backbone = resnet.__dict__[name](pretrained=pretrained, norm_layer=norm_layer)

    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    if trainable_layers == 5:
        layers_to_train.append('bn1')
    else:
        backbone.bn1.track_running_stats = False

    freeze_layers(backbone, layers_to_train)

    assert 0 < returned_layer < 5

    feat_extr = _extract_layers(backbone, returned_layer)

    head_in_dim = getattr(backbone, f'layer{returned_layer}')[-1].conv3.out_channels
    head = nn.Conv2d(in_channels=head_in_dim, out_channels=head_out_dim, kernel_size=1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)

    backbone = nn.Sequential(OrderedDict([
        ('feat_extr', feat_extr),
        ('head', head)
    ]))
    return_layer = {f'head': 'output'}
    return IntermediateLayerGetter(model=backbone, return_layers=return_layer)


def darknet_backbone(pretrained,
                     name='darknet53',
                     trainable_layers=4,
                     returned_layer=5,
                     norm_layer=None,
                     map_location=None,
                     **kwargs):
    assert isinstance(pretrained, str)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    elif norm_layer == 'frozen_bn':
        norm_layer = FrozenBatchNorm2d

    backbone = darknet.__dict__[name](norm_layer=norm_layer)
    state_dict = torch.load(pretrained, map_location=map_location)
    backbone.load_state_dict(state_dict, strict=True)

    assert 0 <= trainable_layers <= 6
    layers_to_train = ['layer5', 'layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    if trainable_layers == 6:
        layers_to_train.append('bn1')
    else:
        backbone.bn1.track_running_stats = False

    freeze_layers(backbone, layers_to_train)

    assert 0 < returned_layer < 6
    return_layer = {f'layer{returned_layer}': 'output'}

    return IntermediateLayerGetter(model=backbone, return_layers=return_layer)

