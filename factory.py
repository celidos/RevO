import torch
import torch.nn as nn
# from utils.utils import object_from_dict
from torch.utils.data import DataLoader, SubsetRandomSampler

from datasets import dataset as datasets
import models
import loss as losses

import metrics
import callbacks


import albumentations as A
import albumentations.pytorch
import numpy as np


def create_model(cfg):
    input_size = cfg.data.input_size
    backbone = models.backbone.resnet_backbone(
        name=cfg.model.backbone.name,
        pretrained=cfg.model.backbone.pretrained,
        trainable_layers=cfg.model.backbone.trainable_layers,
        returned_layer=cfg.model.backbone.returned_layer,
        stride=cfg.model.backbone.stride,
        norm_layer=cfg.model.backbone.norm_layer)

    model = models.prikol.PrikolNet(
        backbone=backbone,
        pool_shape=input_size,
        embd_dim=cfg.model.embd_dim,
        n_head=cfg.model.n_head,
        attn_pdrop=cfg.model.attn_pdrop,
        resid_pdrop=cfg.model.resid_pdrop,
        embd_pdrop=cfg.model.embd_pdrop,
        n_layer=cfg.model.n_layer,
        out_dim=cfg.model.out_dim,
    )
    return model


def create_optimizer(cfg, model: torch.nn.Module):
    if cfg.optimizer.type == 'torch.optim.Adam':
        optimizer = torch.optim.Adam(
            lr=cfg.optimizer.lr,
            params=filter(lambda x: x.requires_grad, model.parameters())
        )
    else:
        raise NotImplementedError('Unknown optimizer: {}'.format(cfg.optimizer.type))
    return optimizer


def create_scheduler(cfg, optimizer: torch.optim.Optimizer):
    if cfg.scheduler.type == 'torch.optim.lr_scheduler.StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma,
            optimizer=optimizer
        )
    else:
        raise NotImplementedError('Unknown scheduler: {}'.format(cfg.scheduler.type))
    return scheduler


def create_loss(cfg):
    if cfg.loss.type == 'loss.CustomYOLOLoss':
        bbox_criterion = losses.EIoULoss(reduction=cfg.loss.bbox_criterion.reduction)
        conf_criterion = nn.BCEWithLogitsLoss()

        loss = losses.CustomYOLOLoss(
            img_size=cfg.loss.img_size,
            grid_size=cfg.loss.grid_size,
            bbox_criterion=bbox_criterion,
            conf_criterion=conf_criterion,
            lambda_noobj=cfg.loss.lambda_noobj,
            lambda_bbox=cfg.loss.lambda_bbox,
            lambda_obj=cfg.loss.lambda_obj
        )
    else:
        raise NotImplementedError('Unknown loss!')
    return loss


def create_train_dataloader(cfg):
    train_dataloaders = dict()
    for dataset_cfg in cfg.data.train_dataset:
        dataset = create_dataset(dataset_cfg)
        dataloader_dict = create_dataloader(dataset_cfg, dataset)
        train_dataloaders[dataloader_dict['name']] = dataloader_dict
    return train_dataloaders


def create_val_dataloader(cfg):
    val_dataloaders = dict()
    for dataset_cfg in cfg.data.validation_dataset:
        dataset = create_dataset(dataset_cfg)
        dataloader_dict = create_dataloader(dataset_cfg, dataset)
        val_dataloaders[dataloader_dict['name']] = dataloader_dict
    return val_dataloaders


def create_dataset(cfg):

    q_transform = create_augmentations(cfg.transforms)
    s_transform = create_augmentations(cfg.transforms)

    print('Creating {} dataset'.format(cfg.type))
    if cfg.type == 'datasets.ObjectDetectionDataset':
        dataset = datasets.ObjectDetectionDataset(
            q_root=cfg.query.root,
            s_root=cfg.support.root,
            q_ann_filename=cfg.query.annotations,
            s_ann_filename=cfg.support.annotations,
            k_shot=cfg.k_shot,
            q_img_size=cfg.input_size,
            backbone_stride=cfg.backbone_stride,
            q_transform=q_transform,
            s_transform=s_transform
        )
    else:
        raise NotImplementedError('Implement creation of {} manually!'.format(cfg.type))

    return dataset


def create_dataloader(cfg, dataset):
    batch_size = cfg.bs
    dataset_length = cfg.len
    shuffle = cfg.shuffle

    if dataset_length:
        if shuffle:
            idx = np.random.choice(len(dataset), dataset_length, replace=False)
            shuffle = False
        else:
            idx = np.arange(dataset_length)

        sampler = SubsetRandomSampler(indices=idx)
    else:
        sampler = None

    if cfg.collate_fn.type == 'datasets.object_detection_collate_fn':
        collate_fn = datasets.object_detection_collate_fn
    else:
        raise NotImplementedError('Implement creation of collate_fn {} manually!'.format(cfg.collate_fn.type))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn
    )

    dataloader_dict = {
        'name': cfg.name,
        'dataloader': dataloader,
        'draw': cfg.draw,
    }

    return dataloader_dict


def create_metrics(cfg):
    ms = []

    ms.append(metrics.Recall)
    ms.append(metrics.Precision)
    ms.append(metrics.AveragePrecision)
    ms.append(metrics.IoU)

    return ms


def create_device(cfg):
    return torch.device(cfg.device)


def create_callbacks(cfg, trainer):
    trainer.register_callback(callbacks.LogCallback(frequency=20))
    trainer.register_callback(callbacks.ValidationCallback(frequency=300))
    trainer.register_callback(callbacks.TensorBoardCallback(frequency=20))
    trainer.register_callback(callbacks.SaveCheckpointCallback(frequency=5000))


def create_augmentations(cfg):
    print(type(cfg))

    transform = A.Compose([
        A.Resize(cfg['resize_height'], cfg['resize_width']),
        A.Normalize(),
        albumentations.pytorch.transforms.ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['bboxes_cats']))

    return transform
