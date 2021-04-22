import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict

import torch


def get_coco_img_ids(coco):
    """
    Return a list of image ids according to annotations
    (use in case annotations were changed after coco loading).
    """
    img_ids = set()
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_ids.add(ann['image_id'])

    return list(img_ids)


def load_coco_samples(coco):
    samples = []

    img_ids = get_coco_img_ids(coco)
    for img_meta in coco.loadImgs(img_ids):
        image_id = img_meta['id']
        file_name = img_meta['file_name']
        height = img_meta['height']
        width = img_meta['width']
        anns = coco.loadAnns(coco.getAnnIds(image_id))

        sample = {
            'image_id': image_id,
            'file_name': file_name,
            'height': height,
            'width': width,
            'anns': anns
        }
        samples.append(sample)

    return samples


def save_coco_anns(anns, filename_to_save):
    with open(filename_to_save, 'w') as file:
        json.dump(anns, file)


def check_bbox_validity(bbox, format='coco'):
    if format == 'coco':
        is_valid = True
        if bbox[0] < 0 or bbox[1] < 0:
            is_valid = False

        if bbox[1] <= 0 or bbox[2] <= 0:
            is_valid = False

    else:
        raise NotImplementedError('unknown bbox format')

    return is_valid


def get_bbox_scale(coco, ann):
    image_id = ann['image_id']
    bbox = ann['bbox']

    image = coco.loadImgs(ids=[image_id])[0]
    width, height = image['width'], image['height']

    x_scale = round(bbox[2] / width, 2)
    y_scale = round(bbox[3] / height, 2)

    return x_scale, y_scale


def get_category_based_anns(coco):
    coco_samples = load_coco_samples(coco)

    category_based_anns = []

    for sample in coco_samples:
        file_name = sample['file_name']
        anns = sample['anns']

        category_dict = defaultdict(list)
        for ann in anns:
            ann.pop("segmentation", None)
            ann.pop("keypoints", None)

            category_id = ann['category_id']
            category_dict[category_id].append(ann)

        for key, item in category_dict.items():
            instance_ann = {
                'image_id': sample['image_id'],
                'file_name': file_name,
                'anns': item
                }
            category_based_anns.append(instance_ann)

    return category_based_anns


def get_kps_set2idx(anns, idx2kps):
    kps_sets = set()
    for ann in anns:
        kps_visibility = ann['keypoints'][2::3]
        kps_set = frozenset(idx2kps[i] for i, v in enumerate(kps_visibility) if v == 2)
        kps_sets.add(kps_set)

    return {kps_set: i for i, kps_set in enumerate(list(kps_sets))}


def get_anns_info_df(coco, save=None):
    anns = coco.loadAnns(coco.getAnnIds())
    cats = coco.cats

    anns_info = defaultdict(list)
    for i, ann in enumerate(anns):
        id = ann['id']
        image_id = ann['image_id']
        is_crowd = ann['iscrowd']
        bbox = ann['bbox']

        anns_info['id'].append(id)
        anns_info['image_id'].append(image_id)
        anns_info['category_id'].append(ann['category_id'])
        anns_info['category'].append(cats[ann['category_id']]['name'])
        anns_info['is_crowd'].append(is_crowd)

        if not is_crowd:
            anns_info['bbox_x'].append(bbox[0])
            anns_info['bbox_y'].append(bbox[1])
            anns_info['bbox_width'].append(bbox[2])
            anns_info['bbox_height'].append(bbox[3])

            x_scale, y_scale = get_bbox_scale(coco, ann)
            anns_info['bbox_x_scale'].append(x_scale)
            anns_info['bbox_y_scale'].append(y_scale)

        else:
            anns_info['bbox_x'].append(-1)
            anns_info['bbox_y'].append(-1)
            anns_info['bbox_width'].append(-1)
            anns_info['bbox_height'].append(-1)

            anns_info['bbox_x_scale'].append(-1)
            anns_info['bbox_y_scale'].append(-1)

    anns_info = pd.DataFrame(anns_info)

    if save:
        anns_info.to_csv(os.path.join('.', f'data/{save}.csv'))

    return anns_info


def to_yolo_target(bboxes, img_size, stride):
    def get_relative_coords(bbox, img_size, cell_size):
        bbox = xlytwh2xcycwh(bbox)
        x, y = (bbox[0] % cell_size[0]) / cell_size[0], (bbox[1] % cell_size[1]) / cell_size[1]
        w, h = bbox[2] / img_size[0], bbox[3] / img_size[1]
        return [x, y, w, h]

    w, h = img_size
    grid_w, grid_h = w // stride, h // stride
    cell_w, cell_h = w // grid_w, h // grid_h

    target = np.zeros((grid_h, grid_w, 5), dtype=np.float)
    for bbox in bboxes:
        img_xc, img_yc = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
        cell_x, cell_y = int(img_xc / cell_w), int(img_yc / cell_h)

        cell_target = [1., *get_relative_coords(bbox, img_size, (cell_w, cell_h))]
        target[cell_y, cell_x] = np.array(cell_target)

    return target


def from_yolo_target(target, img_size, grid_size):
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    cell_size = img_size[0] // grid_size[0], img_size[1] // grid_size[1]
    new_target = target.copy()

    x_offset = np.expand_dims(np.arange(grid_size[1]), axis=0).repeat(grid_size[0], axis=0) * cell_size[0]
    new_target[:, :, 1] = x_offset + new_target[:, :, 1] * cell_size[0]
    y_offset = np.expand_dims(np.arange(grid_size[0]), axis=0).T.repeat(grid_size[1], axis=1) * cell_size[1]
    new_target[:, :, 2] = y_offset + new_target[:, :, 2] * cell_size[1]

    new_target[:, :, 3] = new_target[:, :, 3] * img_size[0]
    new_target[:, :, 4] = new_target[:, :, 4] * img_size[1]

    new_target[:, :, 1] -= new_target[:, :, 3] // 2
    new_target[:, :, 2] -= new_target[:, :, 4] // 2

    new_target = new_target[new_target[:, :, 0] > 0.5][:, 1:].tolist()

    return new_target


def xlytwh2xcycwh(bbox):
    x = bbox[0] + bbox[2] // 2
    y = bbox[1] + bbox[3] // 2
    w, h = bbox[2], bbox[3]
    return [x, y, w, h]
