# +
import os
from collections import defaultdict
from operator import itemgetter
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from tqdm import tqdm
# -

from utils.data import get_category_based_anns, to_yolo_target


class FewShotDataset(Dataset):
    def __init__(
            self,
            q_root,
            s_root,
            q_ann_filename,
            s_ann_filename,
            k_shot,
            q_img_size,
            backbone_stride,
            q_transform=None,
            s_transform=None
    ):
        super(Dataset, self).__init__()
        self.k_shot = k_shot
        self.q_root = q_root
        self.s_root = s_root
        self.q_img_size = q_img_size
        self.backbone_stride = backbone_stride
        self.q_transform = q_transform
        self.s_transform = s_transform

        from pycocotools.coco import COCO
        self.q_coco = COCO(q_ann_filename)
        self.s_coco = COCO(s_ann_filename)
        self.q_anns = get_category_based_anns(self.q_coco)
        self.s_anns = get_category_based_anns(self.s_coco)

        self.s_anns_by_categories = defaultdict(set)
        for i, item in enumerate(self.s_anns):
            self.s_anns_by_categories[item['anns'][0]['category_id']].add(i)

        self.cats = self.q_coco.cats

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.q_anns)

    @staticmethod
    def _imread(filename, flags=None):
        img = cv2.imread(filename, flags)

        if img is None:
            return None
        
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class ObjectDetectionDataset(FewShotDataset):
    def __init__(
            self,
            q_root,
            s_root,
            q_ann_filename,
            s_ann_filename,
            k_shot,
            q_img_size,
            backbone_stride,
            q_transform=None,
            s_transform=None
    ):
        super().__init__(q_root, s_root, q_ann_filename, s_ann_filename, k_shot,
                         q_img_size, backbone_stride, q_transform, s_transform)
        
        self.__preload_hard()

    def __preload_hard(self):
        print("Preloading data q...")
        # ----------------------------
        
        self.cached_q_img = []
        self.cached_q_bbox = []
        self.cached_q_bbox_cat = []
        tqrange = tqdm(range(len(self.q_anns)))
        for idx in tqrange:
            q_ann = self.q_anns[idx]
            q_bbox = list(map(lambda ann: ann['bbox'], q_ann['anns']))
            q_bbox_cat = list(map(lambda ann: ann['category_id'], q_ann['anns']))

            q_file_name = q_ann['file_name']
            tqrange.set_description("{}".format(os.path.basename(q_file_name).ljust(40)), refresh=True)
            q_img = self._imread(os.path.join(self.q_root, q_file_name), cv2.COLOR_BGR2RGB)
            if q_img is not None:
                if self.q_transform:
                    q_transformed = self.q_transform(image=q_img, bboxes=q_bbox, bboxes_cats=q_bbox_cat)
                    q_img = q_transformed['image']
                    q_bbox = list(map(list, q_transformed['bboxes']))
            
            self.cached_q_img.append(q_img)
            self.cached_q_bbox.append(q_bbox)
            self.cached_q_bbox_cat.append(q_bbox_cat)
            
        # -----------------------------
        
        self.cached_s_img = []
        self.cached_s_bbox = []
        self.cached_s_bbox_cat = []
        tqrange = tqdm(range(len(self.s_anns)))
        for idx in tqrange:
            s_ann = self.s_anns[idx]
            s_bbox = list(map(lambda ann: ann['bbox'], s_ann['anns']))
            s_bbox_cat = list(map(lambda ann: ann['category_id'], s_ann['anns']))

            s_file_name = s_ann['file_name']
            tqrange.set_description("{}".format(os.path.basename(s_file_name).ljust(40)), refresh=True)
            s_img = self._imread(os.path.join(self.s_root, s_file_name), cv2.COLOR_BGR2RGB)
            if s_img is not None:
                if self.s_transform:
                    s_transformed = self.s_transform(image=s_img, bboxes=s_bbox, bboxes_cats=s_bbox_cat)
                    s_img = s_transformed['image']
                    s_bboxes = list(map(list, s_transformed['bboxes']))
            
            self.cached_s_img.append(s_img)
            self.cached_s_bbox.append(s_bbox)
            self.cached_s_bbox_cat.append(s_bbox_cat)
            
        
    def __getitem__(self, idx: int):
        """
        Returns:
             sample (dict): data sample
                sample['input'] (dict): input data
                    sample['input']['q_img'] (np.ndarray or torch.Tensor): query image
                    sample['input']['s_imgs'] (List[np.ndarray] or List[torch.Tensor]): K_SHOT support images
                    sample['input']['s_bboxes'] (List[List[float]]): bbox coordinates for support images
                sample['target'] (List[float]): target object presence map (vector actually)
        """
        
        """q_ann = self.q_anns[idx]
        q_bbox = list(map(lambda ann: ann['bbox'], q_ann['anns']))
        q_bbox_cat = list(map(lambda ann: ann['category_id'], q_ann['anns']))

        q_file_name = q_ann['file_name']
        q_img = self._imread(os.path.join(self.q_root, q_file_name), cv2.COLOR_BGR2RGB)

        s_anns_idxs = np.random.choice(list(self.s_anns_by_categories[q_bbox_cat[0]] - {idx}), self.k_shot)
        s_anns = itemgetter(*s_anns_idxs)(self.s_anns)
        if len(s_anns_idxs) == 1:
            s_anns = (s_anns, )
        s_bboxes = []
        s_bbox_cats = []
        s_imgs = []
        for s_ann in s_anns:
            s_bbox = list(map(lambda ann: ann['bbox'], s_ann['anns']))
            s_bbox_cat = list(map(lambda ann: ann['category_id'], s_ann['anns']))

            s_file_name = s_ann['file_name']
            s_img = self._imread(os.path.join(self.s_root, s_file_name), cv2.COLOR_BGR2RGB)

            s_bbox_cats.append(s_bbox_cat)
            s_bboxes.append(s_bbox)
            s_imgs.append(s_img)

        if self.q_transform:
            q_transformed = self.q_transform(image=q_img, bboxes=q_bbox, bboxes_cats=q_bbox_cat)
            q_img = q_transformed['image']
            q_bbox = list(map(list, q_transformed['bboxes']))

        if self.s_transform:
            s_transformed = [
                self.s_transform(image=s_img, bboxes=s_bbox, bboxes_cats=s_bbox_cat)
                for s_img, s_bbox, s_bbox_cat in zip(s_imgs, s_bboxes, s_bbox_cats)
            ]
            s_imgs = [transformed['image'] for transformed in s_transformed]
            s_bboxes = [list(map(list, transformed['bboxes'])) for transformed in s_transformed]"""
        
        q_img = self.cached_q_img[idx]
        q_bbox = self.cached_q_bbox[idx]
        q_bbox_cat = self.cached_q_bbox_cat[idx]
        
        s_anns_idxs = np.random.choice(list(self.s_anns_by_categories[q_bbox_cat[0]] - {idx}), self.k_shot)
        s_bboxes = itemgetter(*s_anns_idxs)(self.cached_s_bbox)
#         s_imgs = itemgetter(*s_anns_idxs)(self.cached_s_img)
        s_imgs = [self.cached_s_img[s_idx].detach() for s_idx in s_anns_idxs]

        target = to_yolo_target(q_bbox, self.q_img_size, self.backbone_stride)
        sample = {'input': {}, 'target': []}
        sample['input']['q_img'] = q_img
        sample['input']['s_imgs'] = s_imgs
        sample['input']['s_bboxes'] = s_bboxes
        sample['target'] = target
        return sample


class ObjectClassificationDataset(FewShotDataset):
    def __init__(
            self,
            q_root,
            s_root,
            q_ann_filename,
            s_ann_filename,
            k_shot,
            q_img_size,
            backbone_stride,
            q_transform=None,
            s_transform=None
    ):
        super().__init__(q_root, s_root, q_ann_filename, s_ann_filename, k_shot,
                         q_img_size, backbone_stride, q_transform, s_transform)

    def __getitem__(self, idx: int):
        """
        Returns:
             sample (dict): data sample
                sample['input'] (dict): input data
                    sample['input']['q_img'] (np.ndarray or torch.Tensor): query image
                    sample['input']['s_imgs'] (List[np.ndarray] or List[torch.Tensor]): K_SHOT support images
                    sample['input']['s_bboxes'] (List[List[float]]): bbox coordinates for support images
                sample['target'] (List[float]): target object presence map (vector actually)
        """
        q_ann = self.q_anns[idx]
        q_cat = q_ann['anns'][0]['category_id']

        q_file_name = q_ann['file_name']
        q_img = self._imread(os.path.join(self.q_root, q_file_name), cv2.COLOR_BGR2RGB)

        class_idx = np.random.choice([q_cat] * len(self.cats) + list(range(len(self.cats))))

        s_anns_idxs = np.random.choice(list(self.s_anns_by_categories[class_idx] - {idx}), self.k_shot)
        s_anns = itemgetter(*s_anns_idxs)(self.s_anns)
        if len(s_anns_idxs) == 1:
            s_anns = (s_anns, )

        s_imgs = []
        s_cats = []
        for s_ann in s_anns:
            s_cat = s_ann['anns'][0]['category_id']
            s_file_name = s_ann['file_name']
            s_img = self._imread(os.path.join(self.s_root, s_file_name), cv2.COLOR_BGR2RGB)

            s_cats.append(s_cat)
            s_imgs.append(s_img)

        if self.q_transform:
            q_transformed = self.q_transform(image=q_img, bboxes=[], bboxes_cats=[])
            q_img = q_transformed['image']

        if self.s_transform:
            s_transformed = [
                self.s_transform(image=s_img, bboxes=[], bboxes_cats=[])
                for s_img in s_imgs
            ]
            s_imgs = [transformed['image'] for transformed in s_transformed]

        target = int(idx == class_idx)
        sample = {'input': {}, 'target': []}
        sample['input']['q_img'] = q_img
        sample['input']['s_imgs'] = s_imgs
        sample['input']['s_bboxes'] = []
        sample['target'] = target
        return sample


def object_detection_collate_fn(batch):
    q_img_batched = default_collate([sample['input']['q_img'] for sample in batch])
    target_batched = torch.as_tensor([sample['target'] for sample in batch], dtype=torch.float)

    if type(batch[0]['input']['s_imgs'][0]).__module__ == 'numpy':
        s_imgs_batched = default_collate(np.array([np.array(sample['input']['s_imgs']) for sample in batch]))
    elif type(batch[0]['input']['s_imgs'][0]).__module__ == 'torch':
        s_imgs_batched = torch.stack([torch.stack(sample['input']['s_imgs']) for sample in batch])
    else:
        raise TypeError('Unknown type of support image')

    s_bboxes_batched = [sample['input']['s_bboxes'] for sample in batch]

    sample_batched = {'input': {}, 'target': []}
    sample_batched['input']['q_img'] = q_img_batched
    sample_batched['input']['s_imgs'] = s_imgs_batched
    sample_batched['input']['s_bboxes'] = s_bboxes_batched
    sample_batched['target'] = target_batched

    return sample_batched


def object_classification_collate_fn(batch):
    q_img_batched = default_collate([sample['input']['q_img'] for sample in batch])
    target_batched = torch.as_tensor([sample['target'] for sample in batch], dtype=torch.float)

    if type(batch[0]['input']['s_imgs'][0]).__module__ == 'numpy':
        s_imgs_batched = default_collate(np.array([np.array(sample['input']['s_imgs']) for sample in batch]))
    elif type(batch[0]['input']['s_imgs'][0]).__module__ == 'torch':
        s_imgs_batched = torch.stack([torch.stack(sample['input']['s_imgs']) for sample in batch])
    else:
        raise TypeError('Unknown type of support image')


    sample_batched = {'input': {}, 'target': []}
    sample_batched['input']['q_img'] = q_img_batched
    sample_batched['input']['s_imgs'] = s_imgs_batched
    sample_batched['target'] = target_batched

    return sample_batched



