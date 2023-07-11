"""
    Alaug interface.
"""
import random
import collections

import albumentations as al
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module
class AlAugment:

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        # init as None
        self.__augmentor = None
        self.keypoint_params = None
        self.transforms = []

        for transform in transforms:
            if isinstance(transform, dict):
                if transform['type'] == 'Compose':
                    self.get_al_params(transform['params'])
                else:
                    transform = self.build_transforms(transform)
                    if transform is not None:
                        self.transforms.append(transform)
            else:
                raise TypeError('transform must be a dict')
        self.build()

    def get_al_params(self, compose):
        if compose['bboxes']:
            raise ValueError(f'Unsupported type bboxes of compose.')
            # self.bbox_params = al.BboxParams(
            #     format='pascal_voc',
            #     min_area=0.0,
            #     min_visibility=0.0,
            #     label_fields=["bbox_labels"])
        if compose['keypoints']:
            # https://albumentations.ai/docs/getting_started/keypoints_augmentation/
            # Albumentations won't return invisible keypoints. remove_invisible is set to True by default,
            # so if you don't pass that argument, Albumentations won't return invisible keypoints.
            self.keypoint_params = al.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True)

    def build_transforms(self, transform):
        if transform['type'] == 'OneOf':
            transforms = transform['transforms']
            choices = []
            for t in transforms:
                parmas = {
                    key: value
                    for key, value in t.items() if key != 'type'
                }
                choice = getattr(al, t['type'])(**parmas)
                choices.append(choice)
            return getattr(al, 'OneOf')(transforms=choices, p=transform['p'])

        parmas = {
            key: value
            for key, value in transform.items() if key != 'type'
        }
        return getattr(al, transform['type'])(**parmas)

    def build(self):
        if len(self.transforms) == 0:
            return
        self.__augmentor = al.Compose(
            self.transforms,
            keypoint_params=self.keypoint_params,
        )

    def __call__(self, data):
        if self.__augmentor is None:
            return data

        img = data['img']
        if 'keypoints' in data and 'class_labels' in data:
            keypoints = data["keypoints"]
            class_labels = data["class_labels"]
        else:
            raise ValueError('Not founded gt_points in the data dict.')

        aug = self.__augmentor(
            image=img,
            keypoints=keypoints,
            class_labels=class_labels,
        )
        data['img'] = aug['image']
        data['img_shape'] = aug['image'].shape
        data['keypoints'] = aug['keypoints']
        data['class_labels'] = aug['class_labels']
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
