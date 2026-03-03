"""
KITTI detection dataset support for RT-DETRv4.

Label format reference:
  type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom ...
"""

import os
from PIL import Image
import torch

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class KittiDetection(DetDataset):
    __inject__ = ['transforms']

    def __init__(self, img_folder, label_folder, transforms=None):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.transforms = transforms

        label_files = [
            f for f in os.listdir(label_folder) if f.endswith(".txt")
        ]
        label_files.sort()

        self.label_files = [os.path.join(label_folder, f) for f in label_files]
        self.image_files = []
        for fname in label_files:
            stem = os.path.splitext(fname)[0]
            img_path = os.path.join(img_folder, stem + ".png")
            if not os.path.exists(img_path):
                img_path = os.path.join(img_folder, stem + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(img_folder, stem + ".jpeg")
            self.image_files.append(img_path)

        self.class_map = {
            "Pedestrian": 0,
            "Person_sitting": 0,
            "Car": 1,
            "Van": 1,
            "Truck": 1,
            "Tram": 1,
            "Bus": 1,
            "Train": 1,
            "Cyclist": 2,
        }

    def __getitem__(self, index):
        image, target = self.load_item(index)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)
        return image, target

    def load_item(self, index):
        image = Image.open(self.image_files[index]).convert("RGB")
        w, h = image.size

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        with open(self.label_files[index], "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 8:
                continue
            obj_type = parts[0]
            if obj_type == "DontCare":
                continue
            if obj_type not in self.class_map:
                continue
            left, top, right, bottom = map(float, parts[4:8])
            if right <= left or bottom <= top:
                continue
            boxes.append([left, top, right, bottom])
            labels.append(self.class_map[obj_type])
            areas.append((right - left) * (bottom - top))
            iscrowd.append(0)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        boxes = convert_to_tv_tensor(boxes, 'boxes', box_format='xyxy', spatial_size=[h, w])

        target = {
            "image_id": torch.tensor([index]),
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([w, h]),
        }

        return image, target
