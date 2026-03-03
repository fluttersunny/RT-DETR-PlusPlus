"""
DIOR detection dataset support for RT-DETRv4 (VOC-style XML).
"""

import os
from PIL import Image
import torch

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class DiorDetection(DetDataset):
    __inject__ = ['transforms']

    def __init__(self, img_folder, ann_folder, img_set, transforms=None):
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.transforms = transforms

        with open(img_set, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        self.image_ids = lines
        self.image_files = [self._resolve_image_path(x) for x in self.image_ids]
        self.ann_files = [self._resolve_ann_path(x) for x in self.image_ids]

        self.class_names = [
            "airplane",
            "airport",
            "baseballfield",
            "basketballcourt",
            "bridge",
            "chimney",
            "dam",
            "Expressway-Service-area",
            "Expressway-toll-station",
            "golffield",
            "groundtrackfield",
            "harbor",
            "overpass",
            "ship",
            "stadium",
            "storagetank",
            "tenniscourt",
            "trainstation",
            "vehicle",
            "windmill",
        ]
        self.class_map = {name: idx for idx, name in enumerate(self.class_names)}

    def _resolve_image_path(self, image_id):
        if os.path.splitext(image_id)[1]:
            filename = image_id
        else:
            filename = image_id + ".jpg"
        path = os.path.join(self.img_folder, filename)
        if os.path.exists(path):
            return path
        alt = os.path.join(self.img_folder, image_id + ".png")
        if os.path.exists(alt):
            return alt
        alt = os.path.join(self.img_folder, image_id + ".jpeg")
        if os.path.exists(alt):
            return alt
        return path

    def _resolve_ann_path(self, image_id):
        if os.path.splitext(image_id)[1]:
            filename = os.path.splitext(image_id)[0] + ".xml"
        else:
            filename = image_id + ".xml"
        return os.path.join(self.ann_folder, filename)

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

        root = ET_parse(self.ann_files[index]).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", default="").strip()
            if name not in self.class_map:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.findtext("xmin", default="0"))
            ymin = float(bnd.findtext("ymin", default="0"))
            xmax = float(bnd.findtext("xmax", default="0"))
            ymax = float(bnd.findtext("ymax", default="0"))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_map[name])
            areas.append((xmax - xmin) * (ymax - ymin))
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
