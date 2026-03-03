"""
CrowdHuman dataset support for RT-DETRv4.

CrowdHuman annotations are provided in `.odgt` format (one JSON per line).
We parse `gtboxes[*].fbox` (x, y, w, h) as the default detection box.

This dataset is registered into the engine workspace via `@register()` so it
can be referenced from YAML configs with `type: CrowdHumanDetection`.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register


def _resolve_image_path(img_folder: str, image_id: str) -> str:
    """Resolve image path for a CrowdHuman record ID."""
    # Some dumps use bare IDs without extension; default images are .jpg
    candidates = []
    image_id = str(image_id)
    candidates.append(os.path.join(img_folder, image_id))

    root, ext = os.path.splitext(image_id)
    if ext == "":
        candidates.append(os.path.join(img_folder, root + ".jpg"))
        candidates.append(os.path.join(img_folder, root + ".png"))

    for p in candidates:
        if os.path.exists(p):
            return p

    # Fall back to the first candidate for a more informative error later.
    return candidates[0]


def _parse_ignore_flag(box: Dict[str, Any]) -> int:
    """CrowdHuman uses `extra.ignore` and sometimes `ignore` as flags."""
    ignore = 0
    extra = box.get("extra", None)
    if isinstance(extra, dict) and "ignore" in extra:
        try:
            ignore = int(extra.get("ignore", 0))
        except Exception:
            ignore = 0
    if "ignore" in box:
        try:
            ignore = int(box.get("ignore", ignore))
        except Exception:
            pass
    return 1 if ignore == 1 else 0


@register()
class CrowdHumanDetection(DetDataset):
    __inject__ = ["transforms"]

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms,
        return_masks: bool = False,
        remove_ignore: bool = True,
        box_field: str = "fbox",
    ) -> None:
        """
        Args:
            img_folder: folder containing images.
            ann_file: path to `.odgt` annotation file.
            transforms: transform pipeline (engine.data.transforms.Compose).
            return_masks: kept for API compatibility (CrowdHuman provides boxes only).
            remove_ignore: whether to drop ignored regions/boxes from training targets.
            box_field: which box field to use from each gt entry. Default: 'fbox'.
        """
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.remove_ignore = remove_ignore
        self.box_field = box_field

        self.records: List[Dict[str, Any]] = []
        with open(self.ann_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        # Stable numeric image ids for COCO-style evaluation
        self._id_map: Dict[str, int] = {}
        for i, rec in enumerate(self.records):
            rid = rec.get("ID", str(i))
            self._id_map[str(rid)] = i

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        image, target = self.load_item(idx)
        if self._transforms is not None:
            image, target, _ = self._transforms(image, target, self)
        return image, target

    def load_item(self, idx: int):
        rec = self.records[idx]
        rid = str(rec.get("ID", str(idx)))

        img_path = _resolve_image_path(self.img_folder, rid)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for gt in rec.get("gtboxes", []) or []:
            # CrowdHuman is person-only for our config, ignore other tags
            tag = gt.get("tag", "person")
            if tag not in ("person", "mask", "ignore"):
                continue

            bbox = gt.get(self.box_field, None)
            if bbox is None:
                # fall back to common alternatives
                bbox = gt.get("fbox", None) or gt.get("hbox", None) or gt.get("bbox", None)
            if bbox is None:
                continue

            x, y, bw, bh = [float(v) for v in bbox]
            ignore = _parse_ignore_flag(gt)

            if self.remove_ignore and ignore == 1:
                continue

            x1, y1, x2, y2 = x, y, x + bw, y + bh
            boxes_xyxy.append([x1, y1, x2, y2])
            labels.append(0)  # single-class: person
            areas.append(max(0.0, bw) * max(0.0, bh))
            iscrowd.append(ignore)

        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
        if boxes.numel() > 0:
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) if boxes.numel() > 0 else torch.zeros((0,), dtype=torch.bool)
        boxes = boxes[keep]

        labels_t = torch.as_tensor(labels, dtype=torch.int64)[keep] if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)
        area_t = torch.as_tensor(areas, dtype=torch.float32)[keep] if len(areas) > 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.int64)[keep] if len(iscrowd) > 0 else torch.zeros((0,), dtype=torch.int64)

        image_id = torch.tensor([self._id_map.get(rid, idx)], dtype=torch.int64)

        target: Dict[str, Any] = {
            "boxes": convert_to_tv_tensor(boxes, key="boxes", box_format="xyxy", spatial_size=[h, w]),
            "labels": labels_t,
            "area": area_t,
            "iscrowd": iscrowd_t,
            "image_id": image_id,
            "orig_size": torch.as_tensor([w, h]),
            "idx": torch.tensor([idx]),
        }

        return image, target

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" remove_ignore: {self.remove_ignore}\n box_field: {self.box_field}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        return s

