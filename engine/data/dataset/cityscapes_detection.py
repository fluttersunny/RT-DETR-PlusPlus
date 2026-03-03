"""
Cityscapes detection dataset wrapper for COCO-style annotations.
"""

from .coco_dataset import CocoDetection
from ...core import register


@register()
class CityscapesDetection(CocoDetection):
    """Cityscapes dataset with COCO-format annotations.

    Cityscapes annotations are typically provided as a single COCO-style JSON.
    The config may use `ann_folder` for backward compatibility; we treat it as
    `ann_file` when provided.
    """

    __inject__ = ['transforms']

    def __init__(
        self,
        img_folder,
        ann_file=None,
        ann_folder=None,
        transforms=None,
        return_masks=False,
        remap_mscoco_category=True,
    ):
        if ann_folder is not None:
            ann_file = ann_folder
        super().__init__(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=remap_mscoco_category,
        )
