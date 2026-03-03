"""
BDD100K detection dataset wrapper for COCO-style annotations.
"""

from .coco_dataset import CocoDetection
from ...core import register


@register()
class BDD100KDetection(CocoDetection):
    """BDD100K dataset with COCO-format annotations."""

    __inject__ = ['transforms']

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms=None,
        return_masks=False,
        remap_mscoco_category=True,
    ):
        super().__init__(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=remap_mscoco_category,
        )
