from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module
class KittiDataset(CocoDataset):
    CLASSES = ('Car')
