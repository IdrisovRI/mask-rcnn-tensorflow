import sys
from typing import List
from model.dataset import PrepareDataSet
from helper.rcnn_lib import add_mask_rcnn

# Добавляется в поиск библиотека MaskRCNN
add_mask_rcnn(sys.argv[0])
from mask_rcnn.mrcnn.utils import Dataset
from mask_rcnn.samples.coco.coco import CocoDataset
from dataset.vgg_coco import VggCocoDataset


class DataSetFactory:
    @staticmethod
    def new_instance(set_dir: str) -> Dataset:
        if set_dir:
            train_set = PrepareDataSet()
            train_set.load_dataset(set_dir)
            train_set.prepare()
            print(f'New data set. Path: {set_dir} Images:{len(train_set.image_ids)}')

            return train_set


class CocoSetFactory:
    @staticmethod
    def new_instance(set_dir: str, subset: str, year: int, class_ids: List[int] = None) -> Dataset:
        if set_dir:
            data_set = CocoDataset()
            data_set.load_coco(set_dir, subset, year, class_ids)
            data_set.prepare()
            print(f'New coco data set. Path: {set_dir} Images:{len(data_set.image_ids)}')

            return data_set


class VGGCocoSetFactory:
    @staticmethod
    def new_instance(set_dir: str, subset: str, year: int, class_ids: List[int] = None) -> Dataset:
        if set_dir:
            data_set = VggCocoDataset()
            data_set.load_coco(set_dir, subset, year, class_ids)
            data_set.prepare()
            print(f'New coco data set. Path: {set_dir} Images:{len(data_set.image_ids)}')

            return data_set