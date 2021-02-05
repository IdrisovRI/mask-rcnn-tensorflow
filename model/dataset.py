import os
import numpy as np
from PIL import Image
from numpy.core._multiarray_umath import zeros
from model.config import OBJECTS
from mask_rcnn.mrcnn.utils import Dataset


# class that defines and loads the kangaroo dataset
class PrepareDataSet(Dataset):
    # load the dataset definitions
    def load_dataset(self, data_path: str):
        # define one class
        self.add_class("dataset", 1, OBJECTS[1])
        self.add_class("dataset", 2, OBJECTS[2])
        self.add_class("dataset", 3, OBJECTS[3])

        image_id = 0
        # find all images
        for filename in os.listdir(data_path):
            # extract image id
            short_name = filename[:-4]

            if filename[-3:] == 'txt':
                continue
            image_id += 1
            # # skip bad images
            # if image_id in ['00090']:
            #     continue

            # Пути к файлам с изображениями и разметкой объектов
            img_path = os.path.join(data_path, filename)
            ann_path = os.path.join(data_path, f'{short_name}.txt')
            # Если текстовый файл с расшифровками объектов отсутствует, то пропускаем его
            if not os.path.exists(ann_path):
                continue
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def get_img_path(self, image_id: int):
        return self.image_info[image_id]['path']

    def get_img_annotation(self, image_id: int):
        return self.image_info[image_id]['annotation']

    def get_img_size(self, image_path):
        # Открывается файл с изображением
        img = Image.open(image_path)
        # Извлекаются параметры исходного изображения из файла
        width, height = img.size
        img.close()
        return width, height

    def extract_boxes(self, image_id: int):
        image_path = self.get_img_path(image_id)
        annotation_path = self.get_img_annotation(image_id)
        img_width, img_height = self.get_img_size(image_path)

        with open(annotation_path, 'r') as f:
            rows = f.read().splitlines()

            boxes = list()
            for row in rows:
                # Разбиваем файл по разделителю пробелу
                row_params = row.split(' ')

                # Высчитываем параметры выделения(Делаем смещение на 1 относительно модели YOLO)
                # В MaskRCNN под нулевой слой идет Background
                class_id = int(row_params[0]) + 1
                x_min = int(float(row_params[1]) * img_width) - int(float(row_params[3]) * img_width / 2)
                y_min = int(float(row_params[2]) * img_height) - int(float(row_params[4]) * img_height / 2)
                x_max = int(float(row_params[1]) * img_width) + int(float(row_params[3]) * img_width / 2)
                y_max = int(float(row_params[2]) * img_height) + int(float(row_params[4]) * img_height / 2)
                coors = [x_min, y_min, x_max, y_max, class_id]
                boxes.append(coors)

        return boxes, img_width, img_height

    # load the masks for an image
    def load_mask(self, image_id: int):
        # Получаем параметры изображения
        boxes, w, h = self.extract_boxes(image_id)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(box[4])

        # Pack instance masks into an array
        if class_ids:
            # mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
        else:
            # Call super class to return an empty mask
            masks = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)
        return masks, class_ids
