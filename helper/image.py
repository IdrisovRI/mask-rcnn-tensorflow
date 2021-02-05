import sys
from typing import NoReturn
from matplotlib.patches import Rectangle
from numpy import expand_dims
from matplotlib import pyplot
from helper.rcnn_lib import add_mask_rcnn
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# Добавляется в поиск библиотека MaskRCNN
add_mask_rcnn(sys.argv[0])
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances


class ImageHelper(object):
    @staticmethod
    def show_image(dataset: Dataset) -> NoReturn:
        """
        Позволяет просматривать изображения в наборах данных
        :param dataset:
        :return:
        """
        # define image id
        image_id = 1
        # load the image
        image = dataset.load_image(image_id)
        # load the masks and the class ids
        mask, class_ids = dataset.load_mask(image_id)
        # extract bounding boxes from the masks
        bbox = extract_bboxes(mask)
        # display image with masks and bounding boxes
        display_instances(image, bbox, mask, class_ids, dataset.class_names)

    @staticmethod
    def plot_compare(dataset, mask_model: MaskRCNN, cfg, n_images=3) -> NoReturn:
        """
        Может использоваться для вывода сравнений результатов набора и результатов обучения
        """
        # load image and mask
        for i in range(n_images):
            # load the image and mask
            image = dataset.load_image(i)
            mask, _ = dataset.load_mask(i)
            # convert pixel values (e.g. center)
            scaled_image = mold_image(image, cfg)
            # convert image into one sample
            sample = expand_dims(scaled_image, 0)
            # make prediction
            yhat = mask_model.detect(sample, verbose=0)[0]
            # define subplot
            pyplot.subplot(n_images, 2, i * 2 + 1)
            # plot raw pixel data
            pyplot.imshow(image)
            pyplot.title('Actual')
            # plot masks
            for j in range(mask.shape[2]):
                pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.1)
            # get the context for drawing boxes
            pyplot.subplot(n_images, 2, i * 2 + 2)
            # plot raw pixel data
            pyplot.imshow(image)
            pyplot.title('Predicted')
            ax = pyplot.gca()
            # plot each box
            for box in yhat['rois']:
                # get coordinates
                y1, x1, y2, x2 = box
                # calculate width and height of the box
                width, height = x2 - x1, y2 - y1
                # create the shape
                rect = Rectangle((x1, y1), width, height, fill=False, color='red')
                # draw the box
                ax.add_patch(rect)
        # Производится отрисовка результата
        pyplot.show()

    @staticmethod
    def process_image(dataset: Dataset, mask_model: MaskRCNN) -> NoReturn:
        """
        Служит для запуска поиска объектов на изображении
        :param dataset: Набор данных
        :param mask_model: Модель
        :return:
        """
        # Производится загрузка фото
        img = load_img('data/val/toronto3_h_1_w_8.png')
        img = img_to_array(img)
        # Запуск поиска
        results = mask_model.detect([img], verbose=0)
        # Получаем данные по предсказанию
        r = results[0]
        # Производит отрисовку фотографии с маской и вероятностью
        display_instances(img, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])
