from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mask_rcnn.mrcnn.config import Config
from mask_rcnn.mrcnn.model import MaskRCNN
from mask_rcnn.mrcnn.model import mold_image
from mask_rcnn.mrcnn.utils import Dataset
from model.config import DetectConfig
from dataset.dataset_factory import DataSetFactory, CocoSetFactory
from numpy import mean
from mask_rcnn.mrcnn.utils import compute_ap
from mask_rcnn.mrcnn.model import load_image_gt
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from datetime import datetime


COCO_PATH = '/home/ids/coco'
CLASS_IDS = [1]
weight_path = 'detect_cfg20200412T1222/mask_rcnn_detect_cfg_0005.h5'

# Определяются параметры использования видеокарты
config_proto = ConfigProto()
# config_proto.gpu_options.per_process_gpu_memory_fraction = 0.8
config_proto.gpu_options.allow_growth = True
session = InteractiveSession(config=config_proto)


# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # if image_id % 1000 == 1:
        #     print(image_id)
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP


# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
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
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        pyplot.subplot(n_images, 2, i * 2 + 1)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Actual')
        # plot masks
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
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
    # show the figure
    pyplot.show()


train_set = CocoSetFactory.new_instance(COCO_PATH, 'train', 2017, CLASS_IDS)
test_set = CocoSetFactory.new_instance(COCO_PATH, 'val', 2017, CLASS_IDS)

# create config
cfg = DetectConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights(weight_path, by_name=True)

# evaluate model on training dataset
print(f"Start evaluate model train :{datetime.now()}")
# train_mAP = evaluate_model(train_set, model, cfg)
# print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
print(f"Start evaluate model test:{datetime.now()}")
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)
print(f"End:{datetime.now()}")

# # plot predictions for train dataset
# plot_actual_vs_predicted(train_set, model, cfg)
# # plot predictions for test dataset
# plot_actual_vs_predicted(test_set, model, cfg)
