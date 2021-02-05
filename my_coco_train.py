import sys
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from model.config import DetectConfig, EPOCHS, MyCocoDetectConfig
from helper.rcnn_lib import add_mask_rcnn
from dataset.dataset_factory import VGGCocoSetFactory
# Добавляется в поиск библиотека MaskRCNN
add_mask_rcnn(sys.argv[0])
from mask_rcnn.mrcnn.model import MaskRCNN
import imgaug


START_MODEL_PATH = 'start_mask/mask_rcnn_start.h5'
# START_MODEL_PATH = 'mask_rcnn_detect_cfg_0005.h5'
COCO_PATH = '/home/ids/my_coco'
CLASS_IDS = [1]

# Определяются параметры использования видеокарты
config_proto = ConfigProto()
# config_proto.gpu_options.per_process_gpu_memory_fraction = 0.8
config_proto.gpu_options.allow_growth = True
session = InteractiveSession(config=config_proto)

# Подготавливаем тренировочный и тестовый набор данных
# train_set = DataSetFactory.new_instance('data/img')
# test_set = DataSetFactory.new_instance('data/val')
train_set = VGGCocoSetFactory.new_instance(COCO_PATH, 'train', 3001, CLASS_IDS)
test_set = VGGCocoSetFactory.new_instance(COCO_PATH, 'val', 3001, CLASS_IDS)

# Определяется конфигурация
config = MyCocoDetectConfig()
config.display()
# Определяется модель
model = MaskRCNN(mode='training', model_dir='./', config=config)
# Произвести загрузку стартовой модели
model.load_weights(
    START_MODEL_PATH,
    by_name=True,
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
    # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
)

augmentation = imgaug.augmenters.Sequential(
    [
        imgaug.augmenters.Fliplr(0.5),
        imgaug.augmenters.Rotate((-45, 45))
    ]
)
# Запуск тренировки тестовой модели
model.train(
    train_set,
    test_set,
    learning_rate=config.LEARNING_RATE,
    epochs=100,
    layers='heads',
    augmentation=augmentation
)
