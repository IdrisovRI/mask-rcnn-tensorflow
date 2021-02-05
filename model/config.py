import sys
import numpy
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from helper.rcnn_lib import add_mask_rcnn
# Добавляется в поиск библиотека MaskRCNN
add_mask_rcnn(sys.argv[0])
from mask_rcnn.mrcnn.config import Config

STEPS = 32000
EPOCHS = 10
OBJECTS = {
    1: '21_bashkir_synabon'
}
COLORS = numpy.random.randint(0, 255, size=(5, 3), dtype="uint8")


# def define_video_config():
#     """
#     Определяются параметры использования видеокарты
#     """
#     config = ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = 0.8
#     config.gpu_options.allow_growth = True
#     InteractiveSession(config=config)


class BaseConfig(Config):
    # GPU_COUNT = 1
    # Число объектов (background + OBJECTS)
    NUM_CLASSES = 1 + 1
    # Число шагов в эпохе
    STEPS_PER_EPOCH = STEPS
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    # USE_MINI_MASK = False

    # RPN_ANCHOR_SCALES = (16, 32, 64, 96, 128)
    # DETECTION_MIN_CONFIDENCE = 0.95
    # DETECTION_NMS_THRESHOLD = 0.0
    # DETECTION_MAX_INSTANCES = 100

    # MINI_MASK_SHAPE = (1920, 1920)
    # IMAGE_RESIZE_MODE = "pad64"


class DetectConfig(BaseConfig):
    # define the name of the configuration
    NAME = "detect_cfg"


class MyCocoDetectConfig(BaseConfig):
    # define the name of the configuration
    NAME = "detect_cfg"
    STEPS_PER_EPOCH = 55


class PredictionConfig(BaseConfig):
    NAME = "predict_cfg"
    IMAGE_MAX_DIM = 1920
    # DETECTION_NMS_THRESHOLD = 0.3
    # DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_MAX_INSTANCES = 200


class PredictionConfig2K(BaseConfig):
    NAME = "predict_cfg_2k"
    # IMAGE_MIN_DIM = 1080
    # IMAGE_MAX_DIM = 1920


def define_video_config():
    """
    Определяются параметры использования видеокарты
    """
    # Определяются параметры использования видеокарты
    config_proto = ConfigProto()
    # config_proto.gpu_options.per_process_gpu_memory_fraction = 0.8
    config_proto.gpu_options.allow_growth = True
    session = InteractiveSession(config=config_proto)
