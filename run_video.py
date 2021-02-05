import datetime
import sys
from model.config import PredictionConfig, define_video_config
from helper.rcnn_lib import add_mask_rcnn
# Добавляется в поиск библиотека MaskRCNN
add_mask_rcnn(sys.argv[0])
from mask_rcnn.mrcnn.model import MaskRCNN
from helper.video import VideoHelper

# WEIGHT_FILE = 'detect_cfg20200413T1725/mask_rcnn_detect_cfg_0016.h5'
# WEIGHT_FILE = 'detect_cfg20200513T1306/mask_rcnn_detect_cfg_0080.h5'
# WEIGHT_FILE = 'mask_rcnn_detect_cfg_0082.h5'
# WEIGHT_FILE = 'detect_cfg20200618T1641/mask_rcnn_detect_cfg_0052.h5'
WEIGHT_FILE = 'detect_cfg20200618T1641/mask_rcnn_detect_cfg_0099.h5'


# VIDEO_PATH = 'flights1.mov'
# VIDEO_PATH = 'test/test.mp4'
# VIDEO_PATH = 'video/профессия нефтяник.mp4'
# VIDEO_PATH = 'video/test.mp4'
# VIDEO_PATH = 'main_test_video.mp4'
VIDEO_PATH = 'test_video/6.mp4'
OUTPUT_FILE = "result/result_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())

# Производится определение параметров использования видеокарты
define_video_config()

# Подготавливается конфигурация
config = PredictionConfig()
config.display()

# Производится определение модели
model = MaskRCNN(mode='inference', model_dir='./', config=config)
# Производится загрузка весов
model.load_weights(
    WEIGHT_FILE,
    by_name=True
)

# Определяется помошник для работы с видео
video_helper = VideoHelper(model, VIDEO_PATH, OUTPUT_FILE)
# Производится запуск
video_helper.run()
