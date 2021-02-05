from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from mask_rcnn.mrcnn.model import MaskRCNN
from mask_rcnn.mrcnn.config import Config
import numpy
from dataclasses import dataclass
from typing import Optional, List


IMG_PATH = 'test_img.jpeg'
WEIGHT_FILE = 'mask_rcnn_detect_cfg_0073.h5'
OBJECTS = {
    1: '0_perzoll',
    2: '1_pasta_and_cheese',
    3: '2_rice_with_vegetables',
    4: '3_green_beans',
    5: '4_barley_porish',
    6: '5_pink_salmon_with_tomato_and_cheese',
    7: '6_offal_goulash',
    8: '7_salad_of_beetroot',
    9: '8_greek_salad',
    10: '9_crab_salad',
    11: '10_chiken_with_pineapples',
    12: '11_mens_dream',
    13: '12_olivie',
    14: '13_solyanka',
    15: '14_cabbage_soup',
    16: '15_rice_porridge',
    17: '16_beans',
    18: '17_grilled_beef_cutlet',
    19: '18_steam_cutlet',
    20: '19_french_meat',
    21: '20_meat_balls_in_sauce',
    22: '21_bashkir_synabon',
    23: '22_fruit_jellied_cheesecake',
    24: '23_cottage_cheese_casserole',
    25: '24_quiche_with_chiken_and_mushrooms',
    26: '25_kurnik',
    27: '26_sour_cream_bun',
    28: '27_sausage_in_the_dough',
    29: '28_poppy_snail',
    30: '29_uchpochmak',
    31: '30_french_cheesecake',
    32: '31_bread',
    33: '32_paella',
    34: '33_fried_humpback_salmon',
    35: '34_chicken_fillet_with_mushrooms',
    36: '35_florence_salad',
    37: '36_shchi_with_meatballs',
    38: '37_bun_with_cottage_cheese_and_sour_cream',
    39: '38_barley_with_vegetables',
    40: '39_chicken_fillet_with_cream',
    41: '40_chiken_grill_cuttlet',
    42: '41_boyar_chicken',
    43: '42_schnitzel_chicken',
    44: '43_white_crackers',
    45: '44_black_crackers',
    46: '45_buza',
    47: '46_fresh_berry_compote',
    48: '47_dried_fruits_compote',
    49: '48_buckthorn_drink',
    50: '49_currant_drink',
    51: '51_meatball_pickle',
    52: '52_elesh',
    53: '53_ideal_with_chicken',
    54: '54_stewed_cabbage',
    55: '57_lemon',
    56: '59_fried_sole',
    57: '62_sugar',
    58: '63_cream',
    59: '65_mushroom_soup_puree',
    60: '67_kharcho_with_meatballs',
    61: '68_tea_bag',
    62: '69_honey_cake',
    63: '73_lean_carrot_cookies',
    64: '74_sweet_gubadiya',
    65: '75_berry_cheesecake_with_filling',
    66: '76_strawberry_muffin',
    67: '79_buckwheat_soup_with_mushrooms',
    68: '80_chicken_noodle_soup',
    69: '81_ginger_puree_soup',
    70: '82_cabbage_with_celery_and_apple',
    71: '83_vitamin_salad',
    72: '84_georgian_salad',
    73: '85_vanilla_cheesecakes',
    74: '86_schnitzel_natural',
    75: '87_homemade_cutlets',
    76: '93_invigorating_drink',
    77: '94_broth_with_egg',
    78: '98_mashed_potatoes',
    79: '99_chicken_fillet_with_cheese_and_tomato',
    80: '100_chicken_cutlets_tenderness',
    81: '105_gourmet',
    82: '106_moscow_bun',
    83: '107_water_in_bottle_ara',
    84: '108_boiled_chicken',
    85: '109_couscous_with_mushrooms',
    86: '110_pizza_with_mushrooms',
    87: '111_pizza_with_sausage_and_ham',
    88: '112_garlic_pompushka',
    89: '116_fried_zucchini_in_egg',
    90: '117_mojito',
    91: '118_fried_chicken_legs',
    92: '119_chiken_kebab',
    93: '121_pilaf_with_beef',
    94: '122_rustic_potatoes',
    95: '123_steamed_vegetables',
    96: '124_crispy_chicken_fillet',
    97: '125_coffee',
    98: '128_sayke_with_an_apple',
    99: '129_salad_with_mushrooms_and_crab_sticks',
    100: '130_light_salad',
    101: '132_porridge_bulgur',
    102: '133_pink_salmon_under_the_marinade',
    103: '135_beetroot_with_prunes_and_nuts',
    104: '136_cheese',
    105: '137_butter',
    106: '138_millet_porridge',
    107: '140_fish_chips',
    108: '141_vegetable_stew',
    109: '142_beef_sausages',
    110: '144_quiche_with_pink_salmon',
    111: '145_sorrel_pie',
    112: '146_curd_snail',
    113: '158_buckwheat',
    114: '159_liver_cake',
    115: '160_boiling_water',
    116: '161_tea',
    117: '162_tartar_sauce',
    118: '163_okroshka',
    119: '164_sabantuy',
    120: '165_sausage',
    121: '166_karski_salad',
    122: '167_pinchetta',
    123: '168_anthill_cake',
    124: '169_baklava',
    125: '170_albanian_meat',
    126: '171_vegetable_mix',
    127: '172_carousel_salad',
    128: '173_broccoli_puree_soup',
    129: '174_rice_with_saffron'
}
COLORS = numpy.random.randint(0, 255, size=(255, 3), dtype="uint8")

CLASS_IDS = [
    2, 3, 4, 10,
    12, 18,
    24, 27,
    31, 32,
    41, 47, 48, 49,
    57,
    61, 68, 69,
    71, 78, 79,
    80, 83,
    90, 91, 94, 97,
    113, 115, 116, 118,
    128
]

# Определяются параметры использования видеокарты
config_proto = ConfigProto()
# config_proto.gpu_options.per_process_gpu_memory_fraction = 0.8
config_proto.gpu_options.allow_growth = True
session = InteractiveSession(config=config_proto)


@dataclass
class Box(object):
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class IdentifiedObject(object):
    class_id: int
    name: str
    mask: numpy.ndarray
    box: Optional[Box]
    score: float


class BaseConfig(Config):
    # GPU_COUNT = 1
    # Число объектов (background + OBJECTS)
    NUM_CLASSES = 1 + 129

    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    USE_MINI_MASK = False

    RPN_TRAIN_ANCHORS_PER_IMAGE = 320
    RPN_ANCHOR_SCALES = (16, 32, 64, 96, 128)


class PredictionConfig(BaseConfig):
    NAME = "predict_cfg"
    DETECTION_MAX_INSTANCES = 30
    NUM_CLASSES = 1 + 32


def process_img(image: numpy.ndarray):
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

    # Производим сопоставление имен идентификаторов(Поскольку обучаем на части объектов)
    objs_ids = dict()
    objs_names = dict()
    for i, class_id in enumerate(CLASS_IDS, start=1):
        # print(i)
        objs_ids[i] = int(OBJECTS[class_id].split('_')[0])
        objs_names[i] = OBJECTS[class_id]
    print(objs_ids)
    print(objs_names)

    identified_objects = list()
    r = model.detect([image], verbose=0)[0]

    # Если на кадре найдены объекты, то проводим их обработку
    if len(r['class_ids']):

        for num, object_id in enumerate(r['class_ids']):
            box = r['rois'][num]
            identified_object = IdentifiedObject(
                class_id=objs_ids[object_id],
                name=objs_names[object_id],
                mask=r['masks'][:, :, num],
                box=Box(box[1], box[0], box[3], box[2]),
                score=r['scores'][num]
            )
            print(identified_object.__dict__)
            identified_objects.append(identified_object)

    return identified_objects


if __name__ == '__main__':
    import cv2
    img = cv2.imread(IMG_PATH)
    process_img(img)
