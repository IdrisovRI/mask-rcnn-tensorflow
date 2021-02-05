import os
import sys
from typing import NoReturn


def add_mask_rcnn(file) -> NoReturn:
    """
    Служит для добавления библиотеки mask_rcnn в поиск для импорта
    :param file: Местонахождение текущего файла
    """
    # Определяется путь к текущей папке с исходниками
    exec_path = os.path.dirname(file)
    # Формируется путь к библиотеке Mask RCNN
    mask_abspath = os.path.abspath(os.path.join(exec_path, 'mask_rcnn'))
    # Добавляем каталог в путь для поиска
    sys.path.append(mask_abspath)
