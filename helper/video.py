import os
import cv2
import sys
from model.config import OBJECTS, COLORS
from helper.rcnn_lib import add_mask_rcnn

# Добавляется в поиск библиотека MaskRCNN
add_mask_rcnn(sys.argv[0])
from mrcnn.model import MaskRCNN
from mrcnn.visualize import apply_mask


class VideoHelper(object):
    def __init__(self, mask_model: MaskRCNN, video_path: str, output: str):
        self.__model = mask_model
        self.__video_path = video_path
        self.__output_file = output

    def run(self):
        if not os.path.exists(self.__video_path):
            raise FileNotFoundError(f"Файл не существует! Неверно указан путь к видеофайлу:{self.__video_path}")

        capture = cv2.VideoCapture(self.__video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        print(f'Source file FPS:{capture.get(cv2.CAP_PROP_FPS)}"')
        # Define codec and create video writer
        writer = cv2.VideoWriter(self.__output_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

        count = 0
        success = True
        while success:
            print("Frame number: ", count)
            # Производится считывание нового изображения
            success, image = capture.read()
            if success:
                r = self.__model.detect([image], verbose=0)[0]
                boxes = r['rois']
                # Если на кадре найдены объекты, то проводим их визуализацию
                if len(r['class_ids']):
                    for num, object_id in enumerate(r['class_ids']):
                        # Извлекаются данные о точности распознавания объекта
                        score = r['scores'][num]
                        # Извлекаются параметры найденного объекта
                        box = boxes[num]
                        start_x = box[1]
                        start_y = box[0]
                        end_x = box[3]
                        end_y = box[2]
                        # Определяется цвет закраски
                        color = [int(c) for c in COLORS[object_id]]
                        # Отрисовывается прямоугольник на изображении
                        cv2.rectangle(
                            image,
                            (start_x, start_y),
                            (end_x, end_y),
                            color,
                            2
                        )

                        mask = r['masks'][:, :, num]
                        image = apply_mask(image, mask, color)

                        # Определяется маркировка найденного объекта
                        label = f"{OBJECTS[object_id]}:{round(score * 100, 2)}"
                        # Добавляем маркировку объекта к изображению
                        cv2.putText(image, label, (box[1], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.75 / 2, (0, 0, 0), 1)
                writer.write(image)
                count += 1
        writer.release()
