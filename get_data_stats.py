import os
import json
from typing import List, Dict
from dataclasses import dataclass

ROOT_DIR = os.path.abspath("./")
LOG_DIR = os.path.join(ROOT_DIR, "data")


@dataclass
class CPoint(object):
    x: int = None
    y: int = None


@dataclass
class CAnnotation(object):
    id: str = None
    name: str = None
    points: List[CPoint] = None

    def __init__(self, id: str = None, name: str = None, points: List[CPoint] = None):
        self.id = id
        self.name = name
        if points is None:
            self.points = list()
        else:
            self.points = points


@dataclass
class CImage(object):
    id: str = None
    filename: str = None
    annotations: List[CAnnotation] = None


def main(filename: str):
    counter: Dict[str, int] = dict()
    labels: Dict[str, str] = dict()
    images: List[CImage] = list()

    print(f'trying open file:{filename}')
    with open(filename, 'r') as f:
        json_obj = json.load(f)
        for key, value in json_obj['_via_attributes']['region']['food']['options'].items():
            labels[key] = value
        img_metadata = json_obj['_via_img_metadata']
        for i, key in enumerate(img_metadata):
            img_obj = img_metadata[key]
            annotations: List[CAnnotation] = list()
            image = CImage(id=str(i), filename=img_obj['filename'], annotations=annotations)
            for region in img_obj['regions']:
                annotation = CAnnotation()
                image.annotations.append(annotation)
                shape_attributes = region['shape_attributes']
                annotation.id = region['region_attributes']['food']
                annotation.name = labels[annotation.id]
                if annotation.id not in counter:
                    counter[annotation.id] = 0
                counter[annotation.id] += 1
                for j, attr_x in enumerate(shape_attributes['all_points_x']):
                    point = CPoint(x=int(attr_x), y=int(shape_attributes['all_points_y'][j]))
                    annotation.points.append(point)
            images.append(image)

    for key, val in sorted(counter.items(), key=lambda item: item[1], reverse=True):
        print(f'"{labels[key]}";{key};{val};')


if __name__ == '__main__':
    main(f'{ROOT_DIR}/via_projects/ViaProjectVal3009c.json')
