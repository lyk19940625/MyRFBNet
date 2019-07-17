# --------------------------------------------------------
# author:wang lei
# updated:2018-9-29 10:04:10
# reference:https://github.com/soumenpramanik/Convert-Pascal-VOC-to-COCO/blob/master/convertVOC2COCO.py
# --------------------------------------------------------

import datetime
import json
import os
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

TIMEFORMAT = '%Y-%m-%d %H:%M:%S'
theTime = datetime.datetime.now().strftime(TIMEFORMAT)
VOC_CLASSES = ('rusty', 'nest', 'bolt', 'poletopleaky', 'poleleakssteel', 'pdzlost', 'pdz')


class VOC2COCO(object):
    """
    change voc to coco format
    """

    def __init__(self, voc_root, data_set, save_path):
        self.voc_root = voc_root
        self.save_path = save_path
        self.class_to_id = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.anno_file = os.path.join(voc_root, 'Annotations', '%s.xml')
        self.save_file = os.path.join(save_path, data_set + '.json')
        self.data_path = os.path.join(voc_root, 'ImageSets', 'Main', data_set + '.txt')
        self.names = list()
        with open(self.data_path, 'r') as f:
            for line in f:
                self.names.append(line.strip())

        # coco template
        self.coco_anno = dict()
        self.info = dict()
        self.images = list()
        self.licenses = list()
        self.annotations = list()
        self.categories = list()

        # fill categories field
        for i, name in enumerate(VOC_CLASSES):
            categorie = dict()
            categorie["supercategory"] = name
            categorie["id"] = i
            categorie["name"] = name
            self.categories.append(categorie)

    def __call__(self):
        # fill info field
        self.info["description"] = "This is small object detection dataset."
        self.info["url"] = ""
        self.info["version"] = 1.0
        self.info['year'] = 2018
        self.info['contributor'] = "wang lei"
        self.info['date_created'] = str(theTime)

        # fill license field
        licenses = dict()
        licenses["url"] = ""
        licenses["id"] = 1
        licenses["name"] = "shandong evay info"
        self.licenses.append(licenses)

        for name in self.names:
            print(name)
            xml = self.anno_file % name
            imgid = int(name.split('_')[2])
            target = ET.parse(xml).getroot()
            filename = target.find('filename').text
            imgsize = target.find('size')
            height = imgsize.find("height").text
            width = imgsize.find("width").text

            # fill image field
            image = dict()
            image["license"] = 1
            image["file_name"] = filename
            image["coco_url"] = ""
            image["height"] = int(height)
            image["width"] = int(width)
            image["date_captured"] = str(theTime)
            image["flickr_url"] = ""
            image["id"] = imgid
            self.images.append(image)

            # fill annotation field
            for id, obj in enumerate(target.iter('object')):
                box = obj.find('bndbox')
                x1 = int(box.find('xmin').text)
                y1 = int(box.find('ymin').text)
                x2 = int(box.find('xmax').text)
                y2 = int(box.find('ymax').text)
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                class_name = obj.find('name').text.lower().strip()

                # fill annotation field
                annotation = dict()
                annotation["id"] = id
                annotation["image_id"] = imgid
                annotation["category_id"] = self.class_to_id[class_name]
                annotation["segmentation"] = None
                annotation["area"] = w * h
                annotation["bbox"] = [x1, y1, w, h]
                annotation["iscrowd"] = 0
                self.annotations.append(annotation)

        self.coco_anno['info '] = self.info
        self.coco_anno['images'] = self.images
        self.coco_anno['annotations'] = self.annotations
        self.coco_anno['categories'] = self.categories
        self.coco_anno['type'] = "instances"
        jsonString = json.dumps(self.coco_anno)
        with open(self.save_file, "w") as f:
            f.write(jsonString)


if __name__ == '__main__':
    voc_root = 'E:/image_data/VOCdevkit/VOCSDYY/'
    data_set = 'test'
    save_path = 'E:/coco/Annotations'
    transform = VOC2COCO(voc_root, data_set, save_path)
    transform()
