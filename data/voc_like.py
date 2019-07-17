# --------------------------------------------------------
# create voc like data set
# Original author: Francisco Massa
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# updated by: wang lei
# update time 2018-10-9
# --------------------------------------------------------

import os
import os.path
import sys
from os import listdir
from os.path import isfile, join
from random import shuffle

import cv2

from utils import parse_xml, build_xml

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

CLASSES = ('__background__', 'rusty', 'nest', 'bolt', 'boltlost', 'boltloose', 'polebroken',
           'poletopleaky', 'poleleakssteel', 'pdzlost', 'pdz', 'signlost', 'sign', 'fxbw')

# define yourself data set root path
data_root = 'E:/image_data/VOCdevkit/VOCSDYY'
anns_file = os.path.join(data_root, 'Annotations', '%s' + '.xml')
imgs_file = os.path.join(data_root, 'JPEGImages', '%s' + '.jpg')
sets_path = os.path.join(data_root, 'ImageSets', 'Main')
imgs_path = os.path.join(data_root, 'JPEGImages')
anns_path = os.path.join(data_root, 'Annotations')


def chips(img):
    """ split image into five block
    :param img:
    :return: image chips
    """
    h, w, _ = img.shape
    x = w // 2
    y = h // 2
    box = []
    if x > 300 and y > 300:
        box.append((0, 0, x, y))
        box.append((x, 0, w, y))
        box.append((0, y, x, h))
        box.append((x, y, w, h))
        box.append((x // 2, y // 2, x + x // 2, y + y // 2))
    else:
        box.append((0, 0, w, h))
    for p in box:
        temp = img[p[1]:p[3], p[0]:p[2]]
        yield temp, p


def chip_image():
    """
    save the image block into local path
    :return: None
    """
    for file in os.listdir(imgs_path):
        name = file.split('.')[0]
        ann = anns_file % name
        if not os.path.isfile(ann):
            continue
        # print(ann)
        image = cv2.imread(imgs_file % name)
        # id = name.split('_')
        for i, (img, shape) in enumerate(chips(image)):
            obj_list = parse_xml(ann, shape)
            if len(obj_list) == 0:
                continue
            # index = int(id[1]) + i + 1
            # index = "%04d" % index
            index = "%04d" % i
            img_name = index + "_" + name  # id[0] + "_" + index + "_" + id[2]
            new_img = imgs_file % img_name
            new_ann = anns_file % img_name
            print(new_ann)
            build_xml(obj_list, img_name, new_img, img.shape, new_ann)
            cv2.imwrite(new_img, img)


def create_imageset(min_sample_size=100):
    """
     create yourself image set
    :param min_sample_size:
    :return: None
    """
    if not os.path.exists(sets_path):
        os.makedirs(sets_path)
    xmlfiles = [f for f in listdir(anns_path) if isfile(join(anns_path, f))]
    shuffle(xmlfiles)

    temp_list = [[] for _ in range(len(CLASSES) - 1)]
    label_img = dict(zip(CLASSES[1:], temp_list))
    for file in xmlfiles:
        img_id = file.split('.')[0]
        path = os.path.join(data_root, 'Annotations', file)
        # print(path)
        target = ET.parse(path).getroot()
        name_set = []
        for obj in target.iter('object'):
            name = obj.find('name').text.strip().lower()
            if name not in CLASSES:
                print('unknown class label', name, path)
                continue
            name_set.append(name)
        # delete the same name
        name_dif = {}.fromkeys(name_set).keys()
        # save label
        for label in list(name_dif):
            label_img[label].append(img_id)
            txtfile = label + '.txt'
            with open(os.path.join(sets_path, txtfile), 'a') as f:
                f.write(img_id + '\n')

    # split data to trainval and test
    trainval = []
    test = []
    names = []
    for label, images in label_img.items():
        size = len(images)
        if size < min_sample_size:
            continue
        test_size = int(size * 0.3)
        shuffle(images)
        trainval = trainval + images[test_size:]
        test = test + images[0:test_size]
        names.append(label)

    # save names
    with open(os.path.join(sets_path, 'names.txt'), 'w') as f:
        for label in names:
            f.write(label + '\n')

    # save trainval
    with open(os.path.join(sets_path, 'trainval.txt'), 'w') as f:
        for img_id in trainval:
            f.write(img_id + '\n')

    # save test
    with open(os.path.join(sets_path, 'test.txt'), 'w') as f:
        for img_id in test:
            f.write(img_id + '\n')

    # save all
    with open(os.path.join(sets_path, 'main.txt'), 'w') as f:
        for img_id in (test + trainval):
            f.write(img_id + '\n')

    # save data report
    with open(os.path.join(sets_path, 'data_report.txt'), 'w') as f:
        for label, images in label_img.items():
            img_size = str(len(images))
            f.write(label + ":" + img_size + '\n')


if __name__ == '__main__':
    # chip_image()
    create_imageset()
