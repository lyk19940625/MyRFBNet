# --------------------------------------------------------
# VOC Dataset
# Original author: Francisco Massa
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# updated by: wang lei
# update time:2018-9-12
# --------------------------------------------------------


import os
import os.path
import pickle
import sys

import cv2
import numpy as np
import torch
import torch.utils.data as data

from .voc_eval import voc_eval

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def Annotation(xml_file, classes, keep_difficult=True):
    """
    :param xml_file:
    :param classes:
    :param keep_difficult:
    :return: ground truth [[xmin,ymin.xman,ymax,label_id]...]
    """
    target = ET.parse(xml_file).getroot()
    class_to_ind = dict(zip(classes, range(len(classes))))
    bboxes = np.empty((0, 5))
    for obj in target.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if not keep_difficult and difficult:
            continue

        # parse classes name
        name = obj.find('name').text.lower().strip()
        if name not in classes:
            continue

        # parse bounding boxes
        bndbox = obj.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        box = []
        for pt in pts:
            cur_pt = int(bndbox.find(pt).text) - 1
            box.append(cur_pt)
        box.append(class_to_ind[name])
        # [[xmin, ymin, xmax, ymax, label_ind], ... ]
        bboxes = np.vstack((bboxes, box))
    return bboxes


class VOCDetection(data.Dataset):
    def __init__(self, voc_root, data_set, classes, augment, keep_difficult=True):
        """
        :param voc_root: VOCdevkit path
        :param data_set: such as (2012,trainval)
        :param classes: classes names
        :param augment: image augment
        :param keep_difficult: bool
        """
        self.voc_root = voc_root
        self.data_set = data_set
        self.classes = classes
        self.augment = augment
        self.keep_difficult = keep_difficult
        self.xml_path = os.path.join('%s', 'Annotations', '%s.xml')
        self.img_path = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.set_path = os.path.join(self.voc_root, 'VOC' + self.data_set[0])
        self.set_name = os.path.join(self.set_path, 'ImageSets', 'Main', self.data_set[1] + '.txt')
        with open(self.set_name, 'r') as f:
            for line in f:
                self.ids.append((self.set_path, line.strip()))

    def __getitem__(self, index):
        id = self.ids[index]
        boxes = Annotation(self.xml_path % id, self.classes, self.keep_difficult)
        image = cv2.imread(self.img_path % id, cv2.IMREAD_COLOR)
        if self.augment is not None:
            image, boxes = self.augment(image, boxes)
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image)
        return tensor, boxes

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        """
        Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            PIL img
        """
        img_id = self.ids[index]
        return cv2.imread(self.img_path % img_id, cv2.IMREAD_COLOR)

    def img_tensor(self, image):
        """
        :param image: change image to pytorch tensor
        :return: tensor
        """
        if self.augment is not None:
            image, _ = self.augment(image)
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image)
        return tensor

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.
        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)

    def _get_voc_results_file_template(self):
        """
         :return: file name to save voc result
        """
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(self.voc_root, 'results', 'VOC' + self.data_set[0], 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        """
        :param all_boxes: write result to voc file
        :return:
        """
        for cls_ind, cls in enumerate(self.classes):
            cls_ind = cls_ind
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        """
        :param output_dir: start evaluate net and report result
        :return:
        """
        annopath = os.path.join(self.set_path, 'Annotations', '{:s}.xml')
        cachedir = os.path.join(self.voc_root, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if self.data_set[0] is '2007' else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename,
                                     annopath,
                                     self.set_name,
                                     cls,
                                     cachedir,
                                     ovthresh=0.5,
                                     use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
