"""myself Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: wanglei
"""

import os
import os.path
import pickle
import sys
from os import listdir
from os.path import isfile, join
from random import shuffle

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .voc_eval import voc_eval

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

SDYY_CLASSES = ('__background__', 'rusty', 'bolt', 'poleLeaksSteel', 'poleTopLeaky', 'pdzLost', 'pdz', 'nest')

class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
        keep_difficult (bool, optional): keep difficult instances or not
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(zip(SDYY_CLASSES, range(len(SDYY_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue

            name = obj.find('name').text.strip()
            if name not in self.class_to_ind.keys():
                continue

            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class SDYYDetection(data.Dataset):
    """input is image, target is annotation
    Arguments:
        root (string): filepath to youself dataset folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the input image
        target_transform (callable, optional): transformation to perform on the  target `annotation`
        dataset_name (string, optional): which dataset to load
    """

    def __init__(self, root, image_sets, preproc=None, target_transform=None, dataset_name='SDYY'):
        self.root = root
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for name in image_sets:
            for line in open(os.path.join(self.root, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((self.root, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)
            # target = self.target_transform(target, width, height)
        # print(target.shape)
        return img, target

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.
        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_results_file(all_boxes)
        self._do_python_eval(output_dir)

    def _get_results_file_template(self):
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(self.root, 'results', 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(SDYY_CLASSES):
            cls_ind = cls_ind
            if cls == '__background__':
                continue
            print('Writing {} results file'.format(cls))
            filename = self._get_results_file_template().format(cls)
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
        rootpath = self.root
        name = self.image_set[0]
        annopath = os.path.join(
            rootpath,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            rootpath,
            'ImageSets',
            'Main',
            name + '.txt')
        cachedir = os.path.join(self.root, 'annotations_cache')
        aps = []

        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for i, cls in enumerate(SDYY_CLASSES):
            if cls == '__background__':
                continue
            filename = self._get_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
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


# create yourself image set
def SDYYImageSets(root='E:/image_data/SDYY', min_sample_size=10):
    annopath = os.path.join(root, 'Annotations')
    setspath = os.path.join(root, 'ImageSets', 'Main')
    if not os.path.exists(setspath):
        os.makedirs(setspath)
    xmlfiles = [f for f in listdir(annopath) if isfile(join(annopath, f))]
    shuffle(xmlfiles)

    temp_list = [[] for _ in range(len(SDYY_CLASSES) - 1)]
    label_img = dict(zip(SDYY_CLASSES[1:], temp_list))
    for file in xmlfiles:
        img_id = file.split('.')[0]
        path = os.path.join(root, 'Annotations', file)
        print(path)
        target = ET.parse(path).getroot()
        name_set = []
        for obj in target.iter('object'):
            name = obj.find('name').text.strip()
            name_set.append(name)
        # delete the same name
        name_dif = {}.fromkeys(name_set).keys()
        # save label
        for label in list(name_dif):
            label_img[label].append(img_id)
            txtfile = label + '.txt'
            with open(os.path.join(setspath, txtfile), 'a') as f:
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
    with open(os.path.join(setspath, 'names.txt'), 'w') as f:
        for label in names:
            f.write(label + '\n')

    # save trainval
    with open(os.path.join(setspath, 'trainval.txt'), 'w') as f:
        for img_id in trainval:
            f.write(img_id + '\n')

    # save test
    with open(os.path.join(setspath, 'test.txt'), 'w') as f:
        for img_id in test:
            f.write(img_id + '\n')

    # save all
    with open(os.path.join(setspath, 'main.txt'), 'w') as f:
        for img_id in (test + trainval):
            f.write(img_id + '\n')

    # save data report
    with open(os.path.join(setspath, 'data_report.txt'), 'w') as f:
        for label, images in label_img.items():
            img_size = str(len(images))
            f.write(label + ":" + img_size + '\n')


if __name__ == '__main__':
    SDYYImageSets()
