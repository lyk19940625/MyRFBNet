# --------------------------------------------------------
# aurthor wanglei
# update time:2018-9-12
# --------------------------------------------------------

from __future__ import print_function

import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import VOC_300, VOC_512
from layers.functions import Detect, PriorBox
from models.RFB_Net_vgg import build_net
from utils.nms_wrapper import nms
from weights import trained_model

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128),
          (0, 255, 0, 128),
          (0, 0, 255, 128),
          (0, 255, 255, 128),
          (255, 0, 255, 128),
          (255, 255, 0, 128))

# data set classes name
LABELS_SET = ('__background__', 'rusty', 'nest', 'polebroken',
              'poletopleaky', 'poleleakssteel', 'pdz')


# transform image for net input
class BaseTransform(object):
    def __init__(self, img_size, rgb_mean=(104, 117, 123)):
        self.rgb_mean = rgb_mean
        self.img_size = img_size

    # assume input is cv2 img for now
    def __call__(self, img):
        img = cv2.resize(np.array(img), (self.img_size, self.img_size)).astype(np.float32)
        if self.rgb_mean is not None:
            img -= self.rgb_mean
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)


# image object detector
class ObjDetector(object):
    def __init__(self, img_size=300, thresh=0.56):
        assert img_size == 300 or img_size == 512, 'net input image size must be 300 or 512'
        self.labels_name = LABELS_SET
        self.labels_numb = len(LABELS_SET)
        self.img_size = img_size
        self.cfg = VOC_300 if img_size == 300 else VOC_512
        self.thresh = thresh
        self.gpu_is_available = torch.cuda.is_available()
        self.gpu_numb = torch.cuda.device_count()
        self.net = build_net('test', self.img_size, self.labels_numb)
        self.detect = Detect(self.labels_numb, 0, self.cfg)
        self.transform = BaseTransform(self.img_size)

        # load net weights
        state_dict = torch.load(trained_model,map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        self.net.eval()
        print('Finished loading model!')

        if self.gpu_numb > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=list(range(self.gpu_numb)))

        # set net gpu or cpu model
        if self.gpu_is_available:
            self.net.cuda()
            cudnn.benchmark = True

        # define box generator
        priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = priorbox.forward()
            if self.gpu_is_available:
                self.priors = self.priors.cuda()

    def __net__(self, img):
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0)
            if self.gpu_is_available:
                x = x.cuda()
                scale = scale.cuda()

        # get net output
        out = self.net(x)
        boxes, scores = self.detect.forward(out, self.priors)
        boxes = boxes[0]
        scores = scores[0]

        # scale each detection back up to the image
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        return boxes, scores

    def __call__(self, image):
        """
        :param image: rgb image
        :return: {'label_name':[x1,y1,x2,y2,score],...}
        """
        boxes = np.empty((0, 4))
        scores = np.empty((0, self.labels_numb))

        for img, p in self.__chips__(image):
            b = [p[0], p[1], p[0], p[1]]
            boxes_t, scores_t = self.__net__(img)
            boxes_t += list(map(float, b))
            boxes = np.vstack((boxes, boxes_t))
            scores = np.vstack((scores, scores_t))

        # filter bounding boxes
        results = dict()
        for j in range(1, self.labels_numb):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            keeped = nms(c_dets, 0.45, force_cpu=0)
            c_dets = c_dets[keeped, :]
            results[self.labels_name[j]] = c_dets
        return results

    def __chips__(self, image):
        h, w, _ = image.shape
        x = w // 2
        y = h // 2
        boxes = []
        if min(h, w) > 1500:
            boxes.append((0, 0, x, y))
            boxes.append((x, 0, w, y))
            boxes.append((0, y, x, h))
            boxes.append((x, y, w, h))
            boxes.append((x // 2, y // 2, x + x // 2, y + y // 2))
        else:
            boxes.append((0, 0, w, h))
        for p in boxes:
            yield image[p[1]:p[3], p[0]:p[2]], p

    def draw(self, image, results):
        # draw bounding boxes
        for label, boxes in results.items():
            for value in boxes:
                x1 = int(value[0])
                y1 = int(value[1])
                x2 = int(value[2])
                y2 = int(value[3])
                # label name and scores
                text = label + ',' + "%.2f" % value[4]
                # select color
                indx = self.labels_name.index(label) % len(COLORS)
                color = COLORS[indx]
                # draw bounding boxe
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                # draw label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.58
                size = cv2.getTextSize(text, font, font_scale, 1)
                # text_w = size[0][0]
                text_h = size[0][1]
                cv2.putText(image, text, (x1, max((y1 - text_h), 0)), font, font_scale, color, 1)
        return image


def main():
    detector = ObjDetector()
    for file in os.listdir('./images'):
        print(file)
        img = cv2.imread('./images/' + file, cv2.IMREAD_COLOR)
        rest = detector(img)
        img_out = detector.draw(img, rest)
        savefile = './images/test_' + file
        cv2.imwrite(savefile, img_out)


if __name__ == '__main__':
    main()
