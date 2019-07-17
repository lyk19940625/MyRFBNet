# --------------------------------------------------------
# Evaluating the network
# Original author: Francisco Massa
# updated by: wanglei
# update time 2018-10-9
# --------------------------------------------------------


from __future__ import print_function

import argparse
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import VOCroot, VOCDetection, CLASSES, VOC_300, VOC_512, ImageAugment
from layers.functions import Detect, PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Receptive Field Block Net')
parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='SDYY',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='weights/Final_RFB_vgg_SDYY.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# start
cfg = (VOC_300, VOC_512)[args.size == '512']

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
else:
    print('Unkown version!')

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

# define test data set
img_size = (300, 512)[args.size == '512']
rgb_mean = (104, 117, 123)
img_set = ('SDYY', 'test')

transform = ImageAugment(img_size, rgb_mean)
testset = VOCDetection(VOCroot, img_set, CLASSES, transform)
num_classes = len(testset.classes)

# define net and load weight
net = build_net('test', img_size, num_classes)
state_dict = torch.load(args.trained_model)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v

# load new state dict
net.load_state_dict(new_state_dict)
net.eval()
print('Finished loading model!')
print(net)

# save net to cpu or gpu
if args.cuda:
    net = net.cuda()
    cudnn.benchmark = True
else:
    net = net.cpu()

top_k = 200
detector = Detect(num_classes, 0, cfg)
save_folder = os.path.join(args.save_folder, args.dataset)


def eval_net(save_folder, net, detector, cuda, testset, max_per_image=300, thresh=0.005):
    """
    :param save_folder: evaluate results save path
    :param net:trained net model
    :param detector: object detector function
    :param cuda:whether use cuda speed up
    :param testset:test data set
    :param max_per_image: max image size
    :param thresh:Confidence threshold
    :return:
    """
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    # num_classes = (21, 81)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return

    for i in range(num_images):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = testset.img_tensor(img).unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        # forward pass
        out = net(x)
        boxes, scores = detector.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                  .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    eval_net(save_folder, net, detector, args.cuda, testset, top_k, thresh=0.01)
