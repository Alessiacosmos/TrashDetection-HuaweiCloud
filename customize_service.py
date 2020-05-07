# -*- encoding: utf-8 -*-
"""
@File    : customize_service.py
@Time    : 2020/5/5 15:32
@Author  : Alessia K
@Email   : ------
"""

from PIL import Image
import logging as log
from model_service.pytorch_model_service import PTServingBaseService
from metric.metrics_manager import MetricsManager
import torch.nn.functional as F

import torch.nn as nn
import torch
import json
import numpy as np
import torchvision
import time
import os
import copy

import sys
import cv2

import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from YourModelDict import model
from save_json import label_list, create_class_dict, get_classes_name, save_result_as_json

print('CUDA available: {}'.format(torch.cuda.is_available()))


logger = log.getLogger(__name__)

IMAGES_KEY = 'images'
MODEL_INPUT_KEY = 'images'



def decode_image(file_content):
    """
    Decode bytes to a single image
    :param file_content: bytes
    :return: ndarray with rank=3
    """
    image = Image.open(file_content)
    image = image.convert('RGB')
    # print(image.shape)
    image = np.asarray(image, dtype=np.float32)
    return image/255.

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image = sample['img']
        img_name = sample['img_name']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))),
                           interpolation=cv2.INTER_LINEAR)

        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        # new_image = np.full((rows + pad_w, cols + pad_h, cns), 114).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        return {'img': torch.from_numpy(new_image), 'img_name': img_name, 'scale': scale}


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image = sample['img']
        img_name=sample['img_name']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'img_name': img_name}


class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PTVisionService, self).__init__(model_name, model_path)

        self.dir_path = os.path.dirname(os.path.realpath(model_path))
        # load label name
        self.label = create_class_dict(os.path.join(self.dir_path, 'data/'))
        self.num_class = len(self.label)
        self.transform = transforms.Compose([Normalizer(), Resizer()])

        # Load your model
        self.model = YourNet(model_path, self.num_class)

    def _preprocess(self, data):

        preprocessed_data = {}

        pre_st = time.time()

        for k, v in data.items():
            for file_name, file_content in v.items():
                # print('\tAppending image: %s' % file_name)
                image1 = decode_image(file_content)
                sample = {'img': image1, 'img_name': file_name}
                sample = self.transform(sample)
                preprocessed_data[k] = sample

        pre_et = time.time()
        self.pre_time = pre_st-pre_et
        return preprocessed_data

    def _inference(self, data):

        sample = data[IMAGES_KEY]  # img, img_name, scale

        img = sample['img']
        # print(img.size())  # torch.Size([832, 640, 3])
        width = img.shape[0]
        height = img.shape[1]

        padded_imgs = torch.zeros(1, width, height, 3)
        padded_imgs[0, :width, :height, :] = img
        padded_imgs = padded_imgs.permute(0, 3, 1, 2)
        # print(padded_imgs.size())  # torch.Size([1, 3, 832, 640])

        # grid = torchvision.utils.make_grid(padded_imgs, nrow=1, padding=2, normalize=True)
        # ndarr = grid.mul(255).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
        # b, g, r = cv2.split(ndarr)
        #
        # ndarr = cv2.merge([r, g, b])
        # cv2.imshow('1', ndarr)
        # cv2.waitKey()

        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, predict_bboxes = self.model(padded_imgs.cuda().float())
            else:
                scores, classification, predict_bboxes = self.model(padded_imgs.float())
            et = time.time()
            # print('Elapsed time: {} or {}(et-st)'.format(time.time() - st, et - st))
            result_i = {'img_name': sample['img_name'], 'scale': sample['scale'],
                        'class': classification, 'pred_boxes': predict_bboxes,
                        'score': scores, 'time': et - st}

        return result_i

    def _postprocess(self, data):
        '''

        :param data:
            result_i = {'img_name': sample['img_name'], 'scale': sample['scale'],
                        'class': classification, 'pred_boxes': predict_bboxes,
                        'score': scores, 'time': et - st}

            img_name   : string         eg: "imgname.jpg"
            scale      : float          eg: 0.8 (used to resize pred_bboxes to fit original image size)
            class      : Tensor [N]     predicted classification result
            pred_boxes : Tensor [N,4]   predicted bboxes result
            score      : Tensor [N]     scores of every bbox
            time       : float          codes runtime
        :return:
            "detection_classes":    []
            "detection_scores":     []  (.4f)
            "detection_bboxes":     []  (xmin、ymin、xmax、ymax) (.1f)
            "latency_time":         ""  (str(.1f))
        '''
        class_name = self.label
        labels = label_list(os.path.join(self.dir_path, 'data/class_name.csv'))  # ('data/class_name.csv')#

        post_st = time.time()
        result = data
        img_name = result['img_name']
        scale = result['scale']
        classification = result['class']
        predict_bboxes = result['pred_boxes']
        scores = result['score']
        lantecy_time = result['time']

        idxs = np.where(scores.cpu() > 0.45)

        detection_bboxes = []
        detection_classes = []
        for j in range(idxs[0].shape[0]):
            bbox = predict_bboxes[idxs[0][j], :]
            label_name = labels[int(classification[idxs[0][j]])]

            bbox /= scale
            classes_name = get_classes_name(class_name, label_name)

            detection_classes.append(classes_name)
            detection_bboxes.append(np.array(bbox))
        post_et = time.time()
        self.post_time = post_st - post_et
        all_run_time = lantecy_time + self.pre_time + self.post_time
        all_run_time *=1000     # ms
        json_file = save_result_as_json(img_name, detection_classes, np.array(scores.cpu()[idxs]),
                                        np.array(detection_bboxes), all_run_time)

        return json_file

    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : map of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')

        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)

        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)

        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = str(round(pre_time_in_ms + infer_in_ms + post_time_in_ms, 1)) + ' ms'
        return data



def YourNet(model_path, num_classes=44):
    # 生成网络
    yournetname = model.yournetclass(num_classes=num_classes, pretrained=False)
    # 加载模型
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    yournetname.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        yournetname = yournetname.cuda()


    if torch.cuda.is_available():
        yournetname = torch.nn.DataParallel(yournetname).cuda()
    else:
        yournetname = torch.nn.DataParallel(yournetname)

    yournetname.eval()

    return yournetname

