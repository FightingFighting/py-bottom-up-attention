import os
import io
import json
import multiprocessing as mp

# import some common libraries
import cv2
import fire
import torch
import PIL.Image
import numpy as np
import tensorflow as tf
from torch import nn
import detectron2
from loguru import logger

from albumentations import (
    BboxParams,
    Crop,
    Compose,
    ShiftScaleRotate,
    RandomBrightness,
    RandomContrast,
    RandomScale,
    Rotate,
    HorizontalFlip,
    MedianBlur,
)

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

import inference


def get_box_feature(boxes, im):
    h, w, c = im.shape
    features = []
    for box in boxes:
        norm_x = box['xmin']
        norm_y = box['ymin']
        norm_w = box['xmax'] - box['xmin']
        norm_h = box['ymax'] - box['ymin']
        features.append([
            norm_x,
            norm_y,
            norm_x + norm_w,
            norm_y + norm_h,
            norm_w,
            norm_h,
        ])
    # (num_boxes, 6)
    return torch.Tensor(features)


class ImageProcessor:

    def __init__(self, output_path, output_queue=None, device='cuda'):
        self.output_path = output_path
        self.npz_path = os.path.join(self.output_path, 'npz')
        tf.io.gfile.makedirs(self.output_path)
        tf.io.gfile.makedirs(self.npz_path)

        self._device = device
        self._predictor = None
        self._vg_classes = None
        self._vg_attrs = None

        self._input_queue = mp.Queue()
        self._output_queue = mp.Queue() if output_queue is None else output_queue
        self._process = mp.Process(
            target=self.process_images,
            args=[self._input_queue, self._output_queue],
            daemon=True,
        )
        self._process.start()
    
    @property
    def predictor(self):
        if self._predictor is None:
            (self._predictor,
             self._vg_classes,
             self._vg_attrs) = inference.build_predictor(get_classes_map=True, device=self._device)
            self._predictor.model = self._predictor.model.to(self._device)
        return self._predictor
    
    def process_images(self, input_queue: mp.Queue, output_queue: mp.Queue):
        while True:
            img = input_queue.get()
            if img == '[STOP]':
                break
            try:
                with logger.catch(reraise=True, message=f'Fail with {img}'):
                    # raw_image = cv2.imread(img)
                    image_name = os.path.basename(img)
                    npz_path = os.path.join(
                        self.npz_path,
                        image_name.replace('.jpg', '.npz').replace('.png', '.npz')
                    )
                    if tf.io.gfile.exists(npz_path):
                        output_queue.put(image_name)
                        continue
                    
                    with tf.io.gfile.GFile(img, mode='rb') as imgf:
                        raw_image = tf.image.decode_image(imgf.read())
                        raw_image = cv2.cvtColor(raw_image.numpy(), cv2.COLOR_RGB2BGR)
                    
                    h, w, c = raw_image.shape
                    if h < 64 or w < 64 or h > 9600 or w > 9600:
                        raise ValueError(f'Size doesnt look right :( , {raw_image.shape}')
                    (instances, roi_features,
                     pred_class_probs, pred_attr_probs) = inference.doit_without_boxes(self.predictor, raw_image, full_pred=True)
                    predict = {
                        'image_name': image_name,
                        'height': int(instances.image_size[0]),
                        'width': int(instances.image_size[1]),
                        'boxes': instances.pred_boxes.tensor.cpu().tolist(),
                        'classes': [self._vg_classes[i] for i in instances.pred_classes.tolist()],
                        'attributes': [self._vg_attrs[i] for i in instances.attr_classes.tolist()],
                        'scores': instances.scores.tolist(),
                        'attr_scores': instances.attr_scores.tolist()
                    }

                    # HACK: GFile can't properly write file for now.
                    # with tf.io.gfile.GFile(npz_path, mode='wb+') as npf:
                    tmp_path = os.path.join(
                        os.path.dirname(__file__),
                        image_name.replace('.jpg', '.npz').replace('.png', '.npz')
                    )
                    np.savez(
                        tmp_path, predict=predict,
                        features=roi_features.cpu().numpy(),
                        pred_class_probs=pred_class_probs.cpu().numpy(),
                        pred_attr_probs=pred_attr_probs.cpu().numpy())
                    tf.io.gfile.copy(tmp_path, npz_path)
                    tf.io.gfile.remove(tmp_path)

                    output_queue.put(image_name)
            except:
                output_queue.put('[FAIL]')
    
    def process_image_folder(self, src_dir):
        img_list = tf.io.gfile.glob(os.path.join(src_dir, '*.jpg'))
        img_list += tf.io.gfile.glob(os.path.join(src_dir, '*.png'))
        for img in img_list:
            self._input_queue.put(img)
    
    def process_image_json(self, json_path):
        with tf.io.gfile.GFile(json_path, mode='r') as jf:
            img_list = json.load(jf)
        img_count = 0
        for img in img_list:
            self._input_queue.put(img)
            img_count += 1
        return img_count
    
    def process_single_image(self, image_path):
        self._input_queue.put(image_path)
    
    def process_image_list(self, image_paths):
        for image_path in image_paths:
            self._input_queue.put(image_path)


def load_json(json_path):
    with tf.io.gfile.GFile(json_path, mode='r') as jf:
        return json.load(jf)


def create_split_image_list_json(image_dir, n_split, output_dir, json_prefix):
    img_list = tf.io.gfile.glob(os.path.join(image_dir, '*.jpg'))
    img_list += tf.io.gfile.glob(os.path.join(image_dir, '*.png'))
    splits = [img_list[i::n_split] for i in range(n_split)]
    for i, split in enumerate(splits):
        json_name = f"{json_prefix}.{i}.json"
        json_path = os.path.join(output_dir, json_name)
        with tf.io.gfile.GFile(json_path, mode='w') as jf:
            json.dump(split, jf)


def run_processor(json_dir, output_dir, n_processor=4, num_gpu=4):
    output_queue = mp.Queue()
    processors = [
        ImageProcessor(output_dir, output_queue=output_queue, device=f"cuda:{i % num_gpu}")
        for i in range(n_processor)]
    
    json_list = tf.io.gfile.glob(os.path.join(json_dir, '*.json'))
    logger.warning(f'Find follow json files: {json_list}')
    img_count = 0

    img_list = []
    for i, js in enumerate(json_list):
        img_list += load_json(js)
    #     pid = i % n_processor
    #     logger.info(f"Assign {js} to [{pid}] processor")
    #     img_count += processors[pid].process_image_json(js)

    img_list_re_split = [img_list[i::n_processor] for i in range(n_processor)]
    for i, sp in enumerate(img_list_re_split):
        processors[i].process_image_list(sp)
        img_count += len(sp)
    
    for p in processors:
        p._input_queue.put('[STOP]')
    
    ret_count = 0
    fail_count = 0
    while True:
        ret = output_queue.get()
        ret_count += 1
        if ret == '[FAIL]':
            fail_count += 1
        logger.info(f"{ret_count}/{img_count} | {fail_count} - {ret}")


if __name__ == "__main__":
    with logger.catch(reraise=True):
        fire.Fire({
            'create_split_image_list_json': create_split_image_list_json,
            'process_jsons': run_processor,
        })