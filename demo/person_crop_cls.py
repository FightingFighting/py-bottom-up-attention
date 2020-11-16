import os
import io
import json
import glob

# import some common libraries
import numpy as np
import cv2
import torch
import detectron2
from PIL import Image
from torch import nn
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

NUM_OBJECTS = 18


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


"""
Load visual gnome labels
"""

# Load VG Classes
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'data/genome/1600-400-20')


vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

vg_attrs = []
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for object in f.readlines():
        vg_attrs.append(object.split(',')[0].lower().strip())


MetadataCatalog.get("vg").thing_classes = vg_classes
MetadataCatalog.get("vg").attr_classes = vg_attrs

"""
Load Fater R-CNN
"""

cfg = get_cfg()
cfg.merge_from_file(os.path.join(current_dir, "../configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml"))
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# VG Weight
# cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl"
cfg.MODEL.DEVICE = 'cuda'
predictor = DefaultPredictor(cfg)


def doit(raw_image, raw_boxes):
    # Process Boxes
    raw_boxes = Boxes(torch.from_numpy(raw_boxes))
    
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        
        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        #print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)
        boxes.tensor = boxes.tensor.to(cfg.MODEL.DEVICE)
        
        # ----
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        
        # Predict classes        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled) and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        pred_class_prob = nn.functional.softmax(pred_class_logits, -1)
        pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)
        
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        
        # Detectron2 Formatting (for visualization only)
        roi_features = feature_pooled
        instances = Instances(
            image_size=(raw_height, raw_width),
            pred_boxes=raw_boxes,
            scores=pred_scores,
            pred_classes=pred_classes,
            attr_scores = max_attr_prob,
            attr_classes = max_attr_label
        )
        
        return instances, roi_features


def doit_without_boxes(raw_image):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))
        
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Pooled features size:', feature_pooled.shape)
        
        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        
        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor    
        
        # NMS
        for nms_thresh in np.arange(0.4, 0.61, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:], 
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break
                
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label
        
        print(instances)
        
        return instances, roi_features


def apply_augs(im, boxes):
    data_dict = {
        'image': im,
        'bboxes': [
            [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
            for box in boxes
        ],
        'fake_label': [0] * len(boxes)
    }
    # print(data_dict['bboxes'])
    
    bbox_params = BboxParams(
        format='albumentations',
        min_area=0, 
        min_visibility=0.2,
        label_fields=['fake_label'])
    
    album_augs = [
        HorizontalFlip(p=0.5),
        # RandomBrightness(limit=0.3, p=0.5),
        # RandomContrast(limit=0.3, p=0.5),
        RandomScale(scale_limit=(-0.3, 0.0), p=0.3),
        # MedianBlur(blur_limit=5, p=0.3),
        # Rotate(limit=10, p=0.25),
    ]
    album_augs = Compose(album_augs, bbox_params=bbox_params)
    
    new_data_dict = album_augs(**data_dict)
    # print(new_data_dict['bboxes'])

    new_boxes = [
        {
            **boxes[i],
            'xmin': new_data_dict['bboxes'][i][0],
            'ymin': new_data_dict['bboxes'][i][1],
            'xmax': new_data_dict['bboxes'][i][2],
            'ymax': new_data_dict['bboxes'][i][3],
        }
        for i in range(len(boxes))
    ]
    return new_data_dict['image'], new_boxes


def crop_image_boxes(img, boxes):
    h, w = img.shape[:2]
    crops = []
    for box_norm in boxes:
        box = [
            int(box_norm['xmin'] * w),
            int(box_norm['ymin'] * h),
            int(box_norm['xmax'] * w),
            int(box_norm['ymax'] * h),
        ]
        crop = img[box[1]: box[3], box[0]: box[2], ...]
        crops.append(crop)
    return crops

def race_images(dataset_root, output_dir, augment=False):
    img_list = glob.glob(os.path.join(dataset_root, '**', '*.jpg'))
    img_list += glob.glob(os.path.join(dataset_root, '**', '*.png'))
    annotations = []
    name2feature = {}
    
    for i, img_path in enumerate(img_list):
        img_class = os.path.basename(os.path.dirname(img_path))
        img_name = os.path.basename(img_path)
        img_id = img_name.replace('.png', '').replace('.jpg', '')
        
        im = cv2.imread(img_path)
        if augment:
            im, boxes = apply_augs(im, boxes)
        h, w, c = im.shape

        instances, features = doit_without_boxes(im)
        pred_boxes = instances.pred_boxes.tensor.cpu()
        pred_boxes /= torch.Tensor([w, h, w, h])
        boxes = [
            {
                'xmin': box[0],
                'ymin': box[1],
                'xmax': box[2],
                'ymax': box[3],
            }
            for box in pred_boxes.tolist()]
        
        vg_class = instances.pred_classes.tolist()
        vg_class_name = [vg_classes[i] for i in instances.pred_classes.tolist()]
        vg_attr_class = instances.attr_classes.tolist()
        vg_attr_class_name = [vg_attrs[i] for i in instances.attr_classes.tolist()]
        vg_attr_score = instances.attr_scores.tolist()
        
        new_boxes = [{
            **box,
            'vg_class': vg_class[b],
            'vg_class_name': vg_class_name[b],
            'vg_attr_class': vg_attr_class[b],
            'vg_attr_class_name': vg_attr_class_name[b],
            'vg_attr_score': vg_attr_score[b],
        } for b, box in enumerate(boxes)]
        
        person_boxes = []
        person_features = []

        for box, feat in zip(new_boxes, features):
            print(box['vg_class_name'])
            if box['vg_class_name'] in ['woman', 'man', 'person']:
                person_boxes.append(box)
                person_features.append(feat)

        name2feature[img_name] = {
            'image_shape': [h, w, c],
            'image_class': img_class,
            'boxes_and_score': person_boxes,
            'features': person_features,
        }

        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_rgb_crops = crop_image_boxes(im_rgb, person_boxes)
        for j, crop in enumerate(im_rgb_crops):
            crop_name = f"{img_id}.{j}.jpg"
            os.makedirs(os.path.join(output_dir, img_class), exist_ok=True)
            out_path = os.path.join(output_dir, img_class, crop_name)
            Image.fromarray(crop).save(out_path)
        
        logger.info(f"{i} / {len(img_list)}, {im.shape}")
    
    pt_path = os.path.join(output_dir, 'det_and_feat.pt')
    torch.save(name2feature, pt_path)


if __name__ == "__main__":
    with logger.catch():
        race_images(
            '/home/ron/Downloads/hateful_meme_cache/JumpStory/race',
            '/home/ron/Downloads/hateful_meme_cache/JumpStory/crop',
            augment=False
        )