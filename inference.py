# %%

import os
import io
import json

# import some common libraries
import numpy as np
import cv2
import torch
import PIL.Image
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

NUM_OBJECTS = 36


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def build_predictor(get_classes_map=False, device='cuda'):
    current_dir = os.path.dirname(__file__)
    """
    Load visual gnome labels
    """

    # Load VG Classes
    data_path = os.path.join(current_dir, 'demo/data/genome/1600-400-20')

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
    cfg.merge_from_file(os.path.join(current_dir, "configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml"))
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    if get_classes_map:
        return predictor, vg_classes, vg_attrs
    else:
        return predictor


def doit_without_boxes(predictor, raw_image, full_pred=False):
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
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
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
        if full_pred:
            return instances, roi_features, pred_class_logits, pred_attr_logits
        else:
            return instances, roi_features


def extract_boxes_feature(predictor, raw_image, raw_boxes):
    # Process Boxes
    raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())
    
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
    
    # ----
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": raw_height, "width": raw_width}]
    images = predictor.model.preprocess_image(inputs)
    
    # Run Backbone Res1-Res4
    # import pdb; pdb.set_trace()
    features = predictor.model.backbone(images.tensor)
    
    # Run RoI head for each proposal (RoI Pooling + Res5)
    proposal_boxes = [boxes]
    features = [features[f] for f in predictor.model.roi_heads.in_features]
    box_features = predictor.model.roi_heads._shared_roi_transform(
        features, proposal_boxes
    )
    feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
    return feature_pooled


def extract_batch_boxes_feature(predictor, raw_images, raw_boxes_list):
    assert len(raw_images) == len(raw_boxes_list)
    
    # Preprocessing
    inputs = []
    for raw_image in raw_images:
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs.append({"image": image, "height": raw_image.shape[0], "width": raw_image.shape[1]})
    images = predictor.model.preprocess_image(inputs)
    
    # Process Boxes
    proposal_boxes = []
    for raw_iamge, raw_boxes in zip(raw_images, raw_boxes_list):
        raw_height, raw_width = raw_image.shape[:2]
        raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())
        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        #print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)
        proposal_boxes.append(boxes)
    
    # ----
    # Run Backbone Res1-Res4
    features = predictor.model.backbone(images.tensor)
    
    # Run RoI head for each proposal (RoI Pooling + Res5)
    features = [features[f] for f in predictor.model.roi_heads.in_features]
    box_features = predictor.model.roi_heads._shared_roi_transform(
        features, proposal_boxes
    )
    feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x
    # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
    features_list = feature_pooled.split([len(bb) for bb in raw_boxes_list])
    return features_list


def apply_augs(im, boxes):
    data_dict = {
        'image': im,
        'bboxes': boxes,
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

    # new_boxes = [
    #     {
    #         **boxes[i],
    #         'xmin': new_data_dict['bboxes'][i][0],
    #         'ymin': new_data_dict['bboxes'][i][1],
    #         'xmax': new_data_dict['bboxes'][i][2],
    #         'ymax': new_data_dict['bboxes'][i][3],
    #     }
    #     for i in range(len(boxes))
    # ]
    return new_data_dict['image'], new_data_dict['bboxes']


def freeze_model_bn(predictor):
    model = predictor.model
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False


def freeze_model_backbone(model, module_names=['stem', 'res2']):
    backbone = model.backbone
    for name, module in backbone.named_children():
        if name in module_names:
            for param in module.parameters():
                param.requires_grad = False
# %%

def run_test():
    # %%
    import matplotlib.pyplot as plt
    from PIL import Image
    img = Image.open('demo/data/images/000456.jpg')
    img = np.array(img)
    p = build_predictor()
    freeze_model_bn(p)
    freeze_model_backbone(p.model)
    img2 = p.transform_gen.get_transform(img).apply_image(img)
    # plt.imshow(img2)

    # %%

    img = np.zeros([512, 512, 3], dtype=np.uint8)
    boxes = np.array([
        [0, 0, 100, 199],
        [0, 50, 100, 199],
    ])
    doit_without_boxes(p, img)
    f = extract_boxes_feature(p, img, boxes)

    imgs = [
        np.zeros([512, 512, 3], dtype=np.uint8),
        np.zeros([384, 384, 3], dtype=np.uint8),
        np.zeros([640, 360, 3], dtype=np.uint8),
    ]
    boxes_list = [
        np.array([
            [0, 0, 100, 199],
            [0, 50, 100, 199],
        ]),
        np.array([
            [0, 0, 100, 199],
            [0, 50, 100, 199],
            [20, 50, 120, 199],
            [20, 30, 120, 199],
        ]),
        np.array([
            [0, 0, 100, 199],
            [0, 50, 100, 199],
        ]),
    ]
    fs = extract_batch_boxes_feature(p, imgs, boxes_list)
    import pdb; pdb.set_trace()
    print(f)
    # %%

if __name__ == "__main__":
    p = build_predictor()
    resnet = p.model.backbone
    im = cv2.imread('/home/ron/Downloads/hateful_meme_data/img/42953.png')
    # given_boxes = np.array(
    #     [[ 294.3217,  734.9891,  559.1793, 1110.9277],
    #     [ 339.7852,  113.7795,  566.7401,  385.4785],
    #     [  55.5721,   75.3408,  674.5536,  623.3557],
    #     [ 239.3932,  650.3275,  757.7485, 1177.7111],
    #     [  32.8885,  346.1343,  673.9932,  626.7932],
    #     [   0.0000,   29.4954,  785.8928,  684.8771],
    #     [ 289.9026,  895.4448,  356.8830,  999.8890]])
    instances, features = doit_without_boxes(p, im)
    pred = instances.to('cpu')