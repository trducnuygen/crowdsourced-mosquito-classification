# HCA97 source code
import torch
import torchvision.transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
from typing import List

IMG_SIZE = (224, 224)
USE_CHANNEL_LAST = False
DATASET = "laion"
DEVICE = "cuda:0"
PRESERVE_ASPECT_RATIO = False
SHIFT = 0


def pre_process_foo(img_size: tuple, dataset: str = "laion") -> T.Compose:
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize(
                size=img_size,
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073)
                if dataset != "imagenet"
                else IMAGENET_DEFAULT_MEAN,
                std=(0.26862954, 0.26130258, 0.27577711)
                if dataset != "imagenet"
                else IMAGENET_DEFAULT_STD,
            ),
        ]
    )


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


@torch.no_grad()
def prep_cls(image: np.ndarray, img_size: tuple) -> torch.Tensor:
    # ignore yolo
    image_cropped = image 

    # assuming image.shape = 224, 224 and no preserve aspect ratio
    if PRESERVE_ASPECT_RATIO:
        w, h = image_cropped.shape[:2]
        if w > h:
            x = torch.unsqueeze(
                pre_process_foo(
                    (img_size[0], max(int(img_size[1] * h / w), 32)), DATASET
                )(image_cropped),
                0,
            )
        else:
            x = torch.unsqueeze(
                pre_process_foo(
                    (max(int(img_size[0] * w / h), 32), img_size[1]), DATASET
                )(image_cropped),
                0,
            )
    else:
        x = torch.unsqueeze(x, 0)
    
    x = x.to(device=DEVICE)
    if USE_CHANNEL_LAST:
        x = x.to(memory_format=torch.channels_last) 

    return x

# -------------------
# for yolo -> clip
def get_bbox(img, yolo_model):
    results = yolo_model(img, verbose=False, device=DEVICE, max_det=1)
    img_w, img_h, _ = img.shape
    bbox = [0, 0, img_w, img_h]
    conf = 0.0
    for result in results:
        _bbox = [0, 0, img_w, img_h]
        _conf = 0.0

        bboxes_tmp = result.boxes.xyxy.tolist()
        confs_tmp = result.boxes.conf.tolist()

        for bbox_tmp, conf_tmp in zip(bboxes_tmp, confs_tmp):
            if conf_tmp > _conf:
                _bbox = bbox_tmp
                _conf = conf_tmp

        if _conf > conf:
            bbox = _bbox
            conf = _conf

    bbox = [int(float(mcb)) for mcb in bbox]
    return bbox, conf

def prep_x(bbox, img, iou, img_size):
    filtered = False
    try:
        if iou < 0.75:
            raise Exception
        image_cropped = img[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
        bbox = [bbox[0] + SHIFT, bbox[1] + SHIFT, bbox[2] - SHIFT, bbox[3] - SHIFT]
    except Exception as e:
        print("Too low confidence\n", e)
        filtered = True
        image_cropped = img # no crop

    if PRESERVE_ASPECT_RATIO:
        w, h = image_cropped.shape[:2]
        if w > h:
            x = torch.unsqueeze(
                pre_process_foo(
                    (img_size[0], max(int(img_size[1] * h / w), 32)), DATASET
                )(image_cropped),
                0,
            )
        else:
            x = torch.unsqueeze(
                pre_process_foo(
                    (max(int(img_size[0] * w / h), 32), img_size[1]), DATASET
                )(image_cropped),
                0,
            )
    else:
        x = torch.unsqueeze(pre_process_foo(img_size, DATASET)(image_cropped), 0)
    
    x = x.to(device=DEVICE)
    return x, filtered

def prepCls2(sample, yolo_model, img_size) -> torch.Tensor:
    img = sample[0]
    bbox_true = sample[2]
    bbox, conf = get_bbox(img, yolo_model)
    iou = calculate_iou(bbox, bbox_true)
    x = prep_x(bbox, img, iou, img_size)
    return x, iou
