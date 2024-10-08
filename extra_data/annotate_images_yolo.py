import csv
import os

from ultralytics import YOLO
import tqdm
import pandas as pd

yolo_path = "./Mosquito-Classifiction/yolo/runs/detect/classic/weights/best.pt"
img_dir = "./data_round_2/final"
save_path = "./data_round_2/phase2_train_v0_cleaned_yolo_best_annotations.csv"
annotations_csv = "./data_round_2/phase2_train_v0_cleaned.csv"
use_max = True

df = pd.read_csv(annotations_csv)

det = YOLO(yolo_path, task="detect")


def detect_images(img_path, t_iou=0.5, t_conf=0.5):
    results = det(os.path.join(img_dir, img_path), iou=t_iou, verbose=False)

    box_max = []

    for result in results:
        _bboxes = result.boxes.xyxy.tolist()
        _confs = result.boxes.conf.tolist()

        for bbox, conf in zip(_bboxes, _confs):
            if conf > t_conf:
                box_max.append(bbox)

    return box_max


def detect_images_conf_max(img_path, t_iou=0.5, shrink=5):
    results = det(os.path.join(img_dir, img_path), iou=t_iou, verbose=False)

    bboxes = []
    confs = []

    conf_max = 0.0
    box_max = []

    for result in results:
        _bboxes = result.boxes.xyxy.tolist()
        _confs = result.boxes.conf.tolist()

        for bbox, conf in zip(_bboxes, _confs):
            if conf > conf_max:
                conf_max = conf
                box_max = [
                    [
                        bbox[0] + shrink,
                        bbox[1] + shrink,
                        bbox[2] - shrink,
                        bbox[3] - shrink,
                    ]
                ]
    return box_max


df_dict = df.to_dict("records")
groupped_dict = {}
for row in df_dict:
    print(row)
    a: dict = groupped_dict.get(row["img_fName"], {})
    boxes: list = a.get("boxes", [])
    boxes.append([row["bbx_xtl"], row["bbx_ytl"], row["bbx_xbr"], row["bbx_ybr"]])
    a["class_label"] = row["class_label"]
    a["img_shape"] = [row["img_w"], row["img_h"]]
    a["boxes"] = boxes
    groupped_dict[row["img_fName"]] = a

for key in tqdm.tqdm(groupped_dict):
    if use_max:
        boxes = detect_images_conf_max(key)
    else:
        boxes = detect_images(key)

    if not boxes:
        continue

    groupped_dict[key]["boxes"] = boxes


data = [
    {
        "img_fName": key,
        "img_w": values["img_shape"][0],
        "img_h": values["img_shape"][1],
        "bbx_xtl": bbx_xtl,
        "bbx_ytl": bbx_ytl,
        "bbx_xbr": bbx_xbr,
        "bbx_ybr": bbx_ybr,
        "class_label": values["class_label"],
    }
    for key, values in groupped_dict.items()
    for bbx_xtl, bbx_ytl, bbx_xbr, bbx_ybr in values["boxes"]
]

with open(save_path, "w") as csvfile:
    fieldnames = [
        "img_fName",
        "img_w",
        "img_h",
        "bbx_xtl",
        "bbx_ytl",
        "bbx_xbr",
        "bbx_ybr",
        "class_label",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
