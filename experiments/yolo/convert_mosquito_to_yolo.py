import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import yaml
import sys

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


#img_dir = "../../data_round_2/final"
#annotation_csv = "../../data_round_2/yolo/train_val_yolo.csv"
#annotation_csv = "../../data_round_2/phase2_train_v0.csv"
class_dict = {
    "albopictus": 0,
    "culex": 1,
    "japonicus/koreicus": 2,
    "culiseta": 3,
    "anopheles": 4,
    "aegypti": 5,
}


output_dir = "../../data_yolo"
yaml_file = "yolo_config_mos.yml"

def clear_directory(directory: str):
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    else:
        print(f"Directory {directory} does not exist.")

def convert_2_yolo_boxxes(img_shape: tuple, bbox: tuple) -> tuple:
    img_w, img_h = img_shape
    x_tl, y_tl, x_br, y_br = bbox

    box_w = x_br - x_tl
    box_h = y_br - y_tl

    x_c = (x_tl + x_br) / 2
    y_c = (y_tl + y_br) / 2

    return (x_c / img_w, y_c / img_h, box_w / img_w, box_h / img_h)


def create_folders(output_dir: str):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)


def create_yolo_folder(df: pd.DataFrame, folder_name: str, start_index: int = 0):
    def _loop(i: int):
        nonlocal folder_name, df, start_index
        global img_dir, output_dir, class_dict

        f_name, w, h, x_tl, y_tl, x_br, y_br, label = df.iloc[i]
        src_path = "../" + f_name
        #src_path = os.path.join(img_dir, f_name)
        dst_path = os.path.join(
            output_dir, "images", folder_name, f"{i + start_index}.jpeg"
        )
        shutil.copy(src_path, dst_path)

        bbox = convert_2_yolo_boxxes((w, h), (x_tl, y_tl, x_br, y_br))

        label_path = os.path.join(
            output_dir, "labels", folder_name, f"{i + start_index}.txt"
        )
        with open(label_path, "w") as f:
            f.write(f"{class_dict[label]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

    with ThreadPoolExecutor(10) as exe:
        jobs = []
        for i in range(len(df)):
            jobs.append(exe.submit(_loop, i))

        for job in tqdm(jobs):
            job.result()


if __name__ == "__main__":
    
    train_df = pd.read_csv("../../data_round_2/closedSet/train.csv")
    val_df = pd.read_csv("../../data_round_2/closedSet/val.csv")

    create_folders(output_dir)
    create_yolo_folder(train_df, "train")
    create_yolo_folder(val_df, "val")

    config = {
        "path": output_dir,
        "train": "./images/train",
        "val": "./images/val",
        "names": dict((v, k) for k, v in class_dict.items()),
    }

    with open(yaml_file, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
