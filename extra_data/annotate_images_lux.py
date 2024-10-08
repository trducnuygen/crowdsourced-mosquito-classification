import csv
import os

from PIL import Image
import numpy as np


def annotate_inaturalist_images(img_dir: str, save_path: str):
    folder_name = os.path.split(img_dir)[-1]
    data = []

    for folder in os.listdir(img_dir):
        current_folder = os.path.join(img_dir, folder)
        for img_path in os.listdir(current_folder):
            if os.path.splitext(img_path)[-1] in [".png", ".jpeg", ".jpg"]:
                print(f"Processing {img_path}...")
                image = Image.open(os.path.join(current_folder, img_path)).convert(
                    "RGB"
                )

                if "aegypti" in current_folder:
                    class_label = "aegypti"
                elif "albopictus" in current_folder:
                    class_label = "albopictus"
                elif "anopheles" in current_folder:
                    class_label = "anopheles"
                elif "culex" in current_folder:
                    class_label = "culex"
                elif "culiseta" in current_folder:
                    class_label = "culiseta"
                elif ("japonicus" in current_folder) or ("koreicus" in current_folder):
                    class_label = "japonicus/koreicus"
                else:
                    raise ValueError(f"Class {current_folder} not know.")

                width, height = image.size
                boxes = [0, 0, width, height]
                data.append(
                    {
                        "img_fName": f"{current_folder.split('/')[-1]}/{img_path}",
                        "img_w": width,
                        "img_h": height,
                        "bbx_xtl": int(boxes[0]),
                        "bbx_ytl": int(boxes[1]),
                        "bbx_xbr": int(boxes[2]),
                        "bbx_ybr": int(boxes[3]),
                        "class_label": class_label,
                    }
                )

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


annotate_inaturalist_images(
    "./inaturalist-six-cropped",
    "./inaturalist-six-cropped/inaturalist.csv",
)

annotate_inaturalist_images(
    "./gbif-cropped",
    "./gbif-cropped/inaturalist.csv",
)
