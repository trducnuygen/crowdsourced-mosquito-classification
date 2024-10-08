# preprocessing steps from HCA97
import os
from typing import Optional, Union, Tuple
import random
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms as T
import albumentations as A
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def pre_process(dataset: str = "laion") -> T.Compose:
    return T.Compose(
        [
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


def aug(
    data_aug: str = "image_net", img_size: Tuple[int, int] = (224, 224)
) -> T.Compose:
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(
                size=img_size,
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
    )
    if data_aug == "image_net":
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                T.Resize(
                    size=img_size,
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )

    elif data_aug == "hca":
        aug8p3 = A.OneOf(
            [
                A.Sharpen(p=0.3),
                A.ToGray(p=0.3),
                A.CLAHE(p=0.3),
            ],
            p=0.5,
        )

        blur = A.OneOf(
            [
                A.GaussianBlur(p=0.3),
                A.MotionBlur(p=0.3),
            ],
            p=0.5,
        )

        transform = A.Compose(
            [
                A.ShiftScaleRotate(
                    rotate_limit=45,
                    scale_limit=0.1,
                    border_mode=cv2.BORDER_REFLECT,
                    interpolation=cv2.INTER_CUBIC,
                    p=0.5,
                ),
                A.Resize(*img_size, cv2.INTER_CUBIC),
                aug8p3,
                blur,
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    elif data_aug == "aug_mix":
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.AugMix(),
                T.Resize(
                    size=img_size,
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )
    elif data_aug == "happy_whale":
        aug8p3 = A.OneOf(
            [
                A.Sharpen(p=0.3),
                A.ToGray(p=0.3),
                A.CLAHE(p=0.3),
            ],
            p=0.5,
        )

        transform = A.Compose(
            [
                A.ShiftScaleRotate(
                    rotate_limit=15,
                    scale_limit=0.1,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
                A.Resize(*img_size, cv2.INTER_CUBIC),
                aug8p3,
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )

    elif data_aug == "cut_out":
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ImageCompression(quality_lower=99, quality_upper=100),
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.7,
                ),
                A.Resize(*img_size, cv2.INTER_CUBIC),
                A.Cutout(
                    max_h_size=int(img_size[0] * 0.4),
                    max_w_size=int(img_size[1] * 0.4),
                    num_holes=1,
                    p=0.5,
                ),
            ]
        )
    elif data_aug == "clip":
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(
                    size=img_size,
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Resize(
                    size=img_size,
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )
    elif data_aug == "clip+image_net":
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                T.RandomResizedCrop(
                    size=img_size,
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Resize(
                    size=img_size,
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )

    return transform


def read_image_cv2(f_name: str, gray_scale: bool = False) -> np.ndarray:
    img = cv2.imread(
        f_name, cv2.IMREAD_ANYCOLOR if not gray_scale else cv2.IMREAD_GRAYSCALE
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def class_balancing(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.class_label.value_counts().to_dict()
    max_label = max(list(counts.items()), key=lambda x: x[1])

    for key, value in counts.items():
        if key == max_label[0]:
            continue

        df_label = df[df.class_label == key].sample(
            n=max_label[1] - value, replace=True
        )
        df = pd.concat([df, df_label])

    return df


class SimpleClassificationDataset(Dataset):
    def __init__(
        self,
        annotations_df: pd.DataFrame,
        img_dir: str,
        class_dict: dict,
        transform: Optional[T.Compose] = None,
        data_augment: Optional[Union[T.Compose, A.Compose]] = None,
        class_balance: bool = True,
        shift_box: bool = False,
    ):
        self.df = annotations_df
        if class_balance:
            self.df = class_balancing(annotations_df)

        self.img_dir = img_dir
        self.class_dict = class_dict
        self.transform = transform
        self.data_augment = data_augment
        self.shift_box = shift_box

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cv2.setNumThreads(6)

        f_name, w, h, x_tl, y_tl, x_br, y_br, label = self.df.iloc[idx]
        img = read_image_cv2(os.path.join(self.img_dir, f_name))

        if self.shift_box:
            _x_tl = min(w, max(0, x_tl + random.randint(-int(0.01 * w), int(0.01 * w))))
            _x_br = min(w, max(0, x_br + random.randint(-int(0.01 * w), int(0.01 * w))))
            _y_tl = min(h, max(0, y_tl + random.randint(-int(0.01 * h), int(0.01 * h))))
            _y_br = min(h, max(0, y_br + random.randint(-int(0.01 * h), int(0.01 * h))))

            if abs(_x_tl - _x_br) > 30 and _x_tl < _x_br:
                x_tl = _x_tl
                x_br = _x_br

            if abs(_y_tl - _y_br) > 30 and _y_tl < _y_br:
                y_br = _y_br
                y_tl = _y_tl

        img_ = img[int(y_tl) : int(y_br), int(x_tl) : int(x_br), :]
        if img_.shape[0] * img_.shape[1] != 0:
            img = img_

        if self.data_augment:
            if isinstance(self.data_augment, A.Compose):
                img = self.data_augment(image=img)["image"]
            else:
                img = self.data_augment(img)

        if self.transform:
            img = self.transform(img)

        if self.class_dict:
            label = self.class_dict[label]
        return (img, label)


class TestYOLOCLIPDataset(Dataset):
    def __init__(
        self,
        annotations_df: pd.DataFrame,
        img_dir: str,
        class_dict: dict,
        transform: Optional[T.Compose] = None,
    ):
        self.df = annotations_df
        self.img_dir = img_dir
        self.class_dict = class_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cv2.setNumThreads(6)

        f_name, w, h, x_tl, y_tl, x_br, y_br, label = self.df.iloc[idx]
        box = (x_tl, y_tl, x_br, y_br)
        img = read_image_cv2(os.path.join(self.img_dir, f_name))

        if self.class_dict:
            label = self.class_dict[label]

        return (img, label, box)


if __name__ == "__main__":
    import torch as th
    import torchvision.transforms as T
    import albumentations as A
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))

    img_dir = "../data/train"
    annotations_csv = "../data/train.csv"
    annotations_df = pd.read_csv(annotations_csv)

    class_dict = {
    "albopictus": th.tensor(0, dtype=th.float),
    "culex": th.tensor(1, dtype=th.float),
    "japonicus/koreicus": th.tensor(2, dtype=th.float),
    "aegypti": th.tensor(3, dtype=th.float),
    "mosquito": th.tensor(4, dtype=th.float)
    }   

    transform = pre_process("")

    data_augmentation = aug("image_net")

    ds = SimpleClassificationDataset(
        annotations_df=annotations_df,
        img_dir=img_dir,
        class_dict=class_dict,
        transform=transform,
        data_augment=data_augmentation,
    )
    for i in range(10):
        res = ds[i]
        img = res["img"]

        img_bbox = th.tensor(
            255 * (img - img.min()) / (img.max() - img.min()), dtype=th.uint8
        )
        show(img_bbox)
        plt.show()
