from typing import List, Dict, Callable, Tuple

import torch as th
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd

import src.utils.classification as lc
import src.data_loader as dl


def _default_callbacks() -> List[Callback]:
    return [
        ModelCheckpoint(
            monitor="val_f1_score",
            mode="max",
            save_top_k=2,
            save_last=False,
            save_weights_only=True,
            filename="{epoch}-{val_loss}-{val_f1_score}-{val_multiclass_accuracy}",
        ),
    ]

def crossValid_callbacks(fold: int) -> List[Callback]:
    return [
        ModelCheckpoint(
            dirpath = "/home/pc2/Downloads/AI-Crowd/Mosquito-Classifiction/experiments/checkpoints/folds/",
            monitor="val_f1_score",
            mode="max",
            save_top_k=2,
            save_last=False,
            save_weights_only=True,
            filename="Fold{fold}-{epoch}-{val_loss}-{val_f1_score}-{val_multiclass_accuracy}",
        ),
        EarlyStopping(monitor="val_f1_score", mode="max", patience=5),
    ]

# for training
# CLASS_DICT = {
#     "albopictus": th.tensor([1, 0, 0, 0, 0, 0], dtype=th.float),
#     "culex": th.tensor([0, 1, 0, 0, 0, 0], dtype=th.float),
#     "japonicus/koreicus": th.tensor([0, 0, 1, 0, 0, 0], dtype=th.float),
#     "culiseta": th.tensor([0, 0, 0, 1, 0, 0], dtype=th.float),
#     "anopheles": th.tensor([0, 0, 0, 0, 1, 0], dtype=th.float),
#     "aegypti": th.tensor([0, 0, 0, 0, 0, 1], dtype=th.float),
# }

# for openmax
CLASS_DICT = {
    "albopictus": th.tensor([1, 0, 0, 0,], dtype=th.float),
    "culex": th.tensor([0, 1, 0, 0,], dtype=th.float),
    "japonicus/koreicus": th.tensor([0, 0, 1, 0,], dtype=th.float),
    "culiseta": th.tensor([0, 0, 0, 1,], dtype=th.float),

}

class_dict = {
"albopictus": th.tensor([1, 0, 0, 0, 0], dtype=th.float),
"culex": th.tensor([0, 1, 0, 0, 0], dtype=th.float),
"japonicus/koreicus": th.tensor([0, 0, 1, 0, 0], dtype=th.float),
"culiseta": th.tensor([0, 0, 0, 1, 0], dtype=th.float),
"mosquito": th.tensor([0, 0, 0, 0, 1], dtype=th.float)
}   



class ExperimentMosquitoClassifier:
    def __init__(
        self,
        img_dir: str,# = #"/home/pc2/AI-Crowd/Mosquito-Classifiction/data_round_2/final",
        annotations_csv: str, # = "",
        class_dict: Dict[str, th.Tensor] = CLASS_DICT,
        class_dict_test: Dict[str, th.Tensor] = class_dict,
    ):
        self.img_dir = img_dir
        self.annotations_csv = annotations_csv
        self.class_dict = class_dict
        self.class_dict_test = class_dict_test

    def get_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dataset_name: str,
        data_aug: str,
        bs: float = 16,
        img_size: Tuple[float, float] = (224, 224),
        shift_box: bool = False,
    ) -> List[DataLoader]:
        transform = dl.pre_process(dataset_name)

        train_dataset = dl.SimpleClassificationDataset(
            train_df,
            self.img_dir,
            self.class_dict,
            transform,
            dl.aug(data_aug, img_size),
            shift_box=shift_box,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        val_dataset = dl.SimpleClassificationDataset(
            val_df,
            self.img_dir,
            self.class_dict,
            transform,
            dl.aug("resize", img_size),
            class_balance=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
        )

        test_dataset = dl.SimpleClassificationDataset(
            test_df,
            self.img_dir,
            self.class_dict_test,
            transform,
            dl.aug("resize", img_size),
            class_balance=False,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
        )

        return train_dataloader, val_dataloader, test_dataloader

    
