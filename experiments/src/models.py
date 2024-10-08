# HCA97 source code
import torch as th
from torch import nn
import open_clip


class HeadV1(nn.Module):
    def __init__(self, f_out: int, f_in: int):
        super().__init__()

        self.label = nn.Sequential(
            nn.BatchNorm1d(f_in),
            nn.Dropout1d(),
            nn.LeakyReLU(),
            nn.Linear(f_in, f_out),
        )

    def forward(self, x):
        return self.label(x)


class HeadV4(nn.Module):
    def __init__(self, f_out: int, f_in: int):
        super().__init__()

        self.label = nn.Sequential(
            nn.BatchNorm1d(f_in),
            nn.Dropout1d(),
            nn.Linear(f_in, f_out),
        )

    def forward(self, x):
        return self.label(x)


class HeadV5(nn.Module):
    def __init__(self, f_out: int, f_in: int):
        super().__init__()

        self.label = nn.Sequential(
            nn.BatchNorm1d(f_in),
            nn.Dropout1d(p=0.9),
            nn.Linear(f_in, f_out),
        )

    def forward(self, x):
        return self.label(x)


class HeadV6(nn.Module):
    def __init__(self, f_out: int, f_in: int):
        super().__init__()

        self.label = nn.Sequential(
            nn.BatchNorm1d(f_in),
            nn.Dropout1d(p=0.75),
            nn.Linear(f_in, f_out),
        )

    def forward(self, x):
        return self.label(x)


class HeadV2(nn.Module):
    def __init__(self, f_out: int, f_in: int):
        super().__init__()

        self.label = nn.Sequential(
            nn.BatchNorm1d(f_in),
            nn.Linear(f_in, f_out),
        )

    def forward(self, x):
        return self.label(x)


class HeadV3(nn.Module):
    def __init__(self, f_out: int, f_in: int):
        super().__init__()

        self.label = nn.Sequential(
            nn.BatchNorm1d(f_in),
            nn.Linear(f_in, f_in, bias=False),
            nn.BatchNorm1d(f_in),
            nn.LeakyReLU(),
            nn.Dropout1d(),
            nn.Linear(f_in, f_in, bias=False),
            nn.BatchNorm1d(f_in),
            nn.LeakyReLU(),
            nn.Dropout1d(),
            nn.Linear(f_in, f_out),
        )

    def forward(self, x):
        return self.label(x)


class HeadV8(nn.Module):
    def __init__(self, f_out: int, f_in: int):
        super().__init__()

        self.label = nn.Sequential(
            nn.Dropout1d(),
            nn.Linear(f_in, f_out),
        )

    def forward(self, x):
        return self.label(x)


class HeadV7(nn.Module):
    def __init__(self, f_out: int, f_in: int):
        super().__init__()

        self.label = nn.Sequential(
            nn.LayerNorm(f_in),
            nn.Dropout1d(),
            nn.Linear(f_in, f_out),
        )

    def forward(self, x):
        return self.label(x)


class CLIPClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int = 6,
        model_name: str = "ViT-L-14",
        data: str = "datacomp_xl_s13b_b90k",
        head_version: int = 1,
        hd_lr: float = None,
        hd_wd: float = None,
    ):
        super().__init__()

        self.backbone = open_clip.create_model_and_transforms(
            model_name.split(".")[0], pretrained=data
        )[0].visual

        if model_name == "ViT-L-14":
            self.n = 768
            self.lrs = dict(
                back_lrs={"8": 1.25e-6, "16": 2.5e-6, "20": 5e-6, "24": 10e-6},
                back_wd=1e-3,
                hd_lr=3e-4 or hd_lr,
                hd_wd=1e-5 or hd_wd,
            )
        elif model_name == "ViT-H-14":
            self.n = 1024
            self.lrs = {
                "back_lrs": {"10": 1.25e-6, "20": 2.5e-6, "26": 5e-6, "32": 10e-6},
                "back_wd": 1e-3,
                "hd_lr": 3e-4 or hd_lr,
                "hd_wd": 1e-5 or hd_wd,
            }
        elif model_name == "ViT-B-16":
            self.n = 512
            self.lrs = {
                "back_lrs": {"1": 2.5e-6, "7": 5e-6, "12": 10e-6},
                "back_wd": 1e-3,
                "hd_lr": 3e-4 or hd_lr,
                "hd_wd": 1e-5 or hd_wd,
            }
        elif model_name in [
            "convnext_large_d",
            "convnext_large_d_320",
        ]:
            self.n = 768
            self.lrs = {
                "back_lrs": {"1": 1.25e-6, "2": 2.5e-6, "3": 5e-6, "4": 10e-6},
                "back_wd": 1e-3,
                "hd_lr": 3e-4 or hd_lr,
                "hd_wd": 1e-5 or hd_wd,
            }
        elif model_name in [
            "convnext_large_d.trunk",
            "convnext_large_d_320.trunk",
        ]:
            self.backbone = self.backbone.trunk
            self.n = 1536
            self.lrs = {
                "back_lrs": {"1": 1.25e-6, "2": 2.5e-6, "3": 5e-6, "4": 10e-6},
                "back_wd": 1e-3,
                "hd_lr": 3e-4 or hd_lr,
                "hd_wd": 1e-5 or hd_wd,
            }
        elif model_name == "convnext_xxlarge":
            self.n = 1024
            self.lrs = {
                "back_lrs": {"1": 1.25e-6, "2": 2.5e-6, "3": 5e-6, "4": 10e-6},
                "back_wd": 1e-3,
                "hd_lr": 3e-4 or hd_lr,
                "hd_wd": 1e-5 or hd_wd,
            }
        elif model_name == "hf-hub:imageomics/bioclip":
            self.n = 512
            self.lrs = {
                "back_lrs": {"1": 2.5e-6, "7": 5e-6, "12": 10e-6},
                "back_wd": 1e-3,
                "hd_lr": 3e-4 or hd_lr,
                "hd_wd": 1e-5 or hd_wd,
            }            
        else:
            raise ValueError

        if head_version == 2:
            self.label = HeadV2(n_classes, self.n)
        elif head_version == 3:
            self.label = HeadV3(n_classes, self.n)
        elif head_version == 4:
            self.label = HeadV4(n_classes, self.n)
        elif head_version == 5:
            self.label = HeadV5(n_classes, self.n)
        elif head_version == 6:
            self.label = HeadV6(n_classes, self.n)
        elif head_version == 7:
            self.label = HeadV7(n_classes, self.n)
        elif head_version == 8:
            self.label = HeadV8(n_classes, self.n)
        else:
            self.label = HeadV1(n_classes, self.n)

        self.n_classes = n_classes

    def forward(self, x: th.tensor) -> th.tensor:
        x = self.backbone(x)
        return self.label(x)

    def get_parameter_section(self, parameters, lr=None, wd=None):
        # https://github.com/IvanAer/G-Universal-CLIP
        parameter_settings = []

        lr_is_dict = isinstance(lr, dict)
        wd_is_dict = isinstance(wd, dict)

        layer_no = None
        for n, p in parameters:
            for split in n.split("."):
                if split.isnumeric():
                    layer_no = int(split)

            if not layer_no:
                layer_no = 0

            if lr_is_dict:
                for k, v in lr.items():
                    if layer_no < int(k):
                        temp_lr = v
                        break
            else:
                temp_lr = lr

            if wd_is_dict:
                for k, v in wd.items():
                    if layer_no < int(k):
                        temp_wd = v
                        break
            else:
                temp_wd = wd

            parameter_setting = {"params": p, "lr": temp_lr, "weight_decay": temp_wd}
            parameter_settings.append(parameter_setting)
        return parameter_settings

    def get_learnable_params(self) -> list:
        back_lrs = self.lrs["back_lrs"]
        back_wd = self.lrs["back_wd"]
        hd_lr = self.lrs["hd_lr"]
        hd_wd = self.lrs["hd_wd"]

        parameter_settings = []

        if back_lrs and back_wd:
            parameter_settings.extend(
                self.get_parameter_section(
                    [(n, p) for n, p in self.backbone.named_parameters()],
                    lr=back_lrs,
                    wd=back_wd,
                )
            )

        parameter_settings.extend(
            self.get_parameter_section(
                [(n, p) for n, p in self.label.named_parameters()], lr=hd_lr, wd=hd_wd
            )
        )

        return parameter_settings


if __name__ == "__main__":
    m = CLIPClassifier(6, "ViT-B-16", "datacomp_l_s1b_b8k")

    x = th.rand([10, 3, 224, 224])

    print(m(x).shape)
