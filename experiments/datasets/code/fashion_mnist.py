import torch
import torch.nn as nn
import torchvision
from torch.nn import Conv2d
from torchvision import datasets


# The dataset of pytorch is very unoptimized. We can drastically increase the loading speed by changing the __getitem__ method:
class OwnFMNIST(datasets.FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                         download=download)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.to(torch.float32).unsqueeze(0) / 255.
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FashionMNIST:
    @classmethod
    def create_dataset(cls, env_config):
        train_ds = OwnFMNIST(
            env_config["path"], train=True, download=True,
            transform=None)
        test_ds = OwnFMNIST(
            env_config["path"], train=False, download=True,
            transform=None)
        return train_ds, test_ds

    @classmethod
    def get_model(cls, env_config):
        if env_config["model_name"] == "cnn":
            return nn.Sequential(
                nn.Conv2d(1, 32, (3, 3), stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(0.20),
                nn.Conv2d(32, 64, (3, 3), stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(0.20),
                nn.Conv2d(64, 128, (3, 3), stride=1, padding=0),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Flatten(),
                nn.Linear(3 * 3 * 128, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10),
            )
        elif env_config["model_name"] == "resnet18":
            model = torchvision.models.resnet18(num_classes=10)
            # change the first layer to accept 1 channel instead of 3
            model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            return model

        else:
            raise ValueError()
