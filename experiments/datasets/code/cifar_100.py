import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as tt
import torchvision.models as tvmodels


# The dataset of pytorch is very unoptimized. We can drastically increase the loading speed by changing the __getitem__ method:
class OwnCifar100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                         download=download)
        self.to_tensor = tt.ToTensor()

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # don't convert to PIL image, but to a tensor directly
        img = self.to_tensor(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class Cifar100:
    TRAIN_MEAN = [129.387, 124.1085, 112.4805]
    TRAIN_STD = [51.2805, 50.6685, 51.6375]

    @classmethod
    def create_dataset(cls, env_config):
        stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
        train_transform = tt.Compose([
            tt.RandomHorizontalFlip(),
            tt.RandomCrop(32, padding=4, padding_mode="reflect"),

            tt.Normalize(*stats)
        ])

        test_transform = tt.Compose([
            tt.Normalize(*stats)
        ])

        train_ds = OwnCifar100(
            env_config["path"], train=True, download=True,
            transform=train_transform)
        test_ds = OwnCifar100(
            env_config["path"], train=False, download=True,
            transform=test_transform)

        return train_ds, test_ds

    @classmethod
    def get_model(cls, env_config):
        if env_config["model_name"] == "cnn":
            return nn.Sequential(
                nn.Conv2d(3, 32, (3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(0.20),
                nn.Conv2d(32, 64, (3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(0.20),
                nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(0.2),
                nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(0.2),
                nn.Flatten(),
                nn.Linear(2 * 2 * 256, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 100),
            )
        elif env_config["model_name"] == "resnet34":
            return tvmodels.resnet34(num_classes=100)
        else:
            raise ValueError()
