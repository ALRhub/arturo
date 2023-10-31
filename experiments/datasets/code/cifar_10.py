import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.transforms as tt
import torchvision


# The Dataset of pytorch is very unoptimized. We can drastically increase the loading speed by changing the __getitem__ method:
class OwnCifar10(datasets.CIFAR10):
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


class Cifar10:
    TRAIN_MEAN = [125.307, 122.961, 113.8575]
    TRAIN_STD = [51.5865, 50.847, 51.255]

    @classmethod
    def create_dataset(cls, env_config):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_ds = OwnCifar10(
            env_config["path"], train=True, download=True,
            transform=transform_train)
        test_ds = OwnCifar10(
            env_config["path"], train=False, download=True,
            transform=transform_test)

        return train_ds, test_ds

    @classmethod
    def get_model(cls, env_config):
        if env_config["model_name"] == "resnet34":
            return torchvision.models.resnet34(num_classes=10)
        elif env_config["model_name"] == "cnn":
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
                nn.Linear(256, 10),
            )
        else:
            raise ValueError()
