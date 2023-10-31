import torch
from torch.utils.data import DataLoader

from experiments.datasets.code.cifar_10 import Cifar10
from experiments.datasets.code.cifar_100 import Cifar100
from experiments.datasets.code.fashion_mnist import FashionMNIST
from arturo.arturo_optimizer import Arturo


def get_env_class(env_config):
    # global name resolve
    env_classes = {
        "fmnist": FashionMNIST,
        "cifar10": Cifar10,
        "cifar100": Cifar100,
    }

    dataset_name = env_config["dataset_name"]
    return env_classes[dataset_name]


def get_model(env_config):
    env_class = get_env_class(env_config)
    return env_class.get_model(env_config)


def get_dataloader(env_config):
    env_class = get_env_class(env_config)
    train_ds, test_ds = env_class.create_dataset(env_config)

    # dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=env_config["train_batch_size"], shuffle=True,
        num_workers=env_config.get("num_worker", 0), pin_memory=env_config.get("pin_memory", False))

    test_dl = DataLoader(
        test_ds, batch_size=env_config["test_batch_size"], shuffle=True,
        num_workers=env_config.get("num_worker", 0), pin_memory=env_config.get("pin_memory", False))

    return train_dl, test_dl


def get_optimizer(optimizer_name, opti_config, model):
    if optimizer_name == "arturo":
        # change the kl_bound to lr due to scheduler
        new_opti_config = {}
        for key in opti_config:
            new_opti_config[key] = opti_config[key]
        new_opti_config["lr"] = new_opti_config["kl_bound"]
        del new_opti_config["kl_bound"]

        return Arturo(model.parameters(), new_opti_config)

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=opti_config["lr"],
                                betas=(opti_config["beta_1"], opti_config["beta_2"]))
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=opti_config["lr"],
                                 betas=(opti_config["beta_1"], opti_config["beta_2"]),
                                 weight_decay=opti_config.get("weight_decay", 0.0))
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=opti_config["lr"], momentum=opti_config.get("momentum", 0.0),
                               weight_decay=opti_config.get("weight_decay", 0.0))

    else:
        raise ValueError(f"Optimizer name '{optimizer_name}' not valid.")
