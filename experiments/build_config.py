import os
import yaml


def build_config(dataset_name, optimizer_name, root_dir):
    env_path = os.path.join(root_dir, "experiments", "datasets", "configs", f"{dataset_name}.yml")
    opti_path = os.path.join(root_dir, "experiments", "hyperparameter_configs", f"{optimizer_name}", f"{dataset_name}.yml")

    with open(env_path, "r") as in_file:
        env_config = yaml.safe_load(in_file)
    with open(opti_path, "r") as in_file:
        opti_config = yaml.safe_load(in_file)

    config = {
        "env": env_config,
        "opti": opti_config,
        "optimizer_name": optimizer_name,
    }

    return config
