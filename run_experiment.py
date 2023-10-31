import os
import argparse
import yaml

from experiments.build_config import build_config
from experiments.train import Experiment

if __name__ == "__main__":
    """
    use the command line arguments to run the experiment:
    
    python run_experiment.py <experiment_name>  <optimizer_name>
    
    For example, to run the CNN experiment on the Fashion-MNIST dataset with the arTuRO optimizer, use the following command:
    
    python run_experiment.py fmnist_cnn  arturo
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",
                        help="Name of the dataset. Choose from [fmnist_cnn, fmnist_resnet18, cifar10_cnn, cifar10_resnet34, cifar100_cnn, cifar100_resnet34]")
    parser.add_argument("optimizer", help="Name of the optimizer. Choose from [arturo, adam, adamw, sgd]")
    args = parser.parse_args()
    root_path = os.getcwd()
    config = build_config(args.dataset, args.optimizer, root_path)

    exp = Experiment(config)

    exp.run()



