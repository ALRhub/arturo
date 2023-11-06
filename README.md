# arTuRO (Information-Theoretic Trust Region Optimization )

This repository contains the code for the paper [Information-Theoretic Trust Regions for Stochastic Gradient-Based
Optimization](https://arxiv.org/abs/2310.20574) by [Philipp Dahlinger](https://github.com/philippdahlinger/), Philipp Becker, Maximilian Hüttenrauch and Gerhard Neumann.

## Installation
Create a virtual environment and install the requirements:
```bash
pip install -r requirements.txt
```

## Usage
Use the command line arguments to run the experiment:
```bash
python run_experiment.py <experiment_name>  <optimizer_name>
```

For example, to run the CNN experiment on the Fashion-MNIST dataset with the arTuRO optimizer, use the following command:
```bash
python run_experiment.py fmnist_cnn  arturo
```
