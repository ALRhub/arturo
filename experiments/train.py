import os
import time
import numpy as np
import torch.optim


from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from experiments.util import get_model, get_dataloader, get_optimizer


class Experiment:
    def __init__(self, config: dict) -> None:
        self.opti_config = config["opti"]
        self.env_config = config["env"]
        self.optimizer_name = config["optimizer_name"]

        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device {self.device}")
        # gpu config
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        # seed
        np.random.seed(self.env_config["seed"])
        torch.manual_seed(self.env_config["seed"])

        # model
        self.model = get_model(self.env_config)
        self.model = self.model.to(self.device)

        # env
        self.train_dl, self.test_dl = get_dataloader(self.env_config)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.num_batches = len(self.train_dl)

        # optimizer
        self.optimizer = get_optimizer(self.optimizer_name, self.opti_config, self.model)

        # scheduler
        scheduler_config = self.env_config["scheduler"]
        if scheduler_config["name"] == "multi_step_lr":
            self.scheduler = MultiStepLR(
                optimizer=self.optimizer,
                milestones=scheduler_config["milestones"],
                gamma=self.opti_config.get("multi_step_lr_gamma", 0.1),
            )
        else:
            raise ValueError(f"Scheduler '{scheduler_config['name']} unknown.")

        # progress bar
        self.pbar = tqdm(self.env_config["epochs"])


    def run(self):
        for epoch in range(self.env_config["epochs"]):
            log_dict = self.iterate()
            # you can the log the dict with any logger you want. We originally used wandb here

    def iterate(self) -> dict:
        # get metrics
        t0 = time.perf_counter()

        train_loss, train_acc = self.compute_metrics(self.train_dl)
        test_loss, test_acc = self.compute_metrics(self.test_dl)

        t1 = time.perf_counter()
        # train one epoch
        forward_time, backward_time, optimizer_time, load_time = self.train_epoch()
        t2 = time.perf_counter()

        metric_time = t1 - t0
        train_time = t2 - t1
        self.scheduler.step()

        # logging
        self.pbar.set_description(f"{train_loss=}, {train_acc=}, {test_loss=}, {test_acc=}")
        self.pbar.update()
        logdict = {"train_loss": train_loss, "train_acc": train_acc,
                   "test_loss": test_loss,
                   "test_acc": test_acc, "forward_time": forward_time, "backward_time": backward_time,
                   "optimizer_time": optimizer_time, "load_time": load_time, "metric_time": metric_time, "train_time": train_time,
                   "wall_time": time.time(), "lr": self.scheduler.get_last_lr()[0]}
        if self.optimizer_name == "arturo":
            logdict["mean_eta"] = self.optimizer.aggregate_etas_and_reset()

        return logdict

    def train_epoch(self):
        self.model.train()
        forward_time = 0.
        backward_time = 0.
        optimizer_time = 0.
        load_time = 0.
        t00 = time.perf_counter()
        for x, y in self.train_dl:
            t0 = time.perf_counter()
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            t1 = time.perf_counter()
            loss.backward()
            t2 = time.perf_counter()
            self.optimizer.step()
            t3 = time.perf_counter()
            forward_time += (t1 - t0)
            backward_time += (t2 - t1)
            optimizer_time += (t3 - t2)
            load_time += t0 - t00

            t00 = time.perf_counter()

        return forward_time, backward_time, optimizer_time, load_time

    def compute_metrics(self, dl):
        if dl is None:
            return np.NAN, 0.0
        loss_sum = 0
        correct_preds = 0
        total_num = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in dl:
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                loss_sum += self.criterion(y_pred, y)
                cls_label = torch.argmax(y_pred, dim=-1)
                correct_preds += torch.sum(cls_label == y)
                total_num += y.shape[0]

        acc = correct_preds / total_num
        loss = loss_sum / len(dl)
        return float(loss), float(acc)
