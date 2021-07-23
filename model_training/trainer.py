from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

import utils.pytorch as pt_utils
from model_training.config import TrainConfig


class Trainer:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config

    def train(
        self, nn_module: nn.Module, dataloaders: Dict[str, torch.utils.data.DataLoader]
    ):
        assert "train" in dataloaders, "No train dataset specified!"
        assert "valid" in dataloaders, "No valid dataset specified!"
        assert torch.cuda.is_available(), "Only GPU training is supported!"

        optimizer = pt_utils.create_optimizer(
            params=nn_module.parameters(),
            optimizer_name=self.train_config.optimization_schedule.optimizer_name,
            init_lr=self.train_config.optimization_schedule.init_lr,
            weight_decay=self.train_config.optimization_schedule.weight_decay,
        )

        lr_scheduler = pt_utils.create_lr_scheduler(
            optimizer=optimizer,
            num_iterations=len(dataloaders["train"]) * self.train_config.num_epochs,
            gamma=self.train_config.optimization_schedule.gamma,
            milestones=self.train_config.optimization_schedule.milestones,
        )

        criterion = nn.CrossEntropyLoss()

        nn_module.train()
        nn_module.cuda()
        for epoch_id in range(self.train_config.num_epochs):
            for iter_id, batch in tqdm(enumerate(dataloaders["train"]), desc=f"{epoch_id} epoch training in progress"):
                batch = pt_utils.to_device(batch, "cuda:0")
                logits = nn_module(batch["input_ids"], batch["attention_masks"])
                optimizer.zero_grad()
                loss = criterion(logits, batch["targets"])
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            metrics = self.validate(
                nn_module=nn_module, dataloader=dataloaders["valid"]
            )

    def validate(
        self, nn_module: nn.Module, dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, Dict[str, float]]:
        nn_module.eval()
        y_true_all = []
        y_pred_all = []
        for batch in tqdm(dataloader, desc="Model evaluation"):
            batch = pt_utils.to_device(batch, "cuda:0")
            y_pred = nn_module(batch["input_ids"], batch["attention_masks"])
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.detach().cpu().numpy()
            y_true = batch["targets"].detach().cpu().numpy()
            y_true_all.extend(list(y_true))
            y_pred_all.extend(list(y_pred))
        print(classification_report(y_true=y_true_all, y_pred=y_pred_all))
        return classification_report(
            y_true=y_true_all, y_pred=y_pred_all, output_dict=True
        )
