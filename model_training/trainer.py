from typing import Dict

import torch
import torch.nn as nn
import utils.pytorch as pt_utils

from model_training.config import TrainConfig


class Trainer:

    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config

    def train(self, nn_module: nn.Module, dataloaders: Dict[str, torch.utils.data.DataLoader]):
        assert "train" in dataloaders, "No train dataset specified!"
        assert "valid" in dataloaders, "No valid dataset specified!"

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
            milestones=self.train_config.optimization_schedule.milestones
        )

        criterion = nn.CrossEntropyLoss()

        nn_module.train()
        for epoch_id in range(self.train_config.num_epochs):
            for iter_id, batch in enumerate(dataloaders["train"]):
                logits = model(batch["input_ids"], batch["attention_masks"])
                optimizer.zero_grad()
                loss = criterion(logits, batch["targets"])
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
