import os
from typing import Dict

import torch
import torch.nn as nn
import transformers
from sklearn.metrics import classification_report
from tqdm import tqdm

import utils.pytorch as pt_utils
from model_training.config import TrainConfig
from utils.helpers import get_repo_root


class Trainer:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config

    def train(
        self, nn_module: nn.Module, dataloaders: Dict[str, torch.utils.data.DataLoader]
    ):
        assert "train" in dataloaders, "No train dataset specified!"
        assert "valid" in dataloaders, "No valid dataset specified!"
        assert torch.cuda.is_available(), "Only GPU training is supported!"

        self._freeze_embedding_layer(nn_module=nn_module)
        optimizer = pt_utils.create_optimizer(
            params=nn_module.parameters(),
            optimizer_name=self.train_config.optimization_schedule.optimizer_name,
            init_lr=self.train_config.optimization_schedule.init_lr,
            weight_decay=self.train_config.optimization_schedule.weight_decay,
        )

        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.train_config.optimization_schedule.num_warmup_steps,
            num_training_steps=len(dataloaders["train"]) * self.train_config.num_epochs,
        )

        criterion = nn.CrossEntropyLoss()

        nn_module.train()
        nn_module.cuda()
        self.best_metric = 0.0
        for epoch_id in range(self.train_config.num_epochs):
            for iter_id, batch in tqdm(
                enumerate(dataloaders["train"]),
                desc=f"{epoch_id} epoch training in progress",
            ):
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
            self._save_model(nn_module=nn_module, metrics=metrics)

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
        nn_module.train()
        print(classification_report(y_true=y_true_all, y_pred=y_pred_all))
        return classification_report(
            y_true=y_true_all, y_pred=y_pred_all, output_dict=True
        )

    def _save_model(self, nn_module: nn.Module, metrics: Dict[str, Dict[str, float]]):
        main_metric_val = metrics["macro avg"][self.train_config.main_metric]
        if main_metric_val > self.best_metric:
            nn_module.cpu()
            output_dir = os.path.join(
                get_repo_root(), "trained_models", self.train_config.output_model_name
            )
            os.makedirs(output_dir, exist_ok=True)
            torch.save(nn_module.state_dict(), os.path.join(output_dir, "weights.pt"))
            self.best_metric = main_metric_val
        nn_module.cuda()

    def _freeze_embedding_layer(self, nn_module: nn.Module):
        if self.train_config.freeze_embeddings:
            embedding_module = nn_module.bert.embeddings
            for param in embedding_module.parameters():
                param.requires_grad = False
