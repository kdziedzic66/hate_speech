import os
from dataclasses import asdict
from typing import Dict, Union

import dacite
import numpy as np
import torch.nn as nn

import utils.pytorch as pt_utils
from data.dataloader import DataLoader
from model_training.config import TrainConfig
from model_training.trainer import Trainer
from pipeline_config import PipelineConfig
from pipeline_steps.bert_classifier_module import BertHateClassifier
from pipeline_steps.text_cleaning import TextCleaningComposer
from pipeline_steps.text_encoding import TextEncoder
from utils.constants import CLASS_NAME_MAPPING
from utils.helpers import get_repo_root, load_json, save_json


class Pipeline:
    def __init__(self, pipeline_config: PipelineConfig):
        self.pipeline_config = pipeline_config
        self.bert_classifier = BertHateClassifier()
        self.text_cleaner = TextCleaningComposer(
            cleaner_names=self.pipeline_config.text_cleaners
        )
        self.text_encoder = TextEncoder(max_seq_len=pipeline_config.max_seq_len)
        if self.pipeline_config.trained_model_name is not None:
            frac_restored = pt_utils.restore_weights_greedy(
                nn_module=self.bert_classifier,
                checkpoint_path=os.path.join(
                    get_repo_root(),
                    "trained_models",
                    self.pipeline_config.trained_model_name,
                    "weights.pt",
                ),
            )
            assert frac_restored == 1.0, "Some weights mismatch!"

    @classmethod
    def from_model_id(cls, model_id: str) -> "Pipeline":
        pipeline_config = dacite.from_dict(
            PipelineConfig,
            data=load_json(
                os.path.join(
                    get_repo_root(), "trained_models", model_id, "pipeline_config.json"
                )
            ),
        )
        return cls(pipeline_config=pipeline_config)

    def predict(self, text: str) -> Dict[str, Union[float, str]]:
        text = self.text_cleaner.clean(text=text)
        text_encoding = self.text_encoder.encode(text=text)
        prediction_logits = self.bert_classifier(
            text_encoding["input_ids"], text_encoding["attention_mask"]
        )
        prediction_softmax = nn.Softmax(dim=1)(prediction_logits)
        prediction_softmax = predictions.detach().numpy()
        prediction_class_id = np.argmax(predictions, axis=1)
        confidence = prediction_softmax[0][prediction_class_id]
        prediction = {
            "harmfulness": CLASS_NAME_MAPPING[prediction_class_id],
            "confidence": confidence,
        }
        return prediction

    def train(self, train_config: TrainConfig):
        dataloader_train = DataLoader(
            data_type="train",
            text_cleaner=self.text_cleaner,
            text_encoder=self.text_encoder,
            batch_size=train_config.batch_size,
            shuffle=True,
            class_balanced_sampling=train_config.class_balanced_sampling,
        )

        dataloader_valid = DataLoader(
            data_type="valid",
            text_cleaner=self.text_cleaner,
            text_encoder=self.text_encoder,
            batch_size=train_config.batch_size,
        )

        dataloader_test = DataLoader(
            data_type="test",
            text_cleaner=self.text_cleaner,
            text_encoder=self.text_encoder,
            batch_size=train_config.batch_size,
        )
        dataloaders = {
            "train": dataloader_train,
            "valid": dataloader_valid,
            "test": dataloader_test,
        }

        trainer = Trainer(train_config=train_config)
        trainer.train(nn_module=self.bert_classifier, dataloaders=dataloaders)
        self.pipeline_config.trained_model_name = train_config.output_model_name
        save_json(
            data=asdict(self.pipeline_config),
            filepath=os.path.join(
                get_repo_root(),
                "trained_models",
                train_config.output_model_name,
                "pipeline_config.json",
            ),
        )
