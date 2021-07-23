from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from data.dataset import HateSpeechDataset
from pipeline_steps.text_cleaning import TextCleaningComposer
from pipeline_steps.text_encoding import TextEncoder


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        data_type: str,
        text_cleaner: TextCleaningComposer,
        text_encoder: TextEncoder,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = True,
    ):
        dataset = HateSpeechDataset(data_type=data_type, text_cleaner=text_cleaner)
        super(DataLoader, self).__init__(
            dataset=dataset,
            collate_fn=_collate_fn(text_encoder=text_encoder),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False,
        )


def _collate_fn(text_encoder: TextEncoder) -> Callable:
    def _make_batch(datapoints: List[Tuple[str, int]]) -> Dict[str, torch.Tensor]:
        attention_masks = []
        input_ids = []
        labels = []
        for text, label in datapoints:
            encoding = text_encoder.encode(text=text)
            attention_masks.append(encoding["attention_mask"])
            input_ids.append(encoding["input_ids"])
            labels.append(label)
        batch = {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_masks": torch.cat(attention_masks, dim=0),
            "targets": torch.from_numpy(np.array(labels)),
        }
        return batch

    return _make_batch
