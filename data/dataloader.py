from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from transformers import BertTokenizer

from data.dataset import HateSpeechDataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        data_type: str,
        batch_size: int,
        tokenizer: BertTokenizer,
        max_seq_len: int,
        num_workers: int = 1,
        shuffle: bool = True,
    ):
        dataset = HateSpeechDataset(data_type=data_type)
        super(DataLoader, self).__init__(
            dataset=dataset,
            collate_fn=_collate_fn(tokenizer=tokenizer, max_seq_len=max_seq_len),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False,
        )


def _collate_fn(tokenizer: BertTokenizer, max_seq_len: int) -> Callable:
    def _make_batch(datapoints: List[Tuple[str, int]]) -> Dict[str, torch.Tensor]:
        attention_masks = []
        input_ids = []
        labels = []
        for text, label in datapoints:
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_seq_len,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
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
