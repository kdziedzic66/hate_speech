from typing import Callable

import torch
from transformers import BertTokenizer

from data.dataset import HateSpeechDataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        batch_size: int,
        tokenizer: BertTokenizer,
        max_seq_len: int,
        num_workers: int = 1,
        shuffle: bool = True,
    ):
        dataset = HateSpeechDataset()
        print(len(dataset))
        super(DataLoader, self).__init__(
            dataset=dataset,
            collate_fn=_collate_fn(tokenizer=tokenizer, max_seq_len=max_seq_len),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False,
        )


def _collate_fn(tokenizer: BertTokenizer, max_seq_len: int) -> Callable:
    def _make_batch(datapoints) -> dict:
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
            "input_ids": torch.cat(input_ids, axis=0),
            "attention_masks": torch.cat(attention_masks, axis=0),
            "targets": torch.from_numpy(np.array(labels)),
        }
        return batch

    return _make_batch
