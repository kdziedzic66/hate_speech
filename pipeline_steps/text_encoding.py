from typing import Dict

import torch
from transformers import BertTokenizer

from utils.constants import POLBERT_PRETRAINED_ID


class TextEncoder:
    def __init__(self, max_seq_len: int, padding: str = "max_length"):
        self.tokenizer = BertTokenizer.from_pretrained(POLBERT_PRETRAINED_ID)
        self.max_seq_len = max_seq_len
        self.padding = padding

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            return_token_type_ids=False,
            padding=self.padding,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoding
