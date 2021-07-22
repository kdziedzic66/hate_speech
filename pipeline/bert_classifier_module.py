import torch
import torch.nn as nn
from transformers import BertModel

from utils.constants import POLBERT_PRETRAINED_ID, NUM_HATE_CLASSES


class BertHateClassifier(nn.Module):
    def __init__(self):
        super(BertHateClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(POLBERT_PRETRAINED_ID, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, NUM_HATE_CLASSES)

    def forward(self, input_ids: torch.Tensor, attention_masks: torch.Tensor):
        output = self.bert(input_ids, attention_masks).pooler_output
        output = self.classifier(output)
        return output
