import os
from collections import Counter
from typing import Tuple

import numpy as np
import torch

from pipeline_steps.text_cleaning import TextCleaningComposer
from utils.helpers import broadcast_list_to_type, get_repo_root, read_text_file


class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_type: str, text_cleaner: TextCleaningComposer):
        texts_file = os.path.join(
            get_repo_root(), "datafiles", f"{data_type}_texts.txt"
        )
        labels_file = os.path.join(
            get_repo_root(), "datafiles", f"{data_type}_tags.txt"
        )

        assert os.path.isfile(texts_file), "Text file does not exist!"
        assert os.path.isfile(labels_file), "Label file does not exist!"

        texts = read_text_file(texts_file)
        text_labels = broadcast_list_to_type(read_text_file(labels_file), int)
        assert len(texts) == len(
            text_labels
        ), "texts and labels should have the same length!"
        texts = map(text_cleaner.clean, texts)

        self.texts = list(texts)
        self.text_labels = text_labels

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        text = self.texts[idx]
        label = self.text_labels[idx]
        return text, label

    def __len__(self) -> int:
        return len(self.texts)

    def get_class_balanced_sampler(self) -> torch.utils.data.WeightedRandomSampler:
        classnames_cnt = Counter(self.text_labels)
        class_weights = {}
        for classname, count in classnames_cnt.items():
            class_weights[classname] = 1.0 / np.log1p(count)
        min_class_weight_value = min(class_weights.values())
        for classname, weight in class_weights.items():
            class_weights[classname] = weight / min_class_weight_value
        print(f"sampler class weights = {class_weights}")
        sample_weights = [class_weights[label] for label in self.text_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return sampler
