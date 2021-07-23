import dacite
from sklearn.metrics import classification_report

from data.dataloader import DataLoader
from model_training.config import TrainConfig
from model_training.trainer import Trainer
from pipeline_steps.bert_classifier_module import BertHateClassifier
from pipeline_steps.text_cleaning import TextCleaningComposer
from pipeline_steps.text_encoding import TextEncoder

text_cleaner = TextCleaningComposer(cleaner_names=["UsernameRemover"])

model = BertHateClassifier()
for param in model.parameters():
    a = "dupa"
text_encoder = TextEncoder(max_seq_len=64)

dataloader_train = DataLoader(
    data_type="train",
    text_cleaner=text_cleaner,
    text_encoder=text_encoder,
    batch_size=32,
    class_balanced_sampling=True
)
dataloader_valid = DataLoader(
    data_type="valid",
    text_cleaner=text_cleaner,
    text_encoder=text_encoder,
    batch_size=32,
)
dataloaders = {"train": dataloader_train, "valid": dataloader_valid}

model = BertHateClassifier()
train_config = {"batch_size": 32, "num_epochs": 50}
train_config = dacite.from_dict(TrainConfig, train_config)
trainer = Trainer(train_config=train_config)
trainer.train(model, dataloaders)
