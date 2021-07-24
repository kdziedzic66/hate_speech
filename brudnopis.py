import dacite

from model_training.config import TrainConfig
from pipeline_config import PipelineConfig
from pipeline import Pipeline



text_cleaner = ["UsernameRemover"]

pipeline_config = {"text_cleaners": text_cleaner}
train_config = {"batch_size": 32, "num_epochs": 10}

pipeline_config = dacite.from_dict(PipelineConfig, pipeline_config)
train_config = dacite.from_dict(TrainConfig, train_config)

pipeline = Pipeline(pipeline_config=pipeline_config)
pipeline.train(train_config=train_config)



