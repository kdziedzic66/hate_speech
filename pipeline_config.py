from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(repr=False)
class PipelineConfig:
    max_seq_len: int = 64
    text_cleaners: List[str] = field(default_factory=list)
    trained_model_name: Optional[str] = None
