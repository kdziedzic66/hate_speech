from dataclasses import dataclass, field

from typing import List


@dataclass(repr=False)
class _OptimizationSchedule:
    optimizer_name: str = "adam"
    init_lr: float = 1e-3
    gamma: float = 1e-1
    weight_decay: float = 0
    milestones: List[float] = field(default_factory=lambda: [0.4, 0.7, 0.9])

    def __post_init__(self):
        assert self.init_lr > 0 and self.gamma > 0 and self.weight_decay >= 0
        self.milestones = sorted(set(self.milestones))
        assert len(self.milestones) > 0 and all(
            [0 < milestone < 1 for milestone in self.milestones]
        )


@dataclass(repr=False)
class TrainConfig:
    batch_size: int
    num_epochs: int
    optimization_schedule: _OptimizationSchedule = field(default_factory=_OptimizationSchedule)