from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def restore_weights_greedy(nn_module: torch.nn.Module, checkpoint_path: str) -> float:
    trained_state_dict = torch.load(checkpoint_path, map_location="cpu")
    return restore_weights_greedy_from_state_dict(
        nn_module=nn_module, trained_state_dict=trained_state_dict
    )


def restore_weights_greedy_from_state_dict(
    nn_module: torch.nn.Module, trained_state_dict: Dict[str, torch.Tensor]
) -> float:
    new_state_dict = {}
    initialized_from_ckpt = 0
    reshaped = 0
    num_params = len(nn_module.state_dict())
    for param_name in nn_module.state_dict():
        param = nn_module.state_dict()[param_name]
        if param_name in trained_state_dict:
            if (
                trained_state_dict[param_name].shape
                == nn_module.state_dict()[param_name].shape
            ):
                param = trained_state_dict[param_name]
                initialized_from_ckpt += 1
            else:
                try:
                    param = reshape_mismatched_shape_tensor(
                        trained_state_dict[param_name], param.shape
                    )
                    print(
                        f"{repr(param_name)} reshaped from {trained_state_dict[param_name].shape} to {param.shape}"
                    )
                    reshaped += 1
                except ValueError:
                    pass
        new_state_dict[param_name] = param
    nn_module.load_state_dict(new_state_dict)
    print(
        f"{initialized_from_ckpt} of {num_params} initialized straightly from checkpoint, {reshaped} have been reshaped"
    )
    init_proc = initialized_from_ckpt / num_params
    return init_proc


def reshape_mismatched_shape_tensor(
    input_tensor: torch.tensor, output_shape: tuple
) -> torch.tensor:
    shape_diff = np.subtract(output_shape, input_tensor.shape)
    mismatched_dimensions = np.count_nonzero(shape_diff)
    if mismatched_dimensions != 1:
        raise ValueError(f"Shape mismatch in {repr(mismatched_dimensions)} dimensions!")
    else:
        mismatch_axis = int(np.nonzero(shape_diff)[0][0])
        if input_tensor.shape[mismatch_axis] < output_shape[mismatch_axis]:
            dim_difference = (
                output_shape[mismatch_axis] - input_tensor.shape[mismatch_axis]
            )
            input_tensor_slice = torch.narrow(
                input_tensor, mismatch_axis, 0, dim_difference
            )
            reshaped_tensor = torch.cat(
                (input_tensor, input_tensor_slice), dim=mismatch_axis
            )
        else:
            reshaped_tensor = torch.narrow(
                input_tensor, mismatch_axis, 0, output_shape[mismatch_axis]
            )
        assert reshaped_tensor.shape == output_shape
        return reshaped_tensor


def create_optimizer(
    params, optimizer_name: str, init_lr: float, weight_decay: float = 0
) -> optim.Optimizer:
    name_to_optimizer = {"sgd": optim.SGD, "adam": optim.Adam}
    if optimizer_name not in name_to_optimizer:
        raise ValueError(
            f"Unknown optimizer name. Must be one of {repr(set(name_to_optimizer))}."
        )
    return name_to_optimizer[optimizer_name](
        params=params, lr=init_lr, weight_decay=weight_decay
    )


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    num_iterations: int,
    gamma: float = 1,
    milestones: Optional[List[float]] = None,
):
    milestones = milestones or [1]
    milestones = [round(milestone * num_iterations) for milestone in sorted(milestones)]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=milestones, gamma=gamma
    )
    return scheduler


def freeze_weights(nn_module: nn.Module) -> None:
    for tensor in nn_module.parameters(recurse=True):
        tensor.requires_grad = False


def unfreeze_weights(nn_module: nn.Module) -> None:
    for tensor in nn_module.parameters(recurse=True):
        tensor.requires_grad = True


def to_device(unit: Union[torch.Tensor, dict], device: str) -> None:
    if isinstance(unit, torch.Tensor):
        unit = unit.to(device)
        return unit
    elif isinstance(unit, dict):
        return {key: to_device(unit[key], device) for key in unit}
