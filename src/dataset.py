import torch
from torch import Tensor


# Dataset ====================================================================
class Dataset:
    def __init__(self, **kwargs) -> None:
        self.inputs = torch.empty(0)
        self.targets = torch.empty(0)
        self.kwargs = kwargs
        self.best_inputs = torch.empty(0)
        self.best_targets = torch.empty(0)

    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.inputs[index], self.targets[index]

    def add(self, x: Tensor, y: Tensor) -> None:
        x = torch.atleast_2d(x)
        y = y.flatten()
        if self.inputs.numel() == 0:
            self.inputs = x
            self.targets = y
        else:
            self.inputs = torch.cat([self.inputs, x], dim=0)
            self.targets = torch.cat([self.targets, y], dim=0)
        self.update_best(x, y)

    def get(self, standardize: bool = False) -> tuple[Tensor, ...]:
        if standardize:
            targets = (self.targets - self.targets.mean()) / self.targets.std()
        else:
            targets = self.targets
        return self.inputs, targets

    def update_best(self, x: Tensor, y: Tensor) -> None:
        x = torch.atleast_2d(x)
        y = y.flatten()
        if self.best_inputs.numel() == 0 or y.max() > self.best_targets.max():
            self.best_inputs = x[y.argmax(), :]
            self.best_targets = y.max()

    def get_best(self) -> tuple[Tensor, Tensor]:
        return self.best_inputs, self.best_targets
    
    def save(self, path: str) -> None:
        dataset_dict = {
            'inputs': self.inputs,
            'targets': self.targets,
            'best_inputs': self.best_inputs,
            'best_targets': self.best_targets,
            'kwargs': self.kwargs
        }
        torch.save(dataset_dict, path)

    def load(self, path: str) -> None:
        dataset_dict = torch.load(path)
        self.inputs = dataset_dict['inputs']
        self.targets = dataset_dict['targets']
        self.best_inputs = dataset_dict['best_inputs']
        self.best_targets = dataset_dict['best_targets']
        self.kwargs = dataset_dict['kwargs']