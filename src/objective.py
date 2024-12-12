from abc import ABC, abstractmethod

import torch
from torch import Tensor


# Base -----------------------------------------------------------------------
class Objective(ABC):
    def __init__(self, bounds: Tensor, noise_std: float = 0.0, **kwargs) -> None:
        self.bounds = bounds
        self.noise_std = noise_std
        self.kwargs = kwargs
        self.evaluation_count = 0

    def __call__(self, x: Tensor) -> Tensor:
        self.validate(x)
        value = self.evaluate(x)
        if self.noise_std > 0.0:
            value += torch.randn_like(value) * self.noise_std
        self.evaluation_count += 1
        return value
    
    def sample(self, n: int) -> Tensor:
        dim = self.bounds.shape[1]
        lower_bounds = self.bounds[0].unsqueeze(0)
        upper_bounds = self.bounds[1].unsqueeze(0)
        return torch.rand(n, dim) * (upper_bounds - lower_bounds) + lower_bounds

    def reset(self) -> None:
        self.evaluation_count = 0

    def to_string(self,) -> str:
        return f'OBJECTIVE:\n'

    @abstractmethod
    def validate(self, x: Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# Objective Functions ========================================================
class Forrester(Objective):
    def __init__(self, bounds: Tensor, noise_std: float = 0.0, **kwargs) -> None:
        super().__init__(bounds, noise_std, **kwargs)

    def to_string(self) -> str:
        prefix = super().to_string()
        base = f' + NAME: {self.__class__.__name__}\n' + \
               f' + LOWER_BOUNDS: {self.bounds[0].item()}\n' + \
               f' + UPPER_BOUNDS: {self.bounds[1].item()}\n' + \
               f' + NOISE_STD: {self.noise_std}\n' + \
               f' + EVALUATION_COUNT: {self.evaluation_count}\n'
        return prefix + base

    def validate(self, x: Tensor) -> None:
        assert (self.bounds[0] <= x).all() and (x <= self.bounds[1]).all()

    def evaluate(self, x: Tensor) -> Tensor:
        return (6 * x - 2) ** 2 * torch.sin(12 * x - 4)


class CustomForrester(Forrester):
    def evaluate(self, x: Tensor) -> Tensor:
        f = super().evaluate(x)
        return self.kwargs['A'] * f + self.kwargs['B'] * (x - 0.5) + self.kwargs['C']

    def to_string(self) -> str:
        prefix = super().to_string()
        base = f' + A: {self.kwargs["A"]}\n' + \
               f' + B: {self.kwargs["B"]}\n' + \
               f' + C: {self.kwargs["C"]}\n'
        return prefix + base


if __name__ == '__main__':
    objective = CustomForrester(bounds=torch.tensor([[0.0], [1.0]]), noise_std=0.0, A=1.0, B=2.0, C=3.0)
    print(objective.to_string())
