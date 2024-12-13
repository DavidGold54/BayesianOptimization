from abc import ABC, abstractmethod

import torch
from torch import Tensor


# Base -----------------------------------------------------------------------
class Objective(ABC):
    def __init__(self, upper_bounds: list, lower_bounds: list, noise_std: float = 0.0, **kwargs) -> None:
        self.upper_bounds = torch.tensor(upper_bounds)
        self.lower_bounds = torch.tensor(lower_bounds)
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
    
    def __str__(self) -> str:
        return 'OBJECTIVE:\n'
    
    def sample(self, n: int) -> Tensor:
        dim = self.upper_bounds.shape[-1]
        lower_bounds = self.lower_bounds.unsqueeze(0)
        upper_bounds = self.upper_bounds.unsqueeze(0)
        return torch.rand(n, dim) * (upper_bounds - lower_bounds) + lower_bounds

    def reset(self) -> None:
        self.evaluation_count = 0

    @abstractmethod
    def validate(self, x: Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# Objective Functions ========================================================
class Forrester(Objective):
    def __init__(self, upper_bounds: list, lower_bounds: list, noise_std: float = 0.0, **kwargs) -> None:
        super().__init__(upper_bounds, lower_bounds, noise_std, **kwargs)

    def __str__(self) -> str:
        prefix = super().__str__()
        base = f'+NAME: {self.__class__.__name__}\n' + \
               f'+LOWER_BOUNDS:\n{self.lower_bounds}\n' + \
               f'+UPPER_BOUNDS:\n{self.upper_bounds}\n' + \
               f'+NOISE_STD: {self.noise_std}\n' + \
               f'+EVALUATION_COUNT: {self.evaluation_count}\n'
        return prefix + base

    def validate(self, x: Tensor) -> None:
        assert (self.lower_bounds <= x).all() and (x <= self.upper_bounds).all()

    def evaluate(self, x: Tensor) -> Tensor:
        return (6 * x - 2) ** 2 * torch.sin(12 * x - 4)


class CustomForrester(Forrester):
    def __str__(self) -> str:
        prefix = super().__str__()
        base = f'+A: {self.kwargs["A"]}\n' + \
               f'+B: {self.kwargs["B"]}\n' + \
               f'+C: {self.kwargs["C"]}\n'
        return prefix + base
    
    def evaluate(self, x: Tensor) -> Tensor:
        f = super().evaluate(x)
        return self.kwargs['A'] * f + self.kwargs['B'] * (x - 0.5) + self.kwargs['C']
