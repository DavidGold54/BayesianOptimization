from functools import singledispatchmethod
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from schedulefree import RAdamScheduleFree

from model import Model


# Base -----------------------------------------------------------------------
class Acquisition(ABC):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    @singledispatchmethod
    def optimize(self, bounds: Tensor, restarts: int = 10) -> Tensor:
        sobol = SobolEngine(bounds.shape[1], scramble=True)
        samples = sobol.draw(restarts)
        samples = samples * (bounds[1] - bounds[0]) + bounds[0]
        for x in samples:
            x.requires_grad = True
            optimizer = RAdamScheduleFree(x)
            optimizer.train()
            for _ in range(self.kwargs['max_iter']):
                optimizer.zero_grad()
                loss = -self(x)
                loss.backward()
                optimizer.step()
            x.requires_grad = False
        return x
    
    @optimize.register
    def _(self, grid: Tensor) -> Tensor:
        acq = self(grid)
        x = grid[acq.argmax()]
        return x


# Acquisition Functions ======================================================
class ExpectedImprovement(Acquisition):
    def __init__(self, model: Model, current_best: Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.current_best = current_best
        self.current_best.requires_grad = True

    def __call__(self, x: Tensor) -> Tensor:
        mean, std = self.model.predict(x)
        mean.requires_grad = True
        std.requires_grad = True
        z = (mean - self.current_best) / std
        norm = torch.distributions.Normal(0, 1)
        alpha = (mean - self.current_best) * norm.cdf(z) + std * torch.exp(norm.log_prob(z))
        return alpha