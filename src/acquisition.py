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
    
    def __str__(self) -> str:
        return 'ACQUISITION:\n'
    
    def optimize_query(self, bounds: Tensor, restarts: int = 10) -> Tensor:
        sobol = SobolEngine(bounds.shape[1], scramble=True)
        samples = sobol.draw(restarts)
        samples = samples * (bounds[1] - bounds[0]) + bounds[0]
        best_x = torch.empty(0)
        best_y = -torch.inf
        for x in samples:
            x.requires_grad = True
            optimizer = RAdamScheduleFree([x])
            optimizer.train()
            for i in range(self.kwargs['max_iter']):
                optimizer.zero_grad()
                loss = -self(x)
                loss.backward()
                optimizer.step()
            x.requires_grad = False
            if self(x) > best_y:
                best_x = x
                best_y = self(x)
        return best_x
    
    def optimize_pool(self, grid: Tensor) -> Tensor:
        acq = self(grid)
        x = grid[acq.argmax()]
        return x


# Acquisition Functions ======================================================
class RandomSearch(Acquisition):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, x: Tensor) -> Tensor:
        return x
    
    def __str__(self) -> str:
        prefix = super().__str__()
        base = f'+NAME: {self.__class__.__name__}\n'
        return prefix + base

    def optimize_query(self, bounds: Tensor, restarts: int = 10) -> Tensor:
        return torch.rand(1, bounds.shape[1]) * (bounds[1] - bounds[0]) + bounds[0]
    
    def optimize_pool(self, grid: Tensor) -> Tensor:
        return grid[torch.randint(0, grid.shape[0], (1,))]
    

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
        z = (mean - self.current_best - self.kwargs['xi']) / std
        norm = torch.distributions.Normal(0, 1)
        alpha = (mean - self.current_best - self.kwargs['xi']) * norm.cdf(z) + std * torch.exp(norm.log_prob(z))
        return alpha
    
    def __str__(self) -> str:
        prefix = super().__str__()
        base = f'+NAME: {self.__class__.__name__}\n' + \
               f'+XI: {self.kwargs["xi"]}\n'
        return prefix + base
    

class ProbabilityOfImprovement(Acquisition):
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
        alpha = norm.cdf(z)
        return alpha
    
    def __str__(self) -> str:
        prefix = super().__str__()
        base = f'+NAME: {self.__class__.__name__}\n'
        return prefix + base


class UpperConfidenceBound(Acquisition):
    def __init__(self, model: Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model

    def __call__(self, x: Tensor) -> Tensor:
        mean, std = self.model.predict(x)
        mean.requires_grad = True
        std.requires_grad = True
        alpha = mean + self.kwargs['kappa'] * std
        return alpha
    
    def __str__(self) -> str:
        prefix = super().__str__()
        base = f'+NAME: {self.__class__.__name__}\n' + \
               f'+KAPPA: {self.kwargs["kappa"]}\n'
        return prefix + base
