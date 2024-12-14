from pprint import pformat
from abc import ABC, abstractmethod

import torch
from torch import Tensor


# Base -----------------------------------------------------------------------
class BaseObjective(ABC):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.evaluation_count = 0

    def __str__(self) -> str:
        bar         = '=' * 79 + '\n'
        title       = f'🎯 Objective Function: {self.__class__.__name__}\n'
        divider     = '-' * 79 + '\n'
        base        = f'- Dimension:       {len(self.kwargs["lower_bounds"])}\n' + \
                      f'- Lower bounds:    {pformat(self.kwargs["lower_bounds"], compact=True)}\n' + \
                      f'- Upper bounds:    {pformat(self.kwargs["upper_bounds"], compact=True)}\n' + \
                      f'- Noise stddev:    {self.kwargs["noise_std"]}\n' + \
                      f'- Coefficients A:  {self.kwargs["A"]}\n'
        prefix      = '\n' + bar + title + divider + base
        return prefix
    
    def __call__(self, x: Tensor) -> Tensor:
        self.validate(x)
        values = self.evaluate(x)
        if self.kwargs['noise_std'] > 0.0:
            values += torch.randn_like(values) * self.kwargs['noise_std']
        self.evaluation_count += values.shape[0]
        return values

    def reset_evaluation_count(self) -> None:
        self.evaluation_count = 0

    def sample(self, n: int) -> Tensor:
        dim = len(self.kwargs['lower_bounds'])
        lower_bounds = torch.tensor([self.kwargs['lower_bounds']])
        upper_bounds = torch.tensor([self.kwargs['upper_bounds']])
        bounds = upper_bounds - lower_bounds
        samples = torch.rand(n, dim) * bounds + lower_bounds
        return samples
    
    def validate(self, x: Tensor) -> None:
        lower_bounds = torch.tensor([self.kwargs['lower_bounds']])
        upper_bounds = torch.tensor([self.kwargs['upper_bounds']])
        assert (lower_bounds <= x).all() and (x <= upper_bounds).all()

    @abstractmethod
    def evaluate(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# Wrapper ====================================================================
class Objective:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.set_objective()

    def __str__(self) -> str:
        return self.objective.__str__()
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.objective(x)
    
    def reset_evaluation_count(self) -> None:
        self.objective.reset_evaluation_count()
    
    def sample(self, n: int) -> Tensor:
        return self.objective.sample(n)
    
    def validate(self, x: Tensor) -> None:
        self.objective.validate(x)
    
    def evaluate(self, x: Tensor) -> Tensor:
        return self.objective.evaluate(x)
    
    def set_objective(self) -> None:
        if self.kwargs['name'] == 'forrester':
            self.objective = Forrester(**self.kwargs)
        elif self.kwargs['name'] == 'branin':
            self.objective = Branin(**self.kwargs)
        elif self.kwargs['name'] == 'goldstein_price':
            self.objective = GoldSteinPrice(**self.kwargs)
        elif self.kwargs['name'] == 'logarithmic_goldstein_price':
            self.objective = LogarithmicGoldsteinPrice(**self.kwargs)
        else:
            raise ValueError(f'Objective {self.kwargs["name"]} not found')


# Objective Functions ========================================================
class Forrester(BaseObjective):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self) -> str:
        prefix = super().__str__()
        base = f'- Coefficients B:  {self.kwargs["B"]}\n' + \
               f'- Coefficients C:  {self.kwargs["C"]}\n'
        suffix = '=' * 79 + '\n'
        return prefix + base + suffix

    def evaluate(self, x: Tensor) -> Tensor:
        A = self.kwargs['A']
        B = self.kwargs['B']
        C = self.kwargs['C']
        f = (6 * x - 2) ** 2 * torch.sin(12 * x - 4)
        return A * f + B * (x - 0.5) + C


class Branin(BaseObjective):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self) -> str:
        prefix = super().__str__()
        suffix = '=' * 79 + '\n'
        return prefix + suffix

    def evaluate(self, x: Tensor) -> Tensor:
        A = self.kwargs['A']
        a = 1.0
        b = 5.1 / (4 * torch.pi ** 2)
        c = 5.0 / torch.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8 * torch.pi)
        f1 = a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2
        f2 = s * (1 - t) * torch.cos(x[:, 0])
        f3 = s
        return A * (f1 + f2 + f3)


class GoldSteinPrice(BaseObjective):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self) -> str:
        prefix = super().__str__()
        suffix = '=' * 79 + '\n'
        return prefix + suffix

    def evaluate(self, x: Tensor) -> Tensor:
        A = self.kwargs['A']
        f1 = (x[:, 0] + x[:, 1] + 1) ** 2
        f2 = 19 - 14 * x[:, 0] + 3 * x[:, 0] ** 2 - \
             14 * x[:, 1] + 6 * x[:, 0] * x[:, 1] + 3 * x[:, 1] ** 2
        f3 = (2 * x[:, 0] - 3 * x[:, 1]) ** 2
        f4 = 18 - 32 * x[:, 0] + 12 * x[:, 0] ** 2 + \
             48 * x[:, 1] - 36 * x[:, 0] * x[:, 1] + 27 * x[:, 1] ** 2
        return A * ((1 + f1 * f2) * (30 + f3 * f4))
    

class LogarithmicGoldsteinPrice(BaseObjective):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self) -> str:
        prefix = super().__str__()
        suffix = '=' * 79 + '\n'
        return prefix + suffix
    
    def evaluate(self, x: Tensor) -> Tensor:
        x = 4 * x - 2
        A = self.kwargs['A']
        f1 = (x[:, 0] + x[:, 1] + 1) ** 2
        f2 = 19 - 14 * x[:, 0] + 3 * x[:, 0] ** 2 - \
             14 * x[:, 1] + 6 * x[:, 0] * x[:, 1] + 3 * x[:, 1] ** 2
        f3 = (2 * x[:, 0] - 3 * x[:, 1]) ** 2
        f4 = 18 - 32 * x[:, 0] + 12 * x[:, 0] ** 2 + \
             48 * x[:, 1] - 36 * x[:, 0] * x[:, 1] + 27 * x[:, 1] ** 2
        return A * (torch.log((1 + f1 * f2) * (30 + f3 * f4)) - 8.693) / 2.427
