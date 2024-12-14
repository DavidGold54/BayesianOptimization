import time
from pprint import pformat
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from schedulefree import RAdamScheduleFree


# Base -----------------------------------------------------------------------
class BaseAcquisition(ABC):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __str__(self) -> str:
        bar     = '=' * 79 + '\n'
        title   = f'🔍 Acquisition Function: {self.__class__.__name__}\n'
        divider = '-' * 79 + '\n'
        return bar + title + divider
    
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def optimize_query(self) -> Tensor:
        lower_bounds = torch.tensor([self.kwargs['lower_bounds']])
        upper_bounds = torch.tensor([self.kwargs['upper_bounds']])
        bounds = upper_bounds - lower_bounds
        sobol = SobolEngine(lower_bounds.shape[-1], scramble=True)
        samples = sobol.draw(self.kwargs['restarts'])
        samples = samples * bounds + lower_bounds
        best_x = torch.empty(0)
        best_loss = torch.inf
        start_log = f'\n🔍 Optimizing Acquisition Function (Query) 🔍\n' + \
                    f'-' * 79 + '\n' + \
                    f'Lower Bounds:     {pformat(self.kwargs["lower_bounds"])}\n' + \
                    f'Upper Bounds:     {pformat(self.kwargs["upper_bounds"])}\n' + \
                    f'Restarts:         {self.kwargs["restarts"]}\n' + \
                    f'Max Iterations:   {self.kwargs["max_iter"]}\n' + \
                    f'Optimizer:        RAdamScheduleFree\n' + \
                    f'-' * 79 + '\n'
        if self.kwargs['logger'] is not None:
            self.kwargs['logger'].info(self)
            self.kwargs['logger'].info(start_log)
        else:
            print(self)
            print(start_log)
        start_time = time.perf_counter()
        for i, sample in enumerate(samples):
            log = f'Restart {i+1}/{self.kwargs["restarts"]} : Initial Point {sample.item()}'
            if self.kwargs['logger'] is not None:
                self.kwargs['logger'].info(log)
            else:
                print(log)
            x = sample.clone().requires_grad_(True)
            optimizer = RAdamScheduleFree([x])
            optimizer.train()
            for j in range(self.kwargs['max_iter']):
                optimizer.zero_grad()
                loss = -self(x)
                loss.backward()
                optimizer.step()
                if (j + 1) % 50 == 0:
                    log = f'[Iter {j+1}/{self.kwargs["max_iter"]}]' + \
                          f' - Loss: {loss.item():.4f}'
                    if self.kwargs['logger'] is not None:
                        self.kwargs['logger'].info(log)
                    else:
                        print(log)
            is_in_bounds = (x >= lower_bounds).all() and (x <= upper_bounds).all()
            if loss < best_loss and is_in_bounds:
                best_x = x.clone()
                best_loss = loss.item()
            best_x = best_x.detach()
        end_time = time.perf_counter()
        end_log = f'\n🏁 Optimizing Acquisition Function (Query) Completed 🏁\n' + \
                  f'-' * 79 + '\n' + \
                  f'Best Point:         {best_x}\n' + \
                  f'Best Acquisition:   {-best_loss}\n' + \
                  f'Total Time:         {end_time - start_time:.2f}s\n' + \
                  f'-' * 79 + '\n'
        if self.kwargs['logger'] is not None:
            self.kwargs['logger'].info(end_log)
            self.kwargs['logger'].info(self)
        else:
            print(end_log)
            print(self)
        return best_x
    
    def optimize_pool(self, pool: Tensor) -> Tensor:
        start_log = f'\n🔍 Optimizing Acquisition Function (Pool) 🔍\n' + \
                    f'-' * 79 + '\n' + \
                    f'Lower Bounds:     {pformat(self.kwargs["lower_bounds"])}\n' + \
                    f'Upper Bounds:     {pformat(self.kwargs["upper_bounds"])}\n' + \
                    f'Pool Size:        {pool.shape[0]}\n' + \
                    f'-' * 79 + '\n'
        if self.kwargs['logger'] is not None:
            self.kwargs['logger'].info(self)
            self.kwargs['logger'].info(start_log)
        else:
            print(self)
            print(start_log)
        start_time = time.perf_counter()
        values = self(pool)
        best_x = pool[values.argmax()]
        end_time = time.perf_counter()
        end_log = f'\n🏁 Optimizing Acquisition Function (Pool) Completed 🏁\n' + \
                  f'-' * 79 + '\n' + \
                  f'Best Point:         {best_x}\n' + \
                  f'Best Acquisition:   {values.max().item()}\n' + \
                  f'Total Time:         {end_time - start_time:.2f}s\n' + \
                  f'-' * 79 + '\n'
        if self.kwargs['logger'] is not None:
            self.kwargs['logger'].info(end_log)
            self.kwargs['logger'].info(self)
        else:
            print(end_log)
            print(self)
        return best_x
    
    def optimize(self, pool: Tensor = None) -> Tensor:
        if pool is None:
            return self.optimize_query()
        else:
            return self.optimize_pool(pool)
    

# Wrapper ====================================================================
class Acquisition:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.set_acquisition()

    def __str__(self) -> str:
        return self.acquisition.__str__()
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.acquisition(x)
    
    def optimize_query(self) -> Tensor:
        return self.acquisition.optimize_query()
    
    def optimize_pool(self, pool: Tensor) -> Tensor:
        return self.acquisition.optimize_pool(pool)
    
    def optimize(self, pool: Tensor = None) -> Tensor:
        return self.acquisition.optimize(pool)
    
    def set_acquisition(self) -> None:
        if self.kwargs['name'] == 'RandomSearch':
            self.acquisition = RandomSearch(**self.kwargs)
        elif self.kwargs['name'] == 'ExpectedImprovement':
            self.acquisition = ExpectedImprovement(**self.kwargs)
        elif self.kwargs['name'] == 'ProbabilityOfImprovement':
            self.acquisition = ProbabilityOfImprovement(**self.kwargs)
        elif self.kwargs['name'] == 'UpperConfidenceBound':
            self.acquisition = UpperConfidenceBound(**self.kwargs)
        else:
            raise ValueError(f'Acquisition {self.kwargs["name"]} not found')


# Acquisition Functions ======================================================
class RandomSearch(BaseAcquisition):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self) -> str:
        prefix = super().__str__()
        suffix = '=' * 79 + '\n'
        return prefix + suffix

    def __call__(self, x: Tensor) -> Tensor:
        return torch.rand(x.shape)
    
    def optimize_query(self) -> Tensor:
        lower_bounds = torch.tensor([self.kwargs['lower_bounds']])
        upper_bounds = torch.tensor([self.kwargs['upper_bounds']])
        bounds = upper_bounds - lower_bounds
        return torch.rand(1, lower_bounds.shape[-1]) * bounds + lower_bounds
    
    def optimize_pool(self, pool: Tensor) -> Tensor:
        return pool[torch.randint(0, pool.shape[0], (1,))]


class ExpectedImprovement(BaseAcquisition):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = self.kwargs['model']
        self.current_best = self.kwargs['current_best']

    def __str__(self) -> str:
        prefix = super().__str__()
        base = f'- Xi: {self.kwargs["xi"]}\n'
        suffix = '=' * 79 + '\n'
        return prefix + base + suffix

    def __call__(self, x: Tensor) -> Tensor:
        mean, std = self.model.predict(x)
        xi = torch.tensor([self.kwargs['xi']])
        z = (mean - self.current_best - xi) / std
        norm = torch.distributions.Normal(0, 1)
        alpha = (mean - self.current_best - xi) * norm.cdf(z) + std * torch.exp(norm.log_prob(z))
        return alpha


class ProbabilityOfImprovement(BaseAcquisition):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = self.kwargs['model']
        self.current_best = self.kwargs['current_best']

    def __str__(self) -> str:
        prefix = super().__str__()
        suffix = '=' * 79 + '\n'
        return prefix + suffix
    
    def __call__(self, x: Tensor) -> Tensor:
        mean, std = self.model.predict(x)
        z = (mean - self.current_best) / std
        norm = torch.distributions.Normal(0, 1)
        alpha = norm.cdf(z)
        return alpha


class UpperConfidenceBound(BaseAcquisition):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = self.kwargs['model']

    def __str__(self) -> str:
        prefix = super().__str__()
        base = f'- Kappa: {self.kwargs["kappa"]}\n'
        suffix = '=' * 79 + '\n'
        return prefix + base + suffix
    
    def __call__(self, x: Tensor) -> Tensor:
        mean, std = self.model.predict(x)
        alpha = mean + self.kwargs['kappa'] * std
        return alpha
