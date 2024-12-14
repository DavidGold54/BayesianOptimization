import time
from pprint import pformat
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from schedulefree import RAdamScheduleFree
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP


# Base -----------------------------------------------------------------------
class BaseModel(ABC):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.GP = None

    def __str__(self) -> str:
        bar         = '\n' + '=' * 79 + '\n'
        title       = f'🤖 Model: {self.__class__.__name__}\n'
        divider     = '-' * 79 + '\n'
        return bar + title + divider
    
    @abstractmethod
    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, test_x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, test_x: Tensor, n: int) -> Tensor:
        raise NotImplementedError
    

# Wrapper ====================================================================
class Model:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.set_model()

    def __str__(self) -> str:
        return self.model.__str__()
    
    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        self.model.fit(train_x, train_y)

    def predict(self, test_x: Tensor) -> tuple[Tensor, Tensor]:
        return self.model.predict(test_x)
    
    def sample(self, test_x: Tensor, n: int) -> Tensor:
        return self.model.sample(test_x, n)
    
    def set_model(self) -> None:
        if self.kwargs['name'] == 'SimpleGP':
            self.model = SimpleGPModel(**self.kwargs)
        else:
            raise ValueError(f'Model {self.kwargs["name"]} not found')


# Models =====================================================================
class SimpleGPModel(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(self) -> str:
        prefix = super().__str__()
        lengthscale = self.GP.covar_module.base_kernel.lengthscale
        outputscale = self.GP.covar_module.outputscale
        noise = self.GP.likelihood.noise_covar.noise
        mean = self.GP.mean_module.constant
        base = f'Input Dimension:   {self.GP.train_inputs[0].shape[1]}\n' + \
               f'Training Data:     {self.GP.train_inputs[0].shape[0]}\n' + \
               f'Kernel:\n' + \
               f'{self.GP.covar_module.__str__()}\n\n' + \
               f'🔧 Hyperparameters:\n' + \
               f'- Length Scale:    {pformat(lengthscale.tolist())}\n' + \
               f'- Signal Variance: {pformat(outputscale.tolist())}\n' + \
               f'- Noise Variance:  {pformat(noise.tolist())}\n' + \
               f'- Mean Constant:   {pformat(mean.tolist())}\n' + \
               f'=' * 79 + '\n'
        return prefix + base
    
    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        likelihood = GaussianLikelihood(
            noise_constraint=Interval(**self.kwargs['noise_constraint'])
        )
        model = SimpleGP(train_x, train_y, likelihood, **self.kwargs)
        mll = ExactMarginalLogLikelihood(likelihood, model)
        optimizer = RAdamScheduleFree(model.parameters())
        start_log = f'Training of {self.__class__.__name__} initiated.\n' + \
                    f'🚀 Training Initialization 🚀\n' + \
                    f'-' * 79 + '\n' + \
                    f'Model:             {self.__class__.__name__}\n' + \
                    f'Max Iterations:    {self.kwargs["max_iter"]}\n' + \
                    f'Optimizer:         RAdamScheduleFree\n' + \
                    f'Learning Rate:     {optimizer.defaults["lr"]}\n' + \
                    f'Parameters:        {len(list(model.parameters()))}\n' + \
                    f'-' * 79
        if self.kwargs['logger'] is not None:
            self.kwargs['logger'].info(start_log)
        else:
            print(start_log)
        start_time = time.perf_counter()
        model.train(); likelihood.train(); optimizer.train()
        for i in range(self.kwargs['max_iter']):
            optimizer.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            optimizer.step()
            if (i + 1) % 50 == 0:
                log = f'[Iter {i+1}/{self.kwargs["max_iter"]}]' + \
                      f' - Loss: {loss.item():.4f}'
                if self.kwargs['logger'] is not None:
                    self.kwargs['logger'].info(log)
                else:
                    print(log)
        self.GP = model
        end_time = time.perf_counter()
        end_log = f'Training of {self.__class__.__name__} completed.\n' + \
                  f'🏁 Training Completed 🏁\n' + \
                  f'-' * 79 + '\n' + \
                  f'Best Loss: {loss.item()}\n' + \
                  f'Total Training Time: {end_time - start_time:.2f}s\n' + \
                  f'-' * 79 + '\n'
        if self.kwargs['logger'] is not None:
            self.kwargs['logger'].info(end_log)
        else:
            print(end_log)

    def predict(self, test_x: Tensor) -> tuple[Tensor, Tensor]:
        self.GP.eval()
        with gpytorch.settings.fast_pred_var():
            pred = self.GP(test_x)
            mean, std = pred.mean, pred.stddev
        return mean, std
    
    def sample(self, test_x: Tensor, n: int) -> Tensor:
        self.GP.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_samples():
            pred = self.GP(test_x)
            samples = pred.sample(n)
        return samples


# GPs ------------------------------------------------------------------------
class SimpleGP(ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: Likelihood, **kwargs) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(
                ard_num_dims=train_x.shape[1],
                lengthscale_constraint=Interval(**kwargs['lengthscale_constraint'])
            ),
            outputscale_constraint=Interval(**kwargs['outputscale_constraint'])
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
