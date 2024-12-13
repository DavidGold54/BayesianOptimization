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
class Model(ABC):
    def __init__(self) -> None:
        self.GP = None

    def __str__(self) -> str:
        return 'MODEL:\n'

    @abstractmethod
    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, test_x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def sample(self, test_x: Tensor, n: int) -> Tensor:
        raise NotImplementedError


# Models =====================================================================
class SimpleGPModel(Model):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        prefix = super().__str__()
        base = f'+GP: {self.GP.__str__()}\n'
        return prefix + base

    def fit(self, train_x: Tensor, train_y: Tensor, **kwargs) -> None:
        likelihood = GaussianLikelihood(
            noise_constraint=Interval(**kwargs['noise_constraint'])
        )
        model = SimpleGP(train_x, train_y, likelihood, **kwargs)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer = RAdamScheduleFree(model.parameters())
        model.train()
        likelihood.train()
        optimizer.train()
        for i in range(kwargs['max_iter']):
            optimizer.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            optimizer.step()
        self.GP = model

    def predict(self, test_x: Tensor) -> tuple[Tensor, Tensor]:
        self.GP.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
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

    def __str__(self) -> str:
        prefix = f'{self.__class__.__name__}\n'
        base = f'+MEAN_MODULE: {self.mean_module.__str__()}\n' + \
               f'+COVAR_MODULE: {self.covar_module.__str__()}'
        return prefix + base

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    

if __name__ == '__main__':
    model = SimpleGPModel()
    train_x = torch.tensor([[0.0], [1.0]])
    train_y = torch.tensor([0.0, 0.5])
    model.fit(train_x, train_y, max_iter=100, noise_constraint={'lower_bound': 0.0, 'upper_bound': 0.1}, lengthscale_constraint={'lower_bound': 0.0, 'upper_bound': 1.0}, outputscale_constraint={'lower_bound': 0.0, 'upper_bound': 1.0})
    print(model)
