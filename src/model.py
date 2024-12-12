from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.optim as optim
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

    def fit(self, train_x: Tensor, train_y: Tensor, **kwargs) -> None:
        likelihood = GaussianLikelihood(
            noise_constraint=Interval(**kwargs['noise_constraint'])
        )
        model = SimpleGP(train_x, train_y, likelihood, **kwargs)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])
        model.train()
        likelihood.train()
        for i in range(kwargs['n_iter']):
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

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)