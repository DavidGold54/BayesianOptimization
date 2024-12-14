import os
from pprint import pformat
from abc import ABC, abstractmethod

from src.objective import Objective
from src.dataset import Dataset
from src.model import Model
from src.acquisition import Acquisition
from src.utils import seed_everything

# Base -----------------------------------------------------------------------
class BOEngine(ABC):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def initialize(self) -> None:
        message = '\n' + '*' * 79 + '\n' + \
                  f'* Task: {self.__class__.__name__}\n' + \
                  '*' * 79 + '\n'
        self.kwargs['logger'].info(message)
    
    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

# Wrapper =====================================================================
class BO:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.set_bo()

    def initialize(self) -> None:
        self.bo.initialize()

    def run(self) -> None:
        self.bo.run()

    def save(self) -> None:
        self.bo.save()

    def set_bo(self) -> None:
        if self.kwargs['task'] == 'SingleTaskBO':
            self.bo = SingleTaskBO(**self.kwargs)
        #elif self.kwargs['task'] == 'MultiTaskBO':
        #    self.bo = MultiTaskBO(**self.kwargs)
        else:
            raise ValueError(f'Invalid task: {self.kwargs["task"]}')


# Bayesian Optimization ======================================================
class SingleTaskBO(BOEngine):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def initialize(self) -> None:
        super().initialize()
        # Seed for Initialization
        seed_everything(self.kwargs['iseed'])
        # Objective Function
        self.objective = Objective(**self.kwargs['objective'])
        x_init = self.objective.sample(self.kwargs['n_init'])
        y_init = self.objective.evaluate(x_init)
        message = f'Objective function has been initialized.'
        self.kwargs['logger'].info(message + str(self.objective))
        # Dataset
        self.dataset = Dataset()
        self.dataset.add(x_init, y_init)
        message = f'Dataset has been initialized.'
        self.kwargs['logger'].info(message + str(self.dataset))
        train_x, train_y = self.dataset.get()
        # Model
        self.kwargs['model']['logger'] = self.kwargs['logger']
        self.model = Model(**self.kwargs['model'])
        self.model.fit(train_x, train_y)
        message = f'Model has been initialized.'
        self.kwargs['logger'].info(message + str(self.model))

    def run(self) -> None:
        # Seed for Running
        seed_everything(self.kwargs['rseed'])
        # Bayesian Optimization
        self.kwargs['acquisition']['logger'] = None
        for i in range(self.kwargs['max_iter']):
            current_best = self.dataset.best_targets
            self.kwargs['acquisition']['current_best'] = current_best
            self.kwargs['acquisition']['model'] = self.model
            message = '\n' + '*' * 79 + '\n' + \
                      f'* Iteration: {i:> 3d}' + ' ' * 62 + '*\n' + \
                      f'* Current Best: {current_best:.7f}' + ' ' * 53 + '*\n' + \
                      '*' * 79 + '\n'
            self.kwargs['logger'].info(message)
            # Acquisition
            self.acquisition = Acquisition(**self.kwargs['acquisition'])
            x_next = self.acquisition.optimize()
            y_next = self.objective(x_next)
            # Update Dataset & Model
            self.dataset.add(x_next, y_next)
            train_x, train_y = self.dataset.get()
            if i != self.kwargs['max_iter'] - 1:
                self.model.fit(train_x, train_y)
        current_best = self.dataset.best_targets
        message = '\n' + '*' * 79 + '\n' + \
                  f'* Iteration: {i+1:> 3d}' + ' ' * 62 + '*\n' + \
                  f'* Current Best: {current_best:.7f}' + ' ' * 53 + '*\n' + \
                  '*' * 79 + '\n'
        self.kwargs['logger'].info(message)

    def save(self) -> None:
        pass