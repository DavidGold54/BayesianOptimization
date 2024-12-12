import os
import time
import random
import logging
import datetime

import numpy as np
import torch


# Random Seed ----------------------------------------------------------------

# References:
#   https://www.kaggle.com/code/rhythmcam/random-seed-everything
#   https://qiita.com/kaggle_grandmaster-arai-san/items/d59b2fb7142ec7e270a5#seed_everything
#   https://pytorch.org/docs/master/notes/randomness.html


DEFAULT_RANDOM_SEED = 42


def seed_basic(seed: int = DEFAULT_RANDOM_SEED) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def seed_torch(seed: int = DEFAULT_RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_everything(seed: int = DEFAULT_RANDOM_SEED) -> None:
    seed_basic(seed)
    seed_torch(seed)


# Logging --------------------------------------------------------------------

# References:
#   https://qiita.com/Galvalume29/items/835b65cddaf094c2b3c2
#   https://github.com/ghmagazine/kagglebook/blob/master/ch04-model-interface/code/util.py
#   https://zenn.dev/roju/articles/bd2552eeeeb3c6


class Logger:
    def __init__(self, result_dir: str) -> None:
        self.general_logger = logging.getLogger('general')
        self.result_logger = logging.getLogger('result')
        stream_handler = logging.StreamHandler()
        general_file_handler = logging.FileHandler(filename=f'{result_dir}/general.log',
                                                   mode='w', encoding='utf-8')
        result_file_handler = logging.FileHandler(filename=f'{result_dir}/result.log',
                                                  mode='w', encoding='utf-8')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(general_file_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(result_file_handler)
            self.result_logger.setLevel(logging.INFO)
        self.id = None
        self.start_time = None
        self.stop_time = None
    
    def info(self, message: str) -> None:
        self.general_logger.info(f'[{self.now_string()}] - {message}')

    def result(self, message: str) -> None:
        self.result_logger.info(message)

    def now_string(self) -> str:
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    def start(self, id: str) -> None:
        self.id = id
        self.general_logger.info(f'[{self.now_string()}] - {self.id} starts.')
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        self.stop_time = time.perf_counter()
        elapsed_time = self.stop_time - self.start_time
        self.general_logger.info(f'[{self.now_string()}] - {self.id} ends in {elapsed_time} s.')