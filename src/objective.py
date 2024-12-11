from abc import ABC, abstractmethod

import torch
from torch import Tensor


# Base -----------------------------------------------------------------------
class Objective(ABC):
    """目的関数の基底クラス。

    最適化問題における目的関数を表現するための抽象基底クラスです。
    ノイズの付加や評価回数のカウントなどの共通機能を提供します。

    Attributes:
        _bounds (Tensor): 変数の上下限値
        _noise_std (float): ガウシアンノイズの標準偏差
        _kwargs (dict): その他のパラメータ
        __evaluation_count (int): 評価回数のカウンタ
    """

    def __init__(self, bounds: Tensor, noise_std: float = 0.0, **kwargs) -> None:
        """
        Args:
            bounds (Tensor): 変数の上下限値 [下限, 上限]
            noise_std (float, optional): ノイズの標準偏差. デフォルトは0.0
            **kwargs: その他のパラメータ
        """
        self._bounds = bounds
        self._noise_std = noise_std
        self._kwargs = kwargs
        self.__evaluation_count = 0

    def __call__(self, x: Tensor) -> Tensor:
        """目的関数の評価を行います。

        Args:
            x (Tensor): 評価点

        Returns:
            Tensor: 目的関数値
        """
        self.validate(x)
        value = self.evaluate(x)
        if self._noise_std > 0.0:
            value += torch.randn_like(value) * self._noise_std
        self.__evaluation_count += 1
        return value
    
    def reset(self) -> None:
        """評価回数のカウンタをリセットします。"""
        self.__evaluation_count = 0

    @abstractmethod
    def validate(self, x: Tensor) -> None:
        """入力値の妥当性を検証します。

        Args:
            x (Tensor): 検証する入力値
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, x: Tensor) -> Tensor:
        """目的関数の実際の評価を行います。

        Args:
            x (Tensor): 評価点

        Returns:
            Tensor: 目的関数値
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int) -> Tensor:
        """定義域からランダムにサンプリングを行います。

        Args:
            n (int): サンプル数

        Returns:
            Tensor: サンプリングされた点
        """
        raise NotImplementedError


# Objective Functions ========================================================
class Forrester(Objective):
    """Forrester関数の実装。

    単変数のテスト関数として広く使用される関数です。
    f(x) = (6x - 2)^2 * sin(12x - 4)
    """

    def __init__(self, bounds: Tensor, noise_std: float = 0.0, **kwargs) -> None:
        """
        Args:
            bounds (Tensor): 変数の上下限値 [下限, 上限]
            noise_std (float, optional): ノイズの標準偏差. デフォルトは0.0
            **kwargs: その他のパラメータ
        """
        super().__init__(bounds, noise_std, **kwargs)

    def validate(self, x: Tensor) -> None:
        """入力値が定義域内にあることを確認します。

        Args:
            x (Tensor): 検証する入力値
        """
        assert (self._bounds[0] <= x).all() and (x <= self._bounds[1]).all()

    def evaluate(self, x: Tensor) -> Tensor:
        """Forrester関数の評価を行います。

        Args:
            x (Tensor): 評価点

        Returns:
            Tensor: 関数値
        """
        return (6 * x - 2) ** 2 * torch.sin(12 * x - 4)
    
    def sample(self, n: int) -> Tensor:
        """定義域から一様分布に従ってサンプリングを行います。

        Args:
            n (int): サンプル数

        Returns:
            Tensor: サンプリングされた点
        """
        return torch.rand(n, 1) * (self._bounds[1] - self._bounds[0]) + self._bounds[0]
