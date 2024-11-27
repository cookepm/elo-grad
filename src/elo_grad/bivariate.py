import abc

from typing import Tuple

from . import BaseModel


class BivariateModel(BaseModel, abc.ABC):

    @abc.abstractmethod
    def calculate_gradient(self, y: Tuple[int, int], *args) -> Tuple[float, ...]:
        ...

    @abc.abstractmethod
    def calculate_expected_scores(self, *args) -> Tuple[float, float]:
        ...
