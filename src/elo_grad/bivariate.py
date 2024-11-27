import abc

from typing import Tuple, Optional, Generator

from . import BaseModel, Regressor


class BivariateModel(BaseModel, abc.ABC):

    @abc.abstractmethod
    def calculate_gradient(self, y: Tuple[int, int], *args) -> Tuple[float, ...]:
        ...

    @abc.abstractmethod
    def calculate_expected_scores(self, *args) -> Tuple[float, float]:
        ...


class BivariateOptimizer(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def _get_penalty(cls, model: BivariateModel, regressor: Regressor) -> float:
        ...

    @abc.abstractmethod
    def calculate_update_step(
        self,
        model: BivariateModel,
        y: Tuple[int, int],
        entity_1: str,
        entity_2: str,
        additional_regressor_values: Optional[Tuple[float, ...]],
        params: Optional[Tuple[float, ...]],
    ) -> Generator[Tuple[float, ...], None, None]:
        ...
