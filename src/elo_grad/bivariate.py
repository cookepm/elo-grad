import abc
from collections import defaultdict

import math
from functools import lru_cache
from typing import Tuple, Optional, Generator, Dict, List

from . import BaseModel, BaseOptimizer, Regressor


class BivariateModel(BaseModel, abc.ABC):

    def __init__(
        self,
        beta: float,
        default_init_rating: Tuple[float, float],
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float, Optional[float]]]] = None,
    ) -> None:
        super().__init__(beta)
        self.init_ratings: Optional[Dict[str, Tuple[Optional[int], float, Optional[float]]]] = init_ratings
        self.ratings: Dict[str, Tuple[Optional[int], float, Optional[float]]] = defaultdict(  # type:ignore
            lambda: (None, *default_init_rating)
        )
        if self.init_ratings is not None:
            self.ratings = self.ratings | self.init_ratings

    @abc.abstractmethod
    def calculate_params(self, *args) -> Tuple[float, ...]:
        ...

    @abc.abstractmethod
    def calculate_gradient_from_params(self, y: Tuple[int, int], params: Tuple[float, ...]) -> Tuple[float, ...]:
        ...

    @abc.abstractmethod
    def calculate_expected_scores(self, *args) -> Tuple[float, float]:
        ...

    @abc.abstractmethod
    def calculate_gradient(self, y: Tuple[int, int], *args) -> Tuple[float, ...]:
        ...


class BivariateOptimizer(BaseOptimizer, abc.ABC):

    def __init__(
        self,
        k_factor: float,
        regressors: Optional[List[Regressor]],
        corr_regressors: Optional[List[Regressor]],
    ):
        super().__init__(k_factor, regressors)
        self.corr_regressors: Optional[List[Regressor]] = corr_regressors

    @abc.abstractmethod
    def calculate_update_step(
        self,
        model: BivariateModel,
        y: Tuple[int, int],
        entity_1: str,
        entity_2: str,
        regressor_values: Optional[Tuple[float, ...]],
        corr_regressor_values: Optional[Tuple[float, ...]],
        params: Optional[Tuple[float, ...]],
    ) -> Generator[Tuple[float, ...], None, None]:
        ...


class BivariatePoissonRegression(BivariateModel):

    # We should make maxsize configurable
    @lru_cache(maxsize=512)
    def calculate_params(self, *args) -> Tuple[float, ...]:
        return tuple(
            math.pow(10, sum(a) / (2 * self.beta)) for a in args
        )

    @staticmethod
    @lru_cache(maxsize=512)
    def _calculate_s(y: Tuple[int, int], k: int, params: Tuple[float, ...]) -> float:
        return (
            (params[2] / math.factorial(k))
            * ((params[0] ** (y[0] - k)) / (y[0] - k))
            * ((params[1] ** (y[1] - k)) / (y[1] - k))
        )

    def _calculate_modified_y(
        self,
        y: Tuple[int, int],
        params: Tuple[float, ...],
        i: int,
    ) -> float:
        if i < 0 or i > 2:
            raise ValueError("i must be between 0 and 2 (inclusive).")

        norm: float = sum(self._calculate_s(y, k, params) for k in range(min(y)))
        if i != 2:
            return (
                sum(
                    self._calculate_s(y, k, params) * (y[i] - k) / params[i]
                    for k in range(min(y))
                )
                / norm
            )
        return (
            sum(
                self._calculate_s(y, k, params) * k / params[i]
                for k in range(min(y))
            )
            / norm
        )

    def calculate_gradient_from_params(self, y: Tuple[int, int], params: Tuple[float, ...]) -> Tuple[float, ...]:
        return tuple(
            self._calculate_modified_y(y, params, i=i) - params[0] for i in range(2)
        )

    def calculate_gradient(self, y: Tuple[int, int], *args) -> Tuple[float, ...]:
        params: Tuple[float, ...] = self.calculate_params(*args)
        grad: Tuple[float, ...] = self.calculate_gradient_from_params(y, params)

        return grad

    def calculate_expected_scores(self, *args) -> Tuple[float, float]:
        params = self.calculate_params(*args)
        return (
            params[0] + params[2],
            params[1] + params[2],
        )


class BivariateSGDOptimizer(BivariateOptimizer):

    def calculate_update_step(
        self,
        model: BivariateModel,
        y: Tuple[int, int],
        entity_1: str,
        entity_2: str,
        regressor_values: Optional[Tuple[float, ...]],
        corr_regressor_values: Optional[Tuple[float, ...]],
        params: Optional[Tuple[float, ...]],
    ) -> Generator[Tuple[float, ...], None, None]:
        if params is not None:
            # If we already know the params, we shouldn't recalculate them
            entity_grad: Tuple[float, ...] = model.calculate_gradient_from_params(y, params)
        else:
            regressor_contrib_1, regressor_contrib_2 = 0.0, 0.0
            if self.regressors is not None:
                regressor_contribs: Generator[Tuple[float, float], None, None] = (
                    (model.ratings[r.name][1] * v, model.ratings[r.name][1] * (1 - v))
                    for r, v in zip(self.regressors, regressor_values)  # type:ignore
                )
                for c1, c2 in regressor_contribs:
                    regressor_contrib_1 += c1
                    regressor_contrib_2 += c2

            corr_regressor_contrib: float = 0.0
            if self.corr_regressors is not None:
                corr_regressor_contrib = sum(
                    model.ratings[r.name][1] * v for r, v in zip(self.corr_regressors, corr_regressor_values)  # type:ignore
                )

            entity_grad = model.calculate_gradient(
                y,
                (
                    model.ratings[entity_1][1],
                    -model.ratings[entity_2][2],  # type:ignore
                    regressor_contrib_1,
                ),
                (
                    model.ratings[entity_2][1],
                    -model.ratings[entity_1][2],  # type:ignore
                    regressor_contrib_2,
                ),
                (corr_regressor_contrib,),  # To handle correlation
            )

        yield tuple(self.k_factor * g for g in entity_grad)
        if self.regressors is not None:
            for r, v in zip(self.regressors, regressor_values):  # type:ignore
                 yield tuple(
                     r.k_factor * ((v * entity_grad[0]) - self._get_penalty(model, r)),  # type:ignore
                     r.k_factor * (((1 - v) * entity_grad[1]) - self._get_penalty(model, r)),  # type:ignore
                     0.0,  # type:ignore
                 )
        if self.corr_regressors is not None:
            for r, v in zip(self.corr_regressors, corr_regressor_values):  # type:ignore
                 yield tuple(
                     0.0,  # type:ignore
                     0.0,  # type:ignore
                     r.k_factor * ((v * entity_grad[2]) - self._get_penalty(model, r)),  # type:ignore
                 )
