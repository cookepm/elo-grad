import abc

import math
from collections import defaultdict
from functools import lru_cache
from typing import Tuple, Optional, Generator, Dict, List

from . import BaseModel, BaseOptimizer, Regressor


class BivariateModel(BaseModel, abc.ABC):

    def __init__(
        self,
        beta: float,
        default_init_rating: Tuple[float, Optional[float]],
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
    def calculate_params(
        self,
        args: Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]],
    ) -> Tuple[float, float, float]:
        ...

    @abc.abstractmethod
    def calculate_gradient_from_params(
        self,
        y: Tuple[int, int],
        params: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        ...

    @abc.abstractmethod
    def calculate_expected_scores(
        self,
        args: Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]],
    ) -> Tuple[float, float]:
        ...

    @abc.abstractmethod
    def calculate_expected_scores_from_params(self, params: Tuple[float, float, float]) -> Tuple[float, float]:
        ...

    @abc.abstractmethod
    def calculate_gradient(
        self,
        y: Tuple[int, int],
        args: Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]],
    ) -> Tuple[float, float, float]:
        ...


class BivariateOptimizer(BaseOptimizer, abc.ABC):

    def __init__(
        self,
        k_factor: float,
        regressors: Optional[Tuple[Optional[List[Regressor]], Optional[List[Regressor]], Optional[List[Regressor]]]],
    ):
        super().__init__(k_factor)
        self.regressors: Optional[Tuple[Optional[List[Regressor]], Optional[List[Regressor]], Optional[List[Regressor]]]] = regressors

    @abc.abstractmethod
    def calculate_update_step(
        self,
        model: BivariateModel,
        y: Tuple[int, int],
        entity_1: str,
        entity_2: str,
        regressor_values: Optional[Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]],
        params: Optional[Tuple[float, float, float]],
    ) -> Generator[float, None, None]:
        ...


class BivariatePoissonRegression(BivariateModel):

    # We should make maxsize configurable
    @lru_cache(maxsize=512)
    def calculate_params(
        self,
        args: Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]],
    ) -> Tuple[float, float, float]:
        return tuple(  # type:ignore
            math.pow(10, sum(a) / (2 * self.beta)) for a in args
        )

    @staticmethod
    @lru_cache(maxsize=512)
    def _calculate_s(y: Tuple[int, int], k: int, params: Tuple[float, float, float]) -> float:
        return (
            (params[2] / math.factorial(k))
            * ((params[0] ** (y[0] - k)) / (y[0] - k))
            * ((params[1] ** (y[1] - k)) / (y[1] - k))
        )

    def _calculate_modified_y(
        self,
        y: Tuple[int, int],
        params: Tuple[float, float, float],
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

    def calculate_gradient_from_params(
        self,
        y: Tuple[int, int],
        params: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        return tuple(  # type:ignore
            self._calculate_modified_y(y, params, i=i) - params[i] for i in range(3)
        )

    def calculate_gradient(
        self,
        y: Tuple[int, int],
        args: Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]],
    ) -> Tuple[float, float, float]:
        params: Tuple[float, float, float] = self.calculate_params(args)
        grad: Tuple[float, float, float] = self.calculate_gradient_from_params(y, params)

        return grad

    def calculate_expected_scores(
        self,
        args: Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]],
    ) -> Tuple[float, float]:
        params = self.calculate_params(args)
        return self.calculate_expected_scores_from_params(params)

    def calculate_expected_scores_from_params(self, params: Tuple[float, float, float]) -> Tuple[float, float]:
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
        regressor_values: Optional[Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]],
        params: Optional[Tuple[float, float, float]],
    ) -> Generator[float, None, None]:
        if params is not None:
            # If we already know the params, we shouldn't recalculate them
            entity_grad: Tuple[float, float, float] = model.calculate_gradient_from_params(y, params)
        else:
            regressor_contrib: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            if self.regressors is not None:
                regressor_contrib = tuple(  # type:ignore
                    sum(
                        model.ratings[r.name][1] * v  # type: ignore
                        for r, v in zip(reg, reg_val)  # type:ignore
                    ) if reg is not None  # type:ignore
                    else 0.0
                    for reg, reg_val in zip(self.regressors, regressor_values)  # type:ignore
                )

            entity_grad = model.calculate_gradient(
                y,
                args=(
                    (
                        model.ratings[entity_1][1],
                        -model.ratings[entity_2][2],  # type:ignore
                        regressor_contrib[0],
                    ),
                    (
                        model.ratings[entity_2][1],
                        -model.ratings[entity_1][2],  # type:ignore
                        regressor_contrib[1],
                    ),
                    (regressor_contrib[2],),  # To handle correlation
                )
            )

        for i in range(2):
            yield self.k_factor * entity_grad[i]
            if self.regressors is not None:
                if self.regressors[i] is not None:
                    for r, v in zip(self.regressors[i], regressor_values[i]):  # type:ignore
                        yield r.k_factor * ((v * entity_grad[i]) - self._get_penalty(model, r))  # type:ignore

        if self.regressors is not None:
            if self.regressors[2] is not None:
                for r, v in zip(self.regressors[2], regressor_values[2]):  # type:ignore
                    yield r.k_factor * ((v * entity_grad[2]) - self._get_penalty(model, r))  # type:ignore
