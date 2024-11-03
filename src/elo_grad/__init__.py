import abc
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version

from array import array
from collections import defaultdict
from typing import Tuple, Optional, Dict, List, Callable, Generator

import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss

from .plot import HistoryPlotterMixin

__all__ = ["EloEstimator", "LogisticRegression", "RatingSystemMixin", "Regressor", "SGDOptimizer"]
__version__ = version(__package__)


@dataclass(frozen=True)
class Regressor:
    """
    Regressor for Elo rating system (additional to entities).

    Parameters
    ----------
    name : str
        Name of regressor column in dataset.
    k_factor : Optional[float]
        k-factor for this regressor's dimension. If None, the global k-factor for entities is used.
    lambda_reg : Optional[float]
        Regularisation parameter for regressor model coefficient, if L1 or L2 regularisation is used.
        NOTE: not implemented yet!
    """
    name: str
    k_factor: Optional[float] = None
    lambda_reg: Optional[float] = None


class Model(abc.ABC):

    loss: str

    def __init__(
        self,
        default_init_rating: float,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = None,
    ) -> None:
        self.ratings: Dict[str, Tuple[Optional[int], float]] = defaultdict(
            lambda: (None, default_init_rating)
        )
        self.init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = init_ratings
        if self.init_ratings is not None:
            self.ratings = self.ratings | self.init_ratings

    @abc.abstractmethod
    def calculate_gradient(self, y: int, *args) -> float:
        ...

    @abc.abstractmethod
    def calculate_expected_score(self, *args) -> float:
        ...


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def calculate_update_step(
        self,
        model: Model,
        y: int,
        entity_1: str,
        entity_2: str,
        # The two arguments below help to optimize things, but are not very natural
        # and will probably come back to bite me!
        regressor_contrib: float,
        regressor_values: Optional[Tuple[float, ...]],
    ) -> Generator[float, None, None]:
        ...

class LogisticRegression(Model):

    loss: str = "log-loss"

    def __init__(
        self,
        beta: float,
        default_init_rating: float,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]],
    ) -> None:
        super().__init__(default_init_rating, init_ratings)
        self.beta: float = beta

    def calculate_gradient(self, y: int, *args) -> float:
        if y not in {0, 1}:
            raise ValueError("Invalid result value %s", y)
        y_pred: float = self.calculate_expected_score(*args)

        return y - y_pred

    # We should make maxsize configurable
    @lru_cache(maxsize=512)
    def calculate_expected_score(self, *args) -> float:
        # I couldn't see any obvious speed-up from using NumPy/Numba data
        # structures but should revisit this.
        return 1 / (1 + math.pow(10, -sum(args) / (2 * self.beta)))


class SGDOptimizer(Optimizer):

    def __init__(self, k_factor: Tuple[float, ...]) -> None:
        self.k_factor: Tuple[float, ...] = k_factor

    def calculate_update_step(
        self,
        model: Model,
        y: int,
        entity_1: str,
        entity_2: str,
        regressor_contrib: float,
        regressor_values: Optional[Tuple[float, ...]],
    ) -> Generator[float, None, None]:
        entity_grad: float = model.calculate_gradient(
            y,
            model.ratings[entity_1][1],
            -model.ratings[entity_2][1],
            regressor_contrib,
        )
        yield self.k_factor[0] * entity_grad
        if regressor_values is not None:
            for i, v in enumerate(regressor_values, start=1):
                yield self.k_factor[i] * v * entity_grad


class RatingSystemMixin:
    """
    Mixin class for rating systems.

    This mixin defines the following functionality:

    - `_estimator_type` class attribute defaulting to `"rating-system"`;
    - `score` method that default to :func:`~sklearn.metrics.log_loss`.
    - enforce that `fit` does not require `y` to be passed through the `requires_y` tag.

    Read more in the :ref:`User Guide <rolling_your_own_estimator>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> from elo_grad import RatingSystemMixin
    >>> # Mixin classes should always be on the left-hand side for a correct MRO
    >>> class MyEstimator(RatingSystemMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         self.is_fitted_ = True
    ...         return self
    ...     def predict(self, X):
    ...         return np.full(shape=X.shape[0], fill_value=self.param)
    >>> estimator = MyEstimator(param=1)
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1, 0, 1])
    >>> estimator.fit(X, y).predict(X)
    array([1, 1, 1])
    >>> estimator.score(X, y)
    0.66...
    """

    _estimator_type = "classifier"
    classes_ = [[0, 1]]

    def score(self, X, y, sample_weight=None):
        """
        Return the log-loss on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """
        return log_loss(
            y, self.predict_proba(X)[:, 1], sample_weight=sample_weight
        )

    def _more_tags(self):
        return {"requires_y": False}


class EloEstimator(HistoryPlotterMixin, RatingSystemMixin, BaseEstimator):
    """
    Elo rating system classifier.

    Attributes
    ----------
    beta : float
        Normalization factor for ratings when computing expected score.
    columns : List[str]
        [entity_1, entity_2, result] columns names.
    default_init_rating : float
        Default initial rating for entities.
    entity_cols : Tuple[str, str]
        Names of columns identifying the names of the entities playing the games.
    init_ratings : Optional[Dict[str, Tuple[Optional[int], float]]]
        Initial ratings for entities (dictionary of form entity: (Unix timestamp, rating))
    k_factor : float
        Elo K-factor/step-size for gradient descent.
    k_factor_vec : Tuple[float, ...]
        Vector of Elo K-factor/step-size for gradient descent.
    model : Model
        Underlying statistical model.
    optimizer : Optimizer
        Optimizer to update the model.
    rating_history : List[Tuple[Optional[int], float]]
        Historical ratings of entities (if track_rating_history is True).
    score_col : str
        Name of score column (1 if entity_1 wins and 0 if entity_2 wins).
        Draws are not currently supported.
    additional_regressors : Optional[List[Regressor]]
        Additional regressors to include, e.g. home advantage.
    track_rating_history : bool
        Flag to track historical ratings of entities.

    Methods
    -------
    fit(X, y=None)
        Fit Elo rating system/calculate ratings.
    record_ratings()
        Record the current ratings of entities.
    predict_proba(X)
        Produce probability estimates.
    predict(X)
        Predict outcome of game.
    """

    def __init__(
        self,
        k_factor: float = 20,
        default_init_rating: float = 1200,
        beta: float = 200,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = None,
        entity_cols: Tuple[str, str] = ("entity_1", "entity_2"),
        score_col: str = "score",
        additional_regressors: Optional[List[Regressor]] = None,
        track_rating_history: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        k_factor : float
            Elo K-factor/step-size for gradient descent for the entities.
        default_init_rating : float
            Default initial rating for entities.
        beta : float
            Normalization factor for ratings when computing expected score.
        init_ratings : Optional[Dict[str, Tuple[Optional[int], float]]]
            Initial ratings for entities (dictionary of form entity: (Unix timestamp, rating))
        entity_cols : Tuple[str, str]
            Names of columns identifying the names of the entities playing the games.
        score_col : str
            Name of score column (1 if entity_1 wins and 0 if entity_2 wins).
            Draws are not currently supported.
        additional_regressors : Optional[List[Regressor]]
            Additional regressors to include, e.g. home advantage.
        track_rating_history : bool
            Flag to track historical ratings of entities.
        """
        self.entity_cols: Tuple[str, str] = entity_cols
        self.score_col: str = score_col
        self.columns: List[str] = list(entity_cols) + [score_col]
        self.beta: float = beta
        self.default_init_rating: float = default_init_rating
        self.init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = init_ratings
        self.model: Model = LogisticRegression(
            beta=beta,
            default_init_rating=default_init_rating,
            init_ratings=init_ratings,
        )
        self.additional_regressors: List[Regressor] = additional_regressors if additional_regressors is not None else []
        if additional_regressors is not None:
            self.columns.extend([r.name for r in additional_regressors])
        self.k_factor: float = k_factor
        self.k_factor_vec: Tuple[float, ...] = (
            k_factor,
            *(r.k_factor if r.k_factor is not None else k_factor for r in self.additional_regressors),
        )
        self.optimizer: Optimizer = SGDOptimizer(k_factor=self.k_factor_vec)
        self.track_rating_history: bool = track_rating_history
        self.rating_history: List[Tuple[Optional[int], float]] = defaultdict(list)  # type:ignore

    @staticmethod
    def _reinitialize_rating_system(method: Callable):
        """
        Decorator to reinitialize the rating system after parameter changes.
        Helpful when peforming a grid search.

        Parameters
        ----------
        method : Callable
            Method to decorate
        """

        def wrapper(self, **params):
            result = method(self, **params)
            self.model = LogisticRegression(
                beta=self.beta,
                default_init_rating=self.default_init_rating,
                init_ratings=self.init_ratings,
            )
            self.k_factor_vec = (
                self.k_factor,
                *(r.k_factor if r.k_factor is not None else self.k_factor for r in self.additional_regressors),
            )
            self.optimizer = SGDOptimizer(k_factor=self.k_factor_vec)

            return result

        return wrapper

    @_reinitialize_rating_system
    def set_params(self, **params):
        return super().set_params(**params)

    def _update_ratings(self, t: int, rating_deltas: Dict[str, float]) -> None:
        for entity in rating_deltas:
            self.model.ratings[entity] = (t, self.model.ratings[entity][1] + rating_deltas[entity])

    def record_ratings(self) -> None:
        """
        Record the current ratings of entities.
        """
        for k, v in self.model.ratings.items():
            self.rating_history[k].append(v)  # type:ignore

    def _transform(self, X: pd.DataFrame, return_expected_score: bool) -> Optional[np.ndarray]:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        X = X[self.columns]

        if not X.index.is_monotonic_increasing:
            raise ValueError("Index must be sorted.")
        current_ix: int = X.index[0]

        additional_regressor_flag: bool = len(self.additional_regressors) > 0
        additional_regressor_contrib: float = 0.0
        additional_regressor_values: Optional[Tuple[float, ...]] = None
        preds = array("f") if return_expected_score else None
        rating_deltas: Dict[str, float] = defaultdict(float)
        for row in X.itertuples(index=True):
            if additional_regressor_flag:
                ix, entity_1, entity_2, score, *additional_regressor_values = row
            else:
                ix, entity_1, entity_2, score = row

            if ix != current_ix:
                self._update_ratings(ix, rating_deltas)
                current_ix, rating_deltas = ix, defaultdict(float)
                if self.track_rating_history:
                    self.record_ratings()

            if additional_regressor_flag:
                additional_regressor_contrib = sum(
                    self.model.ratings[k.name][1] * v  # type:ignore
                    for k, v in zip(self.additional_regressors, additional_regressor_values)  # type:ignore
                )

            expected_score: float = self.model.calculate_expected_score(
                self.model.ratings[entity_1][1],
                -self.model.ratings[entity_2][1],
                additional_regressor_contrib,
            )
            if return_expected_score:
                preds.append(expected_score)  # type:ignore

            _rating_deltas: Generator[float, None, None] = self.optimizer.calculate_update_step(
                model=self.model,
                y=score,
                entity_1=entity_1,
                entity_2=entity_2,
                regressor_contrib=additional_regressor_contrib,
                regressor_values=additional_regressor_values,
            )
            entity_update: float = next(_rating_deltas)
            rating_deltas[entity_1] += entity_update
            rating_deltas[entity_2] -= entity_update
            if additional_regressor_flag:
                for r in self.additional_regressors:
                    rating_deltas[r.name] += next(_rating_deltas)

        self._update_ratings(ix, rating_deltas)
        if self.track_rating_history:
            self.record_ratings()

        if return_expected_score:
            return np.array(preds)
        return None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self._transform(X, return_expected_score=False)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self._transform(X, return_expected_score=True)
        return np.vstack((1 - preds, preds)).T  # type:ignore

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)[:, 1] > 0.5
