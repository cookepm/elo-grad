import abc
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version

from array import array
from collections import defaultdict
from typing import Tuple, Optional, Dict, List, Generator, Type, Any

import math
import narwhals as nw

from .plot import HistoryPlotterMixin

__all__ = [
    "EloEstimator",
    "LogisticRegression",
    "PoissonEloEstimator",
    "PoissonRegression",
    "Regressor",
    "SGDOptimizer",
]
__version__ = version(__package__)

PRED_PROBA: str = "pred_proba"


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
    penalty : Optional[str]
        Specify the norm of the penalty:

        - `None`: no penalty is added
        - `l1`: add a L1 penalty term
        - `l2`: add a L2 penalty term
    """
    name: str
    k_factor: Optional[float] = None
    lambda_reg: Optional[float] = None
    penalty: Optional[str] = None

    def __post_init__(self):
        if self.penalty is not None and self.penalty not in ("l1", "l2"):
            raise ValueError("Penalty must be None, 'l1' or 'l2'.")


class Model(abc.ABC):

    def __init__(
        self,
        beta: float,
        default_init_rating: float,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = None,
    ) -> None:
        self.beta: float = beta
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

    @abc.abstractmethod
    def calculate_gradient_from_expected_score(self, y: int, expected_score: float) -> float:
        ...


class Optimizer(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def _get_penalty(cls, model: Model, regressor: Regressor) -> float:
        ...

    @abc.abstractmethod
    def calculate_update_step(
        self,
        model: Model,
        y: int,
        entity_1: str,
        entity_2: str,
        additional_regressor_values: Optional[Tuple[float, ...]],
        expected_score: Optional[float],
    ) -> Generator[float, None, None]:
        ...


class LogisticRegression(Model):

    def __init__(
        self,
        beta: float,
        default_init_rating: float,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]],
    ) -> None:
        super().__init__(beta, default_init_rating, init_ratings)

    def calculate_gradient_from_expected_score(self, y: int, expected_score: float) -> float:
        if y not in {0, 1}:
            raise ValueError("Invalid score value %s", y)

        return y - expected_score

    def calculate_gradient(self, y: int, *args) -> float:
        y_pred: float = self.calculate_expected_score(*args)
        grad: float = self.calculate_gradient_from_expected_score(y, y_pred)

        return grad

    # We should make maxsize configurable
    @lru_cache(maxsize=512)
    def calculate_expected_score(self, *args) -> float:
        # I couldn't see any obvious speed-up from using NumPy/Numba data
        # structures but should revisit this.
        return 1 / (1 + math.pow(10, -sum(args) / (2 * self.beta)))


class PoissonRegression(Model):

    def __init__(
        self,
        beta: float,
        default_init_rating: float,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]],
    ) -> None:
        super().__init__(beta, default_init_rating, init_ratings)

    def calculate_gradient_from_expected_score(self, y: int, expected_score: float) -> float:
        if not isinstance(y, int):
            raise ValueError("Invalid score value %s", y)

        # This is the same as for the logistic regression model
        # - is this just a property of all exponential family models?
        return y - expected_score

    def calculate_gradient(self, y: int, *args) -> float:
        y_pred: float = self.calculate_expected_score(*args)
        grad: float = self.calculate_gradient_from_expected_score(y, y_pred)

        return grad

    # We should make maxsize configurable
    @lru_cache(maxsize=512)
    def calculate_expected_score(self, *args) -> float:
        return math.pow(10, sum(args) / (2 * self.beta))


class SGDOptimizer(Optimizer):

    def __init__(self, k_factor: float, additional_regressors: Optional[List[Regressor]]) -> None:
        self.k_factor: float = k_factor
        self.additional_regressors: Optional[List[Regressor]] = additional_regressors
        if additional_regressors is None:
            self.k_factor_vec: Tuple[float, ...] = (k_factor,)
        else:
            self.k_factor_vec = (
                k_factor,
                *(r.k_factor if r.k_factor is not None else k_factor for r in additional_regressors),
            )

    @classmethod
    def _get_penalty(cls, model: Model, regressor: Regressor) -> float:
        match regressor.penalty:
            case "l1":
                return regressor.lambda_reg * math.copysign(1, model.ratings[regressor.name][1])  # type:ignore
            case "l2":
                return 2 * regressor.lambda_reg * model.ratings[regressor.name][1]  # type:ignore
            case _:
                return 0.0

    def calculate_update_step(
        self,
        model: Model,
        y: int,
        entity_1: str,
        entity_2: str,
        additional_regressor_values: Optional[Tuple[float, ...]],
        expected_score: Optional[float],
    ) -> Generator[float, None, None]:
        if expected_score is not None:
            # If we already know the expected score, we shouldn't recalculate it
            entity_grad: float = model.calculate_gradient_from_expected_score(y, expected_score)
        else:
            if self.additional_regressors is None:
                additional_regressor_contrib: float = 0.0
            else:
                additional_regressor_contrib = sum(
                    model.ratings[r.name][1] * v  # type:ignore
                    for r, v in zip(self.additional_regressors, additional_regressor_values)  # type:ignore
                )
            entity_grad = model.calculate_gradient(
                y,
                model.ratings[entity_1][1],
                -model.ratings[entity_2][1],
                additional_regressor_contrib,
            )

        yield self.k_factor_vec[0] * entity_grad
        if self.additional_regressors is not None:
            for r, v in zip(self.additional_regressors, additional_regressor_values):  # type:ignore
                 yield r.k_factor * ((v * entity_grad) - self._get_penalty(model, r))  # type:ignore


class RatingSystemMixin(abc.ABC):

    @abc.abstractmethod
    def predict(self, X):
        ...

class ClassifierRatingSystemMixin(RatingSystemMixin):
    """
    Mixin class for classification rating systems.

    This mixin defines the following functionality:

    - `_estimator_type` class attribute defaulting to `"classifier"`;
    - `score` method that default to :func:`~sklearn.metrics.log_loss`.
    - enforce that `fit` does not require `y` to be passed through the `requires_y` tag.
    """

    def predict_proba(self, X):
        pred_proba = self._transform(X, return_expected_score=True)
        return (
            pred_proba
            .to_frame()
            .select(
                (1 - nw.col(PRED_PROBA)).alias("0"),
                nw.col(PRED_PROBA).alias("1"),
            )
        )

    def predict(self, X):
        return (self.predict_proba(X)["1"] >= 0.5).rename(PRED_PROBA)


class RegressionRatingSystemMixin(RatingSystemMixin):
    """
    Mixin class for regression rating systems.

    This mixin defines the following functionality:

    - `_estimator_type` class attribute defaulting to `"regressor"`;
    - `score` method that default to :func:`~sklearn.metrics.mean_poisson_deviance`.
    - enforce that `fit` does not require `y` to be passed through the `requires_y` tag.
    """

    def predict(self, X):
        return self._transform(X, return_expected_score=True)  # type:ignore


class BaseEloEstimator(HistoryPlotterMixin):
    """
    Elo rating system classifier.

    Attributes
    ----------
    model_type : Type[Model]
        Base model class type.
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
    model : Model
        Underlying statistical model.
    optimizer : Optimizer
        Optimizer to update the model.
    rating_history : List[Tuple[Optional[int], float]]
        Historical ratings of entities (if track_rating_history is True).
    score_col : str
        Name of score column (1 if entity_1 wins and 0 if entity_2 wins).
        Draws are not currently supported.
    date_col : str
        Name of date column, which has Unix timestamp (in seconds) of the
        game.
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
        model_type: Type[Model],
        k_factor: float,
        default_init_rating: float,
        beta: float,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]],
        entity_cols: Tuple[str, str],
        score_col: str,
        date_col: str,
        additional_regressors: Optional[List[Regressor]],
        track_rating_history: bool,
    ) -> None:
        """
        Parameters
        ----------
        model_type : Type[Model]
            Base model class type.
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
            Name of score column
        date_col : str
            Name of date column, which has Unix timestamp (in seconds) of the
            game.
        additional_regressors : Optional[List[Regressor]]
            Additional regressors to include, e.g. home advantage.
        track_rating_history : bool
            Flag to track historical ratings of entities.
        """
        self.entity_cols: Tuple[str, str] = entity_cols
        self.score_col: str = score_col
        self.date_col: str = date_col
        self.columns: List[str] = [date_col, *entity_cols, score_col]
        self.beta: float = beta
        self.default_init_rating: float = default_init_rating
        self.init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = init_ratings
        self.model_type: Type[Model] = model_type
        self.model: Model = model_type(
            beta=beta,
            default_init_rating=default_init_rating,
            init_ratings=init_ratings,
        )
        self.additional_regressors: List[Regressor] = additional_regressors if additional_regressors is not None else []
        if additional_regressors is not None:
            self.columns.extend([r.name for r in additional_regressors])
        self.k_factor: float = k_factor
        self.optimizer: Optimizer = SGDOptimizer(
            k_factor=self.k_factor,
            additional_regressors=self.additional_regressors,
        )
        self.track_rating_history: bool = track_rating_history
        self.rating_history: List[Tuple[Optional[int], float]] = defaultdict(list)  # type:ignore

    def reinitialize(self) -> None:
        """
        Reinitialize the rating system after parameter changes.
        Helpful when performing a grid search.
        """
        self.model = self.model_type(
            beta=self.beta,
            default_init_rating=self.default_init_rating,
            init_ratings=self.init_ratings,
        )
        self.optimizer = SGDOptimizer(k_factor=self.k_factor, additional_regressors=self.additional_regressors)

    def _update_ratings(self, t: int, rating_deltas: Dict[str, float]) -> None:
        for entity in rating_deltas:
            self.model.ratings[entity] = (t, self.model.ratings[entity][1] + rating_deltas[entity])

    def record_ratings(self) -> None:
        """
        Record the current ratings of entities.
        """
        for k, v in self.model.ratings.items():
            self.rating_history[k].append(v)  # type:ignore

    def _transform(self, X, return_expected_score):
        df: nw.DataFrame = nw.from_native(X)
        native_namespace: Any = nw.get_native_namespace(df)
        df = df.select(self.columns)

        if not df[self.date_col].is_sorted(descending=False):
            raise ValueError("DataFrame must be sorted by date.")
        current_ix: int = df[self.date_col].item(0)

        additional_regressor_flag: bool = len(self.additional_regressors) > 0
        additional_regressor_contrib: float = 0.0
        additional_regressor_values: Optional[Tuple[float, ...]] = None
        preds = array("f") if return_expected_score else None
        rating_deltas: Dict[str, float] = defaultdict(float)
        for row in df.iter_rows(named=False, buffer_size=512):
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
                additional_regressor_values=additional_regressor_values,
                expected_score=expected_score,
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
            return nw.new_series(name=PRED_PROBA, values=preds, native_namespace=native_namespace)
        return None

    def fit(self, X, y=None):
        self._transform(X, return_expected_score=False)
        return self


class EloEstimator(ClassifierRatingSystemMixin, BaseEloEstimator):
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
    model : Model
        Underlying statistical model.
    optimizer : Optimizer
        Optimizer to update the model.
    rating_history : List[Tuple[Optional[int], float]]
        Historical ratings of entities (if track_rating_history is True).
    score_col : str
        Name of score column (1 if entity_1 wins and 0 if entity_2 wins).
        Draws are not currently supported.
    date_col : str
        Name of date column, which has Unix timestamp (in seconds) of the
        game.
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
        k_factor: float,
        default_init_rating: float,
        beta: float = 200,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = None,
        entity_cols: Tuple[str, str] = ("entity_1", "entity_2"),
        score_col: str = "score",
        date_col: str = "t",
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
        date_col : str
            Name of date column, which has Unix timestamp (in seconds) of the
            game.
        additional_regressors : Optional[List[Regressor]]
            Additional regressors to include, e.g. home advantage.
        track_rating_history : bool
            Flag to track historical ratings of entities.
        """
        super().__init__(
            model_type=LogisticRegression,
            k_factor=k_factor,
            default_init_rating=default_init_rating,
            beta=beta,
            init_ratings=init_ratings,
            entity_cols=entity_cols,
            score_col=score_col,
            date_col=date_col,
            additional_regressors=additional_regressors,
            track_rating_history=track_rating_history,
        )


class PoissonEloEstimator(RegressionRatingSystemMixin, BaseEloEstimator):
    """
    Poisson Elo rating system.

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
    model : Model
        Underlying statistical model.
    optimizer : Optimizer
        Optimizer to update the model.
    rating_history : List[Tuple[Optional[int], float]]
        Historical ratings of entities (if track_rating_history is True).
    score_col : str
        Name of score column (1 if entity_1 wins and 0 if entity_2 wins).
        Draws are not currently supported.
    date_col : str
        Name of date column, which has Unix timestamp (in seconds) of the
        game.
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
    predict(X)
        Predict score.
    """

    def __init__(
        self,
        k_factor: float,
        default_init_rating: float,
        beta: float = 200,
        init_ratings: Optional[Dict[str, Tuple[Optional[int], float]]] = None,
        entity_cols: Tuple[str, str] = ("entity_1", "entity_2"),
        score_col: str = "score",
        date_col: str = "t",
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
        date_col : str
            Name of date column, which has Unix timestamp (in seconds) of the
            game.
        additional_regressors : Optional[List[Regressor]]
            Additional regressors to include, e.g. home advantage.
        track_rating_history : bool
            Flag to track historical ratings of entities.
        """
        super().__init__(
            model_type=PoissonRegression,
            k_factor=k_factor,
            default_init_rating=default_init_rating,
            beta=beta,
            init_ratings=init_ratings,
            entity_cols=entity_cols,
            score_col=score_col,
            date_col=date_col,
            additional_regressors=additional_regressors,
            track_rating_history=track_rating_history,
        )
