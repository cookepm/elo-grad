import abc
from typing import Optional, Dict, Tuple, List, Callable

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_poisson_deviance, log_loss

from elo_grad import EloEstimator, PoissonEloEstimator, Regressor, ClassifierRatingSystemMixin, RegressionRatingSystemMixin

__all__ = [
    "SKEloEstimator",
    "SKPoissonEloEstimator",
]


class SKRatingSystemMixin(abc.ABC):

    @abc.abstractmethod
    def score(self, X, y, sample_weight=None): ...

    def _more_tags(self):
        return {"requires_y": False}


class SKClassifierRatingSystemMixin(ClassifierRatingSystemMixin, SKRatingSystemMixin):
    """
    Mixin class for classification rating systems.

    This mixin defines the following functionality:

    - `_estimator_type` class attribute defaulting to `"classifier"`;
    - `score` method that default to :func:`~sklearn.metrics.log_loss`.
    - enforce that `fit` does not require `y` to be passed through the `requires_y` tag.
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
            Log-loss of ``self.predict_proba(X)[:, 1]`` w.r.t. `y`.
        """
        return log_loss(
            y, self.predict_proba(X)[:, 1], sample_weight=sample_weight
        )


class SKRegressionRatingSystemMixin(RegressionRatingSystemMixin, SKRatingSystemMixin):
    """
    Mixin class for regression rating systems.

    This mixin defines the following functionality:

    - `_estimator_type` class attribute defaulting to `"regressor"`;
    - `score` method that default to :func:`~sklearn.metrics.mean_poisson_deviance`.
    - enforce that `fit` does not require `y` to be passed through the `requires_y` tag.
    """

    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """
        Return the mean Poisson deviance on the given test data.

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
            Mean Poisson deviance of ``self.predict(X)`` w.r.t. `y`.
        """
        return mean_poisson_deviance(
            y, self.predict(X), sample_weight=sample_weight
        )


class SKBaseEstimator(BaseEstimator):
    @staticmethod
    def _reinitialize_dec(method: Callable):
        """
        Decorator to reinitialize the rating system after parameter changes.
        Helpful when performing a grid search.

        Parameters
        ----------
        method : Callable
            Method to decorate
        """

        def wrapper(self, **params):
            result = method(self, **params)
            self.reinitialize()

            return result

        return wrapper

    @_reinitialize_dec
    def set_params(self, **params):
        return super().set_params(**params)


class SKEloEstimator(SKClassifierRatingSystemMixin, EloEstimator, SKBaseEstimator):
    """
    Scikit-learn compatible Elo rating system classifier.

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
        k_factor: float = 20,
        default_init_rating: float = 1200,
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


class SKPoissonEloEstimator(SKRegressionRatingSystemMixin, PoissonEloEstimator, SKBaseEstimator):
    """
    Scikit-learn compatible Poisson Elo rating system.

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
        k_factor: float = 20,
        default_init_rating: float = 1200,
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
