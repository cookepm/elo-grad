# :chess_pawn: EloGrad

**Extended Elo rating system implementation exploiting the equivalence with logistic regression.**

**EloGrad** _(**Elo** as **Grad**ient descent)_ leverages the framing of the 
[Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system)
as logistic regression with stochastic gradient descent
(see [Elo as Logistic Regression](intro.md) for a walkthrough)
to offer a collection of extensions to the rating system.
All models are `scikit-learn` compatible.

## :sparkles: Features

- Standard Elo rating system for binary outcomes.
    - `pandas` and `scikit-learn` compatible.
    - See [`examples/nba.ipynb`](https://github.com/cookepm/elo-grad/blob/main/examples/nba.ipynb) for an example using NBA data.
- Elo rating system for binary outcomes with additional regressors, *e.g.* home advantage.
    - See [Additional Regressors](https://cookepm.github.io/elo-grad/feature_ref/additional_regressors/) for the theory and [`examples/nba.ipynb`](https://github.com/cookepm/elo-grad/blob/main/examples/nba.ipynb) for an example using NBA data.

## :book: Installation

You can install `elo-grad` with:
```bash
pip install elo-grad
```

## :zap: Quick Start

Detailed example notebooks are provided in the `examples/` directory.
To install any extra dependencies required to run the notebooks install with:
```bash
pip install elo-grad[examples]
```

### :clipboard: Minimal Example

```python
from elo_grad import EloEstimator, Regressor

# Input DataFrame with sorted index of Unix timestamps
# and columns entity_1 | entity_2 | score | home
# where score = 1 if player_1 won and score = 0 if
# player_2 won and home is a boolean flag indicating if
# entity_1 has home advantage.
df = ...
estimator = EloEstimator(
    k_factor=20, 
    default_init_rating=1200,
    entity_cols=("player_1", "player_2"),
    score_col="result",
    init_ratings=dict(home=(None, 0)),
    additional_regressors=[Regressor(name='home', k_factor=0.1)],
)
# Get expected scores
expected_scores = estimator.predict_proba(df)
# Get final ratings (of form (Unix timestamp, rating))
ratings = estimator.model.ratings
```

## :compass: Roadmap

In rough order, things we want to add are:

- Poisson model support
- Regularization (L1 & L2)
- Support for Polars
- Head-to-head ratings
- Interaction terms
- Other optimizers, e.g. momentum
- Support for draws
- Extend plotting support, e.g. plotly

## :blue_book: References

1. Elo rating system: https://en.wikipedia.org/wiki/Elo_rating_system
2. Elo rating system as logistic regression with stochastic gradient descent: https://stmorse.github.io/journal/Elo.html
3. Elo rating system for NFL predictions: https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
