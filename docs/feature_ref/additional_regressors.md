# Additional Regressors

Having framed the Elo rating system as logistic regression with stochastic gradient descent
(see [Elo as Logistic Regression](../intro.md))
it is straightforward to add additional regressors.
We modify (1) from [Elo as Logistic Regression](../intro.md) as
$$
\begin{equation}
\mathbb{E}[y_{ij}|r_1,\cdots,r_n;\hat{r}_k] = \frac{1}{1 + 10^{-(r_i - r_j + \hat{r}_k x_k) / 2\beta}},
\end{equation}
$$
where $\hat{r}_k$ denote the "ratings" of the additional regressors
and $x_k$ is the value of the additional regressors for the game.
The update method for the entities remains unchanged, i.e. given by (3) from [Elo as Logistic Regression](../intro.md),
except we now also condition the expectation on the additional regressors.
The update method for the additional regressors is given by

$$
\begin{equation}
\hat{r}^\prime_k=\hat{r}_k + k \left(y_{ij} - \mathbb{E}[y_{ij}|r_1,\cdots,r_n;\hat{r}_k]\right)x_k.
\end{equation}
$$

It is often useful to specify a different k-factor/step-size for the additional regressors than for the entities,
as the variance can be quite different for the additional regressors.
Equivalently, we could scale our regressors appropriately before passing the data to the rating system.
For convenience `elo-grad` allows the specification of different k-factors for the additional regressors
(see [Example](additional_regressors.md#example)).

## Example

```python
from elo_grad import EloEstimator, Regressor

# Input DataFrame with sorted index of Unix timestamps
# and columns entity_1 | entity_2 | score | home
# where score = 1 if player_1 won and score = 0 if
# player_2 won. In all games, entity_1 has home
# advantage, so home = 1 for all rows.
home_col = "home"
df = ...
estimator = EloEstimator(
    k_factor=20,
    default_init_rating=1200,
    entity_cols=("player_1", "player_2"),
    score_col="result",
    # Set the initial rating for home advantage to 0
    init_ratings={home_col: (None, 0)},
    # Set k-factor/step-size to 1 for the home advantage regressor
    regressors=[Regressor(name=home_col, k_factor=1)],
)
# Get expected scores
expected_scores = estimator.predict_proba(df)
# Get final ratings (of form (Unix timestamp, rating)) for home advantage
ratings = estimator.model.ratings[home_col]
```

## Other Approaches

The Elo rating system is the basis for [538](https://abcnews.go.com/538)'s [NFL predictions](https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/).
There are a number of differences in their approach - notably they incorporate margin of victory.

Focusing on regressors, they include additional regressors such as home advantage and rest adjustment,
although these are fixed rather than fitted values.
Interestingly, they also include a play-off adjustment factor which *multiplies* the rating difference.
To include such a factor, we would need to allow *interaction terms* between a play-off flag and the entities 
(see [Roadmap](../index.md#compass-roadmap)).
