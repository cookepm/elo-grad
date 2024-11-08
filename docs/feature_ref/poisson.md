# Poisson Elo

If we can frame the [Elo rating system as logistic regression](../intro.md),
why not define other rating systems based on other 
[generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLMs)?
We do just that and derive a rating system equivalent to 
[Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression)
here.

## Motivation

Poisson regression models *count* data as the response variable.
This can be useful for sports such as football,
where we can predict the number of goals a team will score.

The Poisson model is even [used to model binary outcomes](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0519-5)
in cases where we are interested in [relative risk ratios](https://en.wikipedia.org/wiki/Relative_risk)
rather than the [odds ratio](https://en.wikipedia.org/wiki/Odds_ratio).
The latter is easily extracted from the logistic regression model but is non-intuitive,
while the former is hard to extract from the logistic regression model
but straightforward to extract from the Poisson model.

## Poisson Regression

Suppose we have a count response $y|\mathbf{x}\sim\text{Poisson}(\lambda)$,
where $\mathbf{x}$ is the set of input variables/regressors 
and $\lambda$ is the expectation of response in a given interval.
If we model this outcome with model parameters $\mathbf{w}$, the likelihood function is given by
$$
\begin{equation}
\mathcal{L}(\mathbf{w})=
\frac{\lambda(\mathbf{w;x})^y e^{-\lambda(\mathbf{w;x})}}{y!}.
\end{equation}
$$
As in [Elo as Logistic Regression](../intro.md), we are ignoring the product over samples for the moment.

The Poisson model assumes the count response is given by
$$
\begin{equation}
\lambda(\mathbf{w};\mathbf{x})=
e^{\mathbf{w}\cdot x}.
\end{equation}
$$
We have not included an intercept.

Suppose each observation corresponds to the number of points/goals scored in a given interval between two of $n$ entities.
We can define two ratings for each entity -
one for their attacking ability and one for their defensive ability.
Let the index $i$ run over the attacking abilities
and the index $\underbar{i}$ run over the defensive abilities.

We can write our regressors as

$$
\begin{equation}
\mathbf{x}_k=\delta_{ik} - \delta_{j\underbar{k}},
\end{equation}
$$

where $\delta_{ij}$ is the *Kronecker delta*.
The table below shows how the data would look like for three observations:

- team $1$ scores two goals against team $2$ on 01/01/24,
- team $2$ scores one goal against team $1$ on 01/01/24,
- team $n$ score no goals against team $1$ on 02/01/24.

| Date     | y   | Team 1 (A) | Team 2 (A) | Team 3 (A) | ... | Team n (A) | Team 1 (D) | Team 2 (D) | Team 3 (D) | ... | Team n (D) |
|----------|-----|------------|------------|------------|-----|------------|------------|------------|------------|-----|------------|
| 01/01/24 | 2   | 1          | 0          | 0          | ... | 0          | 0          | -1         | 0          | ... | 0          |
| 01/01/24 | 1   | 0          | 1          | 0          | ... | 1          | -1         | 0          | 0          | ... | 0          |
| 02/01/24 | 0   | 0          | 0          | 0          | ... | 1          | -1         | 0          | 0          | ... | 0          |
| ...      | ... | ...        | ...        | ...        | ... | ...        | ...        | ...        | ...        | ... | ...        |

where *(A)* indicates attacking ability and *(D)* defensive ability.

With the regressors described above, we can rewrite (2) as
$$
\begin{equation}
\lambda(\mathbf{w};\mathbf{x})=
\exp(w_i - w_{\underbar{j}}),
\end{equation}
$$
where $y_{ij}$ represents the goals scored by team $i$ against $j$.
Recalling that $\lambda$ is the expected count, we can then write
$$
\begin{equation}
\mathbb{E}\left[y_{ij}|w_1,\cdots w_n\right]=
\exp(w_i - w_{\underbar{j}}).
\end{equation}
$$

## Stochastic Gradient Descent

It remains to find the update method for the rating system,
which is just (mini-batch) stochastic gradient descent (SGD).

We can perform gradient descent for Poisson regression by minimising the negative log-likelihood.
The gradient of the negative log likelihood for Poisson regression is given by
$$
\begin{align}
\nabla_{\mathbf{w}}\mathcal l &=
-\sum_a \left[
y_a \frac{\nabla_{\mathbf{w}}\lambda(\mathbf{w};\mathbf{x}_a)}{\lambda(\mathbf{w};\mathbf{x}_a)}
- \lambda(\mathbf{w};\mathbf{x}_a) \nabla_a \lambda(\mathbf{w};\mathbf{x}_a)
\right],\\\\
&=-\sum_a\left[
y_a \mathbf{x}_a
- (1 - y_a)p(y|\mathbf{w};\mathbf{x}_a))
\right]\mathbf{x}_a,\\\\
&=\sum_a\left[\lambda(\mathbf{w};\mathbf{x}) - y_a\right]\mathbf{x}_a,
\end{align}
$$
where $\lambda(\mathbf{w};\mathbf{x})$ is the expected count (2),
$l$ is the negative log-likelihood and $a$ runs over the set of observations/games.

The update method for the model parameters is then given by
$$
\begin{equation}
\mathbf{w}^t
=\mathbf{w}^{t-1} + \alpha \sum_a \left(y_a - \lambda(\mathbf{w};\mathbf{x})\right).
\end{equation}
$$

If we have the regressors described in the 
[Poisson Regression section](poisson.md#poisson-regression) 
above
and restrict ourselves to stochastic gradient descent, 
this becomes
$$
\begin{equation}
\mathbf{w}^t
=\mathbf{w}^{t-1} + \alpha \left(y_{ij} - \exp(w_i - w_{\underbar{j}})\right)
\end{equation}
$$
or
$$
\begin{equation}
\mathbf{r}^t
=\mathbf{r}^{t-1} + k \left(y_{ij} - 10^{(r_i - r_{\underbar{j}}) / 2\beta}\right).
\end{equation}
$$
where $r_i:=2\beta w_i / \ln 10$ and $k:=2\alpha\beta / \ln10$.
We perform this change of variables to align with Elo conventions.
Now (5) is given by
$$
\begin{equation}
\mathbb{E}\left[y_{ij}|r_1,\cdots r_n\right]=
10^{(r_i - r_{\underbar{j}}) / 2\beta}.
\end{equation}
$$

## With Additional Regressors

Following [Additional Regressors](./additional_regressors.md),
equation (11) become

$$
\begin{equation}
\mathbb{E}\left[y_{ij}|r_1,\cdots r_n;\hat{r}_k\right]=
10^{(r_i - r_{\underbar{j}} + \hat{r}_k x_k) / 2\beta}.
\end{equation}
$$

with (10) modified accordingly. 

## Example

We provide a more detailed example,
including details of how data can be pre-processed to pass to the rating system in
[`examples/football.ipynb`](https://github.com/cookepm/elo-grad/blob/main/examples/football.ipynb).

```python
from elo_grad import PoissonEloEstimator, Regressor

# Input DataFrame with sorted index of Unix timestamps
# and columns entity_1_attacking | entity_2_defensive | score | home
# where score is the number of points/goals scored
# by entity 1 against entity 2.
# In all games, entity_1 has home
# advantage, so home = 1 for all rows.
home_col = "home"
df = ...
estimator = PoissonEloEstimator(
    k_factor=20, 
    default_init_rating=1200,
    entity_cols=("entity_1_attacking", "entity_2_defensive"),
    score_col="result",
    # Set the initial rating for home advantage to 0
    init_ratings={home_col: (None, 0)},  
    # Set k-factor/step-size to 1 for the home advantage regressor
    additional_regressors=[Regressor(name=home_col, k_factor=1)],
)
# Get expected scores
expected_scores = estimator.predict(df)
# Get final ratings (of form (Unix timestamp, rating)) for home advantage
ratings = estimator.model.ratings[home_col]
```

## Other Approaches

The Elo rating system is the basis for 
[538](https://abcnews.go.com/538)'s [NFL predictions](https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/).
They use the traditional, logistic regression-based Elo rating system
but also incorporate margin of victory.
They do this by including a multiplier in the update method.
Given the natural equivalence of the Elo rating system with logistic regression,
we believe this is not a very natural way to include margin of victory
and that using a Poisson-based Elo should be preferred.

## References

A Poisson regression-based Elo rating system was implemented 
[here](https://github.com/octonion/puzzles/blob/master/elo/poisson.py)
and provided the inspiration for this.
Note that the update methods used there are not the same.
