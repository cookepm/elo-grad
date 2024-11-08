# Elo as Logistic Regression

*EloGrad* is motivated by the framing of the Elo rating system as logistic regression with gradient descent,
and the natural extensions to the rating system that arise from that.
We derive that relationship here.

In `elo-grad`, we implement the Elo rating system as logistic regression with mini-batch gradient descent, 
where batches are defined by date rather than stochastic gradient descent.

## Elo Rating System

In the [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system),
we have a set of entities, $i=1,\cdots,n$, with corresponding ratings $r_1,\cdots,r_n$.
The expected score/outcome of a game between entities $i$ and $j$ is given by
$$
\begin{equation}
\mathbb{E}[y_{ij}|r_1,\cdots,r_n] = \frac{1}{1 + 10^{-(r_i - r_j) / 2\beta}},
\end{equation}
$$
where $\beta$ is a normalisation factor and $y_{ij}$ is the score/outcome of a game between entity $i$ and $j$:
$$
\begin{equation}
y_{ij} = 
\begin{cases}
1 & \text{if}\,i\,\text{wins}, \\\\
0 & \text{if}\,j\,\text{wins}.
\end{cases}
\end{equation}
$$
**NOTE:** we are restricting ourselves to a binary outcome.

After a game, each entity's rating is updated as
$$
\begin{equation}
r^\prime_i=r_i + k \left(y_{ij} - \mathbb{E}[y_{ij}|r_1,\cdots,r_n]\right),
\end{equation}
$$
where $k$ is the *k-factor*. 

## Logistic Regression

Suppose we have a binary outcome $y|\mathbf{x}\sim\text{Bernoulli}(p)$,
where $\mathbf{x}$ is the set of input variables/regressors and $p$ is the outcome probability.
If we model this outcome with model parameters $\mathbf{w}$, the likelihood function is given by
$$
\begin{equation}
\mathcal{L}(\mathbf{w})=
p(y|\mathbf{w};\mathbf{x})^y\left(1-p(y|\mathbf{w};\mathbf{x})\right)^{1-y}.
\end{equation}
$$

The logistic regression model assumes the outcome probability is given by
$$
\begin{equation}
p(y|\mathbf{w};\mathbf{x})=
\frac{1}{1 + e^{-\mathbf{w}\cdot\mathbf{x}}}.
\end{equation}
$$
We have not included an intercept.

Suppose each observation corresponds to a game between two of $n$ entities
and we have one regressor for each entity.
For each game we (arbitrarily) define an order for the entities
- let the first entity be denoted by $i$ and the second by $j$.
For each game, the regressor corresponding to entity $i$ takes value $1$ and the regressor corresponding to entity $j$ takes value $-1$.
Regressors corresponding to other entities take value $0$.
We can represent this as

$$
\begin{equation}
\mathbf{x}_k=\delta_{ik} - \delta_{jk},
\end{equation}
$$

where $\delta_{ij}$ is the *Kronecker delta*.

The table below shows how the data would look like for two games:

- team $1$ beats team $2$ on 01/01/24,
- team $n$ beats team $1$ on 02/01/24.

| Date     | y    | Team 1 | Team 2 | Team 3 | ... | Team n |
|----------|------|--------|--------|--------|-----|--------|
| 01/01/24 | 1    | 1      | -1     | 0      | ... | 0      |
| 02/01/24 | 0    | -1     | 0      | 0      | ... | 1      |
| ...      | ...  | ...    | ...    | ...    | ... | ...    |

We can equivalently represent the games as

| Date     | y   | Team 1 | Team 2  | Team 3 | ... | Team n |
|----------|-----|--------|---------|--------|-----|--------|
| 01/01/24 | 0   | -1     | 1       | 0      | ... | 0      |
| 02/01/24 | 1   | 1      | 0       | 0      | ... | -1     |
| ...      | ... | ...    | ...     | ...    | ... | ...    |

*i.e.* flipping the outcome variable and changing the sign of the regressors.

With the regressors described above, we can rewrite (5) as
$$
\begin{equation}
p(y_{ij}|\mathbf{w};\mathbf{x})=
\frac{1}{1 + e^{-(w_i - w_j)}},
\end{equation}
$$
where, as in (2), $y_{ij}$ represents the score/outcome of a game between team $i$ and $j$.
If we define $r_i:=2\beta w_i/\ln 10$ then we recover the Elo expected score/outcome equation (1).

## Stochastic Gradient Descent

[Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD)
is a commonly used optimisation method.
It is an approximation of [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) 
and can be used to perform maximum likelihood estimation (MLE) for logistic regression.
SGD approximates the gradient descent process by updating parameters based on *one* sample at time
(or a small set of samples at a time, which is sometimes called *mini-batch* gradient descent).
This is typically used when it is computationally infeasible to fit the model using all samples at once.

Gradient descent identifies the direction in parameter space in which a loss function decreases fastest.
It updates the parameters by stepping in this direction.
We can write this as
$$
\begin{equation}
\mathbf{w}^t=\mathbf{w}^{t-1} - \alpha\nabla_{\mathbf{w}} L,
\end{equation}
$$
where $L$ is the loss function to be optimized,
$\alpha$ is the *step size*,
$\mathbf{w}^t$ are the model parameters at step $t$
and $\nabla_{\mathbf{w}}$ is the gradient with respect to the model parameters.

The update method for SGD is given by
$$
\begin{equation}
\mathbf{w}^t=\mathbf{w}^{t-1} - \alpha\nabla_{\mathbf{w}}^{a} L,
\end{equation}
$$
where $\nabla_{\mathbf{w}}^{a}$ is the gradient with respect to the model parameters, 
evaluated for observation $a$ which is usually randomly selected.
For mini-batch gradient descent, $a$ represents a *set* of samples.

### SGD for Logistic Regression

We can perform gradient descent for logistic regression by minimising the negative log-likelihood.
The gradient of the negative log likelihood for logistic regression is given by
$$
\begin{align}
\nabla_{\mathbf{w}}\mathcal l &=
-\sum_a \left[
y_a \frac{\nabla_{\mathbf{w}}p(y_a|\mathbf{w};\mathbf{x_a})}{p(y_a|\mathbf{w};\mathbf{x_a})}
-\frac{(1 - y_a)(\nabla_{\mathbf{w}}p(y|\mathbf{w};\mathbf{x}_a)}{1 - p(y_a|\mathbf{w};\mathbf{x}_a)}
\right],\\\\
&=-\sum_a\left[
y_a(1 - p(y_a|\mathbf{w};\mathbf{x}_a))
- (1 - y_a)p(y_a|\mathbf{w};\mathbf{x}_a))
\right]\mathbf{x}_a,\\\\
&=\sum_a\left[p(y_a|\mathbf{w};\mathbf{x}_a) - y_a\right]\mathbf{x}_a,
\end{align}
$$
where $p(y_a|\mathbf{w};\mathbf{x}_a)$ is the logistic function (5),
$l$ is the negative log-likelihood and $a$ runs over the set of observations/games.

The update method for the model parameters is then given by
$$
\begin{equation}
\mathbf{w}^t
=\mathbf{w}^{t-1} + \alpha \sum_a \left(y_a - p(y|\mathbf{w};\mathbf{x}_a)\right).
\end{equation}
$$

If we have the regressors described in the [Logistic Regression section](intro.md#logistic-regression) above
and restrict ourselves to stochastic gradient descent, 
this becomes
$$
\begin{equation}
\mathbf{w}^t
=\mathbf{w}^{t-1} + \alpha \left(y_{ij} - \frac{1}{1 + e^{-(w_i - w_j)}}\right).
\end{equation}
$$
Using $r_i=2\beta w_i/\ln 10$, this becomes
$$
\begin{equation}
\mathbf{r}^t
=\mathbf{r}^{t-1} + \frac{2\alpha\beta}{\ln10} \left(y_{ij} - \frac{1}{1 + 10^{-(r_i - r_j) / 2\beta}}\right).
\end{equation}
$$
Defining $k:=2\alpha\beta / \ln10$, we recover (3) and the Elo rating system.

## References

This package was inspired by reading [this blog](https://stmorse.github.io/journal/Elo.html).
