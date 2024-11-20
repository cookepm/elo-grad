import numpy as np
import pandas as pd
import pytest

from elo_grad import LogisticRegression, SGDOptimizer, EloEstimator, Regressor, PoissonRegression, PoissonEloEstimator


class TestLogisticRegression:
    def test_calculate_expected_score_equal_ratings(self):
        model = LogisticRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        assert model.calculate_expected_score(1, -1) == 0.5

    def test_calculate_expected_score_higher_rating(self):
        model = LogisticRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        assert model.calculate_expected_score(2, -1) > 0.5

    def test_calculate_expected_score_inverse(self):
        model_1 = LogisticRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        model_2 = LogisticRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        assert model_1.calculate_expected_score(1, -1) == model_2.calculate_expected_score(-1, 1)


class TestPoissonRegression:
    def test_calculate_expected_score_equal_ratings(self):
        model = PoissonRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        assert model.calculate_expected_score(1, -1) == 1

    def test_calculate_expected_score_higher_rating(self):
        model = PoissonRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        assert model.calculate_expected_score(2, -1) > 1

    def test_calculate_expected_score_inverse(self):
        model_1 = PoissonRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        model_2 = PoissonRegression(
            default_init_rating=1,
            init_ratings=None,
            beta=1,
        )
        assert model_1.calculate_expected_score(1, -1) == model_2.calculate_expected_score(-1, 1)


class TestSGDOptimizer:

    def test_calculate_update_step(self):
        model_1 = LogisticRegression(
            default_init_rating=1000,
            init_ratings=dict(entity_1=(None, 1500), entity_2=(None, 1600)),
            beta=200,
        )
        opt_1 = SGDOptimizer(k_factor=32, additional_regressors=None)
        updates_1 = opt_1.calculate_update_step(model_1, 1, "entity_1", "entity_2", None, None)
        entity_update_1 = next(updates_1)
        with pytest.raises(StopIteration):
            next(updates_1)

        assert round(entity_update_1, 2) == 20.48

        model_2 = LogisticRegression(
            default_init_rating=1000,
            init_ratings=dict(entity_2=(None, 1600)),
            beta=200,
        )
        opt_2 = SGDOptimizer(k_factor=20, additional_regressors=None)
        updates_2 = opt_2.calculate_update_step(model_2, 0, "entity_1", "entity_2", None, None)
        entity_update_2 = next(updates_2)

        assert round(entity_update_2, 2) == -0.61

    def test_calculate_update_step_additional_regressors(self):
        model_1 = LogisticRegression(
            default_init_rating=1000,
            init_ratings=dict(entity_1=(None, 1500), entity_2=(None, 1600), home=(None, 10)),
            beta=200,
        )
        opt_1 = SGDOptimizer(k_factor=32, additional_regressors=[Regressor(name="home", k_factor=1)])
        updates_1 = opt_1.calculate_update_step(model_1, 1, "entity_1", "entity_2", (1,), None)
        entity_update_1 = next(updates_1)
        home_update_1 = next(updates_1)
        with pytest.raises(StopIteration):
            next(updates_1)

        assert round(entity_update_1, 2) == 20.05
        assert round(home_update_1, 2) == 0.63

        model_2 = LogisticRegression(
            default_init_rating=1000,
            init_ratings=dict(entity_2=(None, 1600), home=(None, 20)),
            beta=200,
        )
        opt_2 = SGDOptimizer(k_factor=20, additional_regressors=[Regressor(name="regressor_1", k_factor=1)])
        updates_2 = opt_2.calculate_update_step(model_2, 0, "entity_1", "entity_2", (0,), None)
        entity_update_2 = next(updates_2)
        home_update_2 = next(updates_2)

        assert round(entity_update_2, 2) == -0.61
        assert round(home_update_2, 2) == 0.0

    def test_calculate_update_step_regularisation(self):
        model = LogisticRegression(
            default_init_rating=1000,
            init_ratings=dict(entity_1=(None, 1500), entity_2=(None, 1600), home=(None, 10)),
            beta=200,
        )
        opt_1 = SGDOptimizer(
            k_factor=32,
            additional_regressors=[Regressor(name="home", k_factor=1, penalty="l1", lambda_reg=0.1)],
        )
        updates_1 = opt_1.calculate_update_step(
            model=model,
            y=1,
            entity_1="entity_1",
            entity_2="entity_2",
            additional_regressor_values=(1,),
            expected_score=None,
        )
        entity_update_1 = next(updates_1)
        home_update_1 = next(updates_1)
        with pytest.raises(StopIteration):
            next(updates_1)

        assert round(entity_update_1, 2) == 20.05
        assert round(home_update_1, 2) == 0.53


        opt_2 = SGDOptimizer(
            k_factor=32,
            additional_regressors=[Regressor(name="home", k_factor=1, penalty="l2", lambda_reg=0.1)],
        )
        updates_2 = opt_2.calculate_update_step(
            model=model,
            y=1,
            entity_1="entity_1",
            entity_2="entity_2",
            additional_regressor_values=(1,),
            expected_score=None,
        )
        entity_update_2 = next(updates_2)
        home_update_2 = next(updates_2)
        with pytest.raises(StopIteration):
            next(updates_2)

        assert round(entity_update_2, 2) == 20.05
        assert round(home_update_2, 2) == -1.37


        opt_3 = SGDOptimizer(
            k_factor=32,
            additional_regressors=[Regressor(name="home", k_factor=1, penalty="l2", lambda_reg=0.0)],
        )
        updates_3 = opt_3.calculate_update_step(
            model=model,
            y=1,
            entity_1="entity_1",
            entity_2="entity_2",
            additional_regressor_values=(1,),
            expected_score=None,
        )
        entity_update_3 = next(updates_3)
        home_update_3 = next(updates_3)
        with pytest.raises(StopIteration):
            next(updates_3)

        assert round(entity_update_3, 2) == 20.05
        assert round(home_update_3, 2) == 0.63


    def test_calculate_gradient_raises(self):
        model = LogisticRegression(
            default_init_rating=1000,
            init_ratings=None,
            beta=200,
        )
        opt = SGDOptimizer(k_factor=20, additional_regressors=None)
        with pytest.raises(ValueError, match="Invalid score value"):
            next(
                opt.calculate_update_step(model, -1, "entity_1", "entity_2", (0,), None)
            )


class TestEloEstimator:

    estimator = EloEstimator(k_factor=20, default_init_rating=1200)

    def test_transform_raises(self):
        with pytest.raises(TypeError, match="Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe"):
            self.estimator.fit(1)

        df = pd.DataFrame(
            columns=["t", "entity_1", "entity_2", "score"],
            index=[3, 2, 1],
        )
        with pytest.raises(ValueError, match="DataFrame must be sorted by date."):
            self.estimator.fit(df)

    def test_transform(self):
        df = pd.DataFrame(
            data=[
                (1, "A", "B", 1),
                (2, "A", "C", 1),
                (3, "B", "C", 0),
                (4, "C", "A", 0),
                (4, "C", "B", 1),
            ],
            columns=["t", "entity_1", "entity_2", "score"],
        )

        expected_arr = np.array([0.5, 0.51, 0.5, 0.47, 0.53])

        output_arr = self.estimator.predict_proba(df)[:, 1]

        # Check expected scores
        np.testing.assert_allclose(expected_arr, output_arr, atol=1e-2)

        # Check ratings
        expected_ratings = {
            "A": (2, 1200 + 10 + 9.7123 + 9.4413),
            "B": (1, 1200 - 10 - 9.9917 - 9.4172),
            "C": (2, 1200 - 9.7123 + 9.9917 - 9.4413 + 9.4172),
        }
        for k, v in self.estimator.model.ratings.items():
            assert round(expected_ratings[k][1], 2) == round(v[1], 2)

    def test_transform_w_additional_regressors(self):
        estimator = EloEstimator(
            k_factor=20,
            default_init_rating=1200,
            init_ratings=dict(home=(None, 0)),
            additional_regressors=[Regressor(name="home", k_factor=1)],
        )
        df = pd.DataFrame(
            data=[
                (1, "A", "B", 1, 1),
                (2, "A", "C", 1, 1),
                (3, "B", "C", 0, 0),
                (4, "C", "A", 0, 0),
                (4, "C", "B", 1, 1),
            ],
            columns=["t", "entity_1", "entity_2", "score", "home"],
        )

        expected_arr = np.array([0.5, 0.52, 0.5, 0.47, 0.53])

        output_arr = estimator.predict_proba(df)[:, 1]

        # Check expected scores
        np.testing.assert_allclose(expected_arr, output_arr, atol=1e-2)

        # Check ratings
        expected_ratings = {
            "A": (2, 1200 + 10 + 9.6979 + 9.4421),
            "B": (1, 1200 - 10 - 9.9913 - 9.3886),
            "C": (2, 1200 - 9.6979 + 9.9913 - 9.4421 + 9.3886),
            "home": (2, 0 + 0.5 + 0.4849 + 0.4694)
        }
        for k, v in estimator.model.ratings.items():
            assert round(expected_ratings[k][1], 2) == round(v[1], 2)

    def test_transform_w_regularisation(self):
        estimator = EloEstimator(
            k_factor=20,
            default_init_rating=1200,
            init_ratings=dict(home=(None, 0)),
            additional_regressors=[Regressor(name="home", k_factor=1, penalty="l1", lambda_reg=0.1)],
        )
        df = pd.DataFrame(
            data=[
                (1, "A", "B", 1, 1),
                (2, "A", "C", 1, 1),
                (3, "B", "C", 0, 0),
                (4, "C", "A", 0, 0),
                (4, "C", "B", 1, 1),
            ],
            columns=["t", "entity_1", "entity_2", "score", "home"],
        )

        expected_arr = np.array([0.5, 0.515, 0.4996, 0.4721, 0.5301])

        output_arr = estimator.predict_proba(df)[:, 1]

        # Check expected scores
        np.testing.assert_allclose(expected_arr, output_arr, atol=1e-2)

        # Check ratings
        expected_ratings = {
            "A": (2, 1200 + 10 + 9.7008 + 9.4419),
            "B": (1, 1200 - 10 - 9.9914 - 9.3973),
            "C": (2, 1200 - 9.7008 + 9.9914 - 9.4419 + 9.3973),
            "home": (2, 0.0 + 0.4 + 0.385 - 0.1 - 0.1 + 0.3699)
        }
        for k, v in estimator.model.ratings.items():
            assert round(expected_ratings[k][1], 2) == round(v[1], 2)


class TestPoissonEloEstimator:

    estimator = PoissonEloEstimator(k_factor=20, default_init_rating=1200)

    def test_transform_raises(self):
        with pytest.raises(TypeError, match="Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe"):
            self.estimator.fit(1)

        df = pd.DataFrame(
            columns=["t", "entity_1", "entity_2", "score"],
            index=[3, 2, 1],
        )
        with pytest.raises(ValueError, match="DataFrame must be sorted by date."):
            self.estimator.fit(df)

    def test_transform(self):
        df = pd.DataFrame(
            data=[
                (1, "A", "B", 1),
                (2, "A", "C", 2),
                (3, "B", "C", 0),
                (4, "C", "A", 0),
                (4, "C", "B", 4),
            ],
            columns=["t", "entity_1", "entity_2", "score"],
        )

        expected_arr = np.array([1.0, 1.0, 1.122, 0.9039, 1.154])

        output_arr = self.estimator.predict(df)

        # Check expected scores
        np.testing.assert_allclose(expected_arr, output_arr, atol=1e-2)

        # Check ratings
        expected_ratings = {
            "A": (2, 1200 + 0 + 20 + 18.078),
            "B": (1, 1200 - 0 - 22.44 - 56.92),
            "C": (2, 1200 - 20 + 22.44 - 18.078 + 56.92),
        }
        for k, v in self.estimator.model.ratings.items():
            assert round(expected_ratings[k][1], 2) == round(v[1], 2)

    def test_transform_w_additional_regressors(self):
        estimator = PoissonEloEstimator(
            k_factor=20,
            default_init_rating=1200,
            init_ratings=dict(home=(None, 0)),
            additional_regressors=[Regressor(name="home", k_factor=1)],
        )
        df = pd.DataFrame(
            data=[
                (1, "A", "B", 1, 1),
                (2, "A", "C", 2, 1),
                (3, "B", "C", 0, 0),
                (4, "C", "A", 0, 1),
                (4, "C", "B", 4, 1),
            ],
            columns=["t", "entity_1", "entity_2", "score", "home"],
        )

        expected_arr = np.array([1.0, 1.0, 1.122, 0.9091, 1.1606])

        output_arr = estimator.predict(df)

        # Check expected scores
        np.testing.assert_allclose(expected_arr, output_arr, atol=1e-2)

        # Check ratings
        expected_ratings = {
            "A": (2, 1200 + 0 + 20 + 18.182),
            "B": (1, 1200 - 0 - 22.44 - 56.788),
            "C": (2, 1200 - 20 + 22.44 - 18.182 + 56.788),
            "home": (2, 0 + 0 + 1 - 0.9091 + 2.8394)
        }
        for k, v in estimator.model.ratings.items():
            assert round(expected_ratings[k][1], 2) == round(v[1], 2)

    def test_transform_w_regularisation(self):
        estimator = PoissonEloEstimator(
            k_factor=20,
            default_init_rating=1200,
            init_ratings=dict(home=(None, 0)),
            additional_regressors=[Regressor(name="home", k_factor=1, penalty="l2", lambda_reg=0.1)],
        )
        df = pd.DataFrame(
            data=[
                (1, "A", "B", 1, 1),
                (2, "A", "C", 2, 1),
                (3, "B", "C", 0, 0),
                (4, "C", "A", 0, 1),
                (4, "C", "B", 4, 1),
            ],
            columns=["t", "entity_1", "entity_2", "score", "home"],
        )

        expected_arr = np.array([1.0, 1.0, 1.122, 0.908, 1.1593])

        output_arr = estimator.predict(df)

        # Check expected scores
        np.testing.assert_allclose(expected_arr, output_arr, atol=1e-2)

        # Check ratings
        expected_ratings = {
            "A": (2, 1200 + 0.0 + 20.0 + 18.1606),
            "B": (1, 1200 + 0.0 - 22.4404 - 56.8137),
            "C": (2, 1200 - 20.0 + 22.4404 - 18.1606 + 56.8137),
            "home": (2, 0 + 0.0 + 1.0 - 0.2 - 1.068 + 2.6807)
        }
        for k, v in estimator.model.ratings.items():
            assert round(expected_ratings[k][1], 2) == round(v[1], 2)
