from typing import List, no_type_check

import pandas as pd


class HistoryPlotterMixin:
    """
    A mixin class for recording and plotting the history of entity ratings.

    Attributes
    ----------
    rating_history : List[Tuple[Optional[int], float]]
        Historical ratings of entities (if track_rating_history is True).

    Methods
    -------
    plot(entities, ax=None, **kwargs)
        Plot the rating history of specified entities.
    """

    @no_type_check
    def plot(self, entities: List[str], ax=None, **kwargs):
        """
        Plot the rating history of specified entities.

        Parameters
        ----------
        entities : List[str]
            A list of player names whose rating histories are to be plotted.
        ax : matplotlib.axes.Axes, optional
            A Matplotlib Axes object to plot on. If None, a new figure and axes
            are created.
        **kwargs
            Keyword arguments to pass to Axe.plot

        Returns
        -------
        matplotlib.axes.Axes
            The Matplotlib Axes object with the plotted rating histories.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        for player in entities:
            ax.plot(
                pd.to_datetime([p[0] for p in self.rating_history[player]]),
                [p[1] for p in self.rating_history[player]],
                label=player,
                **kwargs,
            )
        ax.set(
            title="Elo rating over time",
            xlabel="Date",
            ylabel="Elo Rating",
        )
        ax.legend()

        return ax