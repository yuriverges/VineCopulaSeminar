import numpy as np
import matplotlib.pyplot as plt


def compute_pnl_evolution(w, log_returns, normalize_quote=True):

    if normalize_quote:
        return np.sum(w * np.exp(np.cumsum(log_returns, axis=0)), axis=1) / np.sum(w)
    else:
        return np.sum(w * np.exp(np.cumsum(log_returns, axis=0)), axis=1)


def plot_pnl_evolution(w, scenarios, set_size_inches=None):
    if set_size_inches is None:
        set_size_inches = [20, 20]

    fig, ax = plt.subplots()

    for k, scenario in enumerate(scenarios):
        ax.plot(compute_pnl_evolution(w, scenario), color='gray')

    fig.set_size_inches(*set_size_inches)

    return fig, ax
