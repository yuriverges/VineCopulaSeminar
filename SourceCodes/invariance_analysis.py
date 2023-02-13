import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def quick_invariance_analysis(serie: pd.Series, n_chunks=1, nbins=10, histogram_density=True, factor=1.1, **kwargs):

    df = pd.concat([serie, serie.shift(1)], axis=1)
    nobs = df.shape[0]
    df.columns = ["data", "lagged_data"]
    df["chunk_index"] = 0

    # Fillnan with 0 - TO avoid OLS problem
    df = df.fillna(0)

    nobs_remainder = nobs % n_chunks
    nobs_chunk = (nobs - nobs_remainder) / n_chunks

    # Defining Chunks
    for k in range(n_chunks):
        if k > 0:
            df.iloc[int(k * nobs_chunk): int((k + 1) * nobs_chunk), -1] = k
        elif k > 0 and k == n_chunks - 1:
            df.iloc[int(k * nobs_chunk):, -1] = k

    # Creating figure
    fig, axs = plt.subplots(n_chunks, 2, sharex=True)

    if n_chunks == 1:
        axs[0].scatter(df.loc[df["chunk_index"] == k, "lagged_data"], df.loc[df["chunk_index"] == k, "data"])
        axs[0].set_xlim(serie.min(), serie.max())
        axs[0].set_ylim(serie.min(), serie.max())
        axs[0].set_title(f"Scatter - {serie.name} - Chunk Index{k}")
        axs[0].set(xlabel='Lagged Data', ylabel='Data')

        m, b = np.polyfit(df.loc[df["chunk_index"] == k, "lagged_data"], df.loc[df["chunk_index"] == k, "data"], 1)
        axs[0].plot(df.loc[df["chunk_index"] == k, "lagged_data"], m * df.loc[df["chunk_index"] == k, "lagged_data"] + b, color="green")

        axs[1].hist(df.loc[df["chunk_index"] == k, "data"], range=(serie.min(), serie.max()), bins=nbins,
                    density=histogram_density)
        axs[1].set_title(f"Histogram - Chunk Index{k}")
        axs[1].set(xlabel='Data')

    else:
        for k in range(n_chunks):
            axs[k, 0].scatter(df.loc[df["chunk_index"] == k, "lagged_data"], df.loc[df["chunk_index"] == k, "data"])
            axs[k, 0].set_xlim(serie.min(), serie.max())
            axs[k, 0].set_ylim(serie.min(), serie.max())
            axs[k, 0].set_title(f"Scatter - {serie.name} - Chunk Index{k}")
            axs[k, 0].set(xlabel='Lagged Data', ylabel='Data')

            m, b = np.polyfit(df.loc[df["chunk_index"] == k, "lagged_data"], df.loc[df["chunk_index"] == k, "data"], 1)
            axs[k, 0].plot(df.loc[df["chunk_index"] == k, "lagged_data"],
                           m * df.loc[df["chunk_index"] == k, "lagged_data"] + b, color="green")

            axs[k, 1].hist(df.loc[df["chunk_index"] == k, "data"], range=(serie.min(), serie.max()), bins=nbins,
                           density=histogram_density)
            axs[k, 1].set_title(f"Histogram - Chunk Index{k}")
            axs[k, 1].set(xlabel='Data')

        for ax in axs.flat:
            ax.label_outer()

    return fig, axs
