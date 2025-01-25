from functools import partial

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def snr_topo(
    snrs: np.ndarray,
    info: mne.Info,
    freqs: np.ndarray,
    fmin: float | None = None,
    fmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    picks: list[str] | None = None,
):
    if ymin is None:
        ymin = snrs.min()
    if ymax is None:
        ymax = snrs.max()
    if fmin is None:
        fmin = freqs.min()
    if fmax is None:
        fmax = freqs.max()

    show_fun = partial(
        mne.viz.topo._plot_timeseries_unified,
        data=[snrs],
        times=[freqs],
        color="w",
        ylim=(ymin, ymax),
    )

    click_fun = partial(
        mne.viz.topo._plot_timeseries,
        data=[snrs],
        times=[freqs],
        color="w",
    )

    fig = mne.viz.topo._plot_topo(
        info,
        times=(fmin, fmax),
        show_func=show_fun,
        click_func=click_fun,
        layout=mne.channels.find_layout(info),
        unified=True,
        x_label="Frequency (Hz)",
        y_label="SNR",
    )
    fig.show()
    return fig


def itc_wholetrial_topo(
    itcs: pd.DataFrame,
    info: mne.Info,
    fmin: float | None = None,
    fmax: float | None = None,
    vlines: None | list = None,
):
    def plotcallback(ax, ch_idx):
        ax.plot(itcs.columns.to_numpy(), itcs.iloc[ch_idx], color="w")
        if vlines is not None:
            for vline in vlines:
                ax.axvline(vline, color="r", linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_xlim(fmin, fmax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("ITC")

    itertopo = mne.viz.iter_topography(info, on_pick=plotcallback)

    for ax, idx in itertopo:
        ax.plot(itcs.columns.to_numpy(), itcs.iloc[idx], color="white", lw=1.0)
        ax.set_ylim(0, 1)
        ax.set_xlim(fmin, fmax)
        if vlines is not None:
            for vline in vlines:
                ax.axvline(vline, color="w", linestyle="--")
    fig = plt.gcf()
    return fig


def itc_singlefreq_topo(
    itc: mne.time_frequency.AverageTFR,
    data: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    freq: float,
    picks: list[str] | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    vlines: list | None = None,
):
    if picks is None:
        picks = mne.pick_types(itc.info, meg=True)
    if tmin is None:
        tmin = times.min()
    if tmax is None:
        tmax = times.max()

    fidx = np.abs(freqs - freq).argmin()
    fdata = data[:, fidx, :]

    if ymin is None:
        ymin = fdata.min()
    if ymax is None:
        ymax = fdata.max()

    pick_fun = partial(
        mne.viz.topo._plot_timeseries_unified,
        data=[fdata],
        times=[times],
        color="w",
        vline=vlines,
        ylim=(ymin, ymax),
    )

    click_fun = partial(
        mne.viz.topo._plot_timeseries,
        data=[fdata],
        times=[times],
        color="w",
        vline=vlines,
        ylim=(ymin, ymax),
    )

    fig = mne.viz.topo._plot_topo(
        itc.info,
        times=(tmin, tmax),
        show_func=pick_fun,
        click_func=click_fun,
        layout=mne.channels.find_layout(itc.info),
        unified=True,
        x_label="Time (s)",
        y_label="ITC",
    )

    fig.suptitle(f"ITC at {freq:.2f} Hz")
    return fig
