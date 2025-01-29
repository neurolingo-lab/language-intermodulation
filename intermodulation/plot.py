from functools import partial

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def snr_topo(
    snrs: np.ndarray,
    epochs: mne.Epochs,
    freqs: np.ndarray,
    fmin: float | None = None,
    fmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    vlines: list | None = None,
):
    if ymin is None:
        ymin = snrs.min()
    if ymax is None:
        ymax = snrs.max()
    if fmin is None:
        fmin = freqs.min()
    if fmax is None:
        fmax = freqs.max()

    def plotcallback(ax, ch_idx):
        ax.plot(freqs, snrs[ch_idx], color="w")
        if vlines is not None:
            for vline in vlines:
                ax.axvline(vline, color="w", linestyle="--", alpha=0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(fmin, fmax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("SNR")

    fig = plt.figure(figsize=(16, 16), dpi=600)
    itertopo = mne.viz.iter_topography(epochs.info, on_pick=plotcallback, fig=fig)

    for ax, idx in itertopo:
        mne.Epochs._keys_to_idx
        ax.plot(freqs, snrs[idx], color="w", lw=0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(fmin, fmax)
        if vlines is not None:
            for vline in vlines:
                ax.axvline(vline, color="w", linestyle="--", alpha=0.5, lw=0.2)

    fig.show()
    return fig


def itc_wholetrial_topo(
    itcs: pd.DataFrame,
    info: mne.Info,
    fmin: float | None = None,
    fmax: float | None = None,
    vlines: None | list = None,
    picks: list[str] | None = None,
):
    if picks is None:
        picks = []
    def plotcallback(ax, ch_idx):
        ax.plot(itcs.columns.to_numpy(), itcs.iloc[ch_idx], color="w")
        if vlines is not None:
            for vline in vlines:
                ax.axvline(vline, color="r", linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_xlim(fmin, fmax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("ITC")

    fig = plt.figure(figsize=(16, 16), dpi=600)
    itertopo = mne.viz.iter_topography(info, on_pick=plotcallback, fig=fig)

    for ax, idx in itertopo:
        if itcs.index[idx] not in picks:
            continue
        ax.plot(itcs.columns.to_numpy(), itcs.iloc[idx], color="white", lw=1.0)
        ax.set_ylim(0, 1)
        ax.set_xlim(fmin, fmax)
        if vlines is not None:
            for vline in vlines:
                ax.axvline(vline, color="w", linestyle="--")
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


def plot_snr(psds, snrs, freqs, fmin, fmax, titleannot="", fig=None, axes=None, tagfreq=None):
    if len(titleannot) > 0:
        titleannot = ": " + titleannot
    if fig is None:
        fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
    freq_range = range(
        np.where(np.floor(freqs) == fmin)[0][0], np.where(np.ceil(freqs) == fmax - 1)[0][0]
    )

    psds_plot = 10 * np.log10(psds)
    psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
    psds_std = psds_plot.std(axis=(0, 1))[freq_range]
    axes[0].plot(freqs[freq_range], psds_mean, color="b")
    axes[0].fill_between(
        freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
    )
    axes[0].set(title="PSD spectrum" + titleannot, ylabel="Power Spectral Density [dB]")

    # SNR spectrum
    snr_mean = snrs.mean(axis=(0, 1))[freq_range]
    snr_std = snrs.std(axis=(0, 1))[freq_range]

    axes[1].plot(freqs[freq_range], snr_mean, color="r")
    axes[1].fill_between(
        freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.2
    )
    axes[1].set(
        title="SNR spectrum" + titleannot,
        xlabel="Frequency [Hz]",
        ylabel="SNR",
        ylim=[-2, 30],
        xlim=[fmin, fmax],
    )
    if tagfreq is not None:
        axes[0].vlines(tagfreq, *axes[0].get_ylim(), color="r", linestyle="--")
        axes[1].vlines(tagfreq, 0, 8, color="r", linestyle="--")

    axes[1].set_xlim([5, 25])
    axes[1].set_ylim([0, 8])

    fig.show()
    return fig, axes
