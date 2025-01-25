import mne
import numpy as np
import pandas as pd


def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.

    Notes
    -----
    Taken from MNE documentation example for SSVEP at
    https://mne.tools/stable/auto_tutorials/time-freq/50_ssvep.html
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs),
    ))
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise


def itc_epochs(
    epochs: mne.Epochs,
    fmin: float,
    fmax: float,
    tmin: float = None,
    tmax: float = None,
    n_jobs=-1,
):
    picks = mne.pick_types(epochs.info, meg=True, exclude=[])
    psd = epochs.compute_psd(
        picks=picks,
        method="welch",
        n_fft=int(epochs.info["sfreq"] * (tmax - tmin)),
        n_overlap=0,
        n_per_seg=None,
        tmin=tmin,
        tmax=tmax,
        fmin=fmin,
        fmax=fmax,
        window="boxcar",
        output="complex",
        n_jobs=n_jobs,
    )
    ch_names = epochs[0].copy().pick(picks, exclude="bads").ch_names
    psds, freqs = psd.get_data(picks=picks, return_freqs=True)
    angles = np.angle(psds)
    coherences = np.abs(np.mean(np.exp(1j * angles), axis=0))
    coherences = pd.DataFrame(coherences, index=np.array(ch_names)[picks], columns=freqs)
    return coherences
