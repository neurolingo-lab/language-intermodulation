import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import mne_bids as mnb

import intermodulation.analysis as ima
import intermodulation.plot as imp
from intermodulation import freqtag_spec

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--proc",
        type=str,
        default="raw",
        help="Processing type: raw, sss, filt, or clean",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="02",
        help="Subject ID",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="01",
        help="Session ID",
    )
    parser.add_argument(
        "--bids_root",
        type=Path,
        default="/srv/beegfs/scratch/users/g/gercek/syntax_im/syntax_dataset",
        help="Root directory for BIDS dataset",
    )
    parser.add_argument(
        "--savepath",
        type=Path,
        default="/srv/beegfs/scratch/users/g/gercek/syntax_im/results",
        help="Directory in which to save SNR data",
    )
    args = parser.parse_args()

    processing = None if args.proc == "raw" else args.proc
    procdir = "raw" if args.proc == "raw" else f"proc-{args.proc}"

    # STORAGE LOCATIONS
    derivatives_root = args.bids_root / "derivatives/mne-bids-pipeline"

    args.savepath.mkdir(parents=True, exist_ok=True)
    procpath = args.savepath / f"sub-{args.subject}" / procdir
    procpath.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to {procpath}")
    plotpath = procpath / "figures"
    plotpath.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {plotpath}")

    raw_bidspath = mnb.BIDSPath(
        subject=args.subject,
        session=args.session,
        task="syntaxIM",
        processing=processing,
        split="01" if processing in ("sss", "filt", None) else None,
        datatype="meg",
        suffix="raw" if processing is not None else None,
        extension=".fif",
        root=derivatives_root if processing is not None else args.bids_root,
        check=False,  # Need to disable checking for derivatives
    )

    minidur = freqtag_spec.WORD_DUR * freqtag_spec.MINIBLOCK_LEN

    try:
        raw = mne.io.read_raw_fif(raw_bidspath.fpath, verbose=False)
    except FileNotFoundError:
        raw = mne.io.read_raw_fif(raw_bidspath.copy().update(split=None).fpath, verbose=False)
    events, evdict = mne.events_from_annotations(raw)
    keepev = {k: v for k, v in evdict.items() if k.split("/")[0] == "MINIBLOCK"}
    print("Raw sampled at ", raw.info["sfreq"])
    # if raw.info["sfreq"] == 2000:
    #     decim = 4
    #     print("Decimating to 500Hz from 2000Hz")
    # else:
    #     decim = 1

    epochs = mne.Epochs(raw, event_id=keepev, tmin=-0.2, tmax=minidur, picks="all", verbose=False)
    epochs.load_data()
    del raw

    # Global parameters for different FFTs
    fmin = 0.5
    fmax = 60.0
    tmin = 0.0
    tmax = minidur
    snr_neighbor_freqs = 0.8  # Hz
    snr_skip_freqs = 0.1  # Hz total either side
    snr_skip_neighbor_J = int((snr_skip_freqs / 2) / (1 / (tmax - tmin)))
    snr_neighbor_K = int((snr_neighbor_freqs / 2) / (1 / (tmax - tmin)) - snr_skip_neighbor_J)
    psd_kwargs = dict(
        method="welch",
        n_fft=int(epochs.info["sfreq"] * (tmax - tmin)),
        n_overlap=0,
        n_per_seg=None,
        tmin=tmin,
        tmax=tmax,
        fmin=fmin,
        fmax=fmax,
        window="boxcar",
    )

    print("Computing SNR for oneword+twoword, per condition and all conditions...")
    owbase = f"ses-{args.session}_task-syntaxIM_spectraSNR_oneword"
    twbase = f"ses-{args.session}_task-syntaxIM_spectraSNR_twoword"
    allcond_spectra_ow = {}
    allcond_spectra_tw = {}
    percond_spectra_ow = {}
    percond_spectra_tw = {}
    for tag in ["F1", "F2"]:
        twtag = "F1LEFT" if tag == "F1" else "F1RIGHT"
        spectrum = epochs[f"MINIBLOCK/ONEWORD/{tag}"].compute_psd(
            exclude="bads", n_jobs=-1, verbose=False, **psd_kwargs
        )
        twspectrum = epochs[f"MINIBLOCK/TWOWORD/{twtag}"].compute_psd(
            exclude="bads", n_jobs=-1, verbose=False, **psd_kwargs
        )
        psds, freqs = spectrum.get_data(return_freqs=True)
        twpsds, twfreqs = twspectrum.get_data(return_freqs=True)
        snrs = ima.snr_spectrum(
            psds,
            noise_n_neighbor_freqs=snr_neighbor_K,
            noise_skip_neighbor_freqs=snr_skip_neighbor_J,
        )
        twsnrs = ima.snr_spectrum(
            twpsds,
            noise_n_neighbor_freqs=snr_neighbor_K,
            noise_skip_neighbor_freqs=snr_skip_neighbor_J,
        )
        allcond_spectra_ow[tag] = dict(psds=psds, freqs=freqs, snrs=snrs)
        allcond_spectra_tw[twtag] = dict(psds=twpsds, freqs=twfreqs, snrs=twsnrs)
        for cond in ["WORD", "NONWORD"]:
            fulltag = f"ONEWORD/{cond}/{tag}"
            spectrum = epochs["MINIBLOCK/" + fulltag].compute_psd(
                exclude="bads", n_jobs=-1, **psd_kwargs
            )
            psds, freqs = spectrum.get_data(return_freqs=True)
            snrs = ima.snr_spectrum(
                psds,
                noise_n_neighbor_freqs=snr_neighbor_K,
                noise_skip_neighbor_freqs=snr_skip_neighbor_J,
            )
            percond_spectra_ow[fulltag] = dict(psds=psds, freqs=freqs, snrs=snrs)
        for cond in ["PHRASE", "NONPHRASE", "NONWORD"]:
            fulltag = f"TWOWORD/{cond}/{twtag}"
            spectrum = epochs["MINIBLOCK/" + fulltag].compute_psd(
                exclude="bads", n_jobs=-1, **psd_kwargs
            )
            psds, freqs = spectrum.get_data(return_freqs=True)
            snrs = ima.snr_spectrum(
                psds,
                noise_n_neighbor_freqs=snr_neighbor_K,
                noise_skip_neighbor_freqs=snr_skip_neighbor_J,
            )
            percond_spectra_tw[fulltag] = dict(psds=psds, freqs=freqs, snrs=snrs)
    print("Done. Saving data...")
    with open(
        procpath / f"{owbase}_allcond.pkl",
        "wb",
    ) as f:
        pickle.dump(allcond_spectra_ow, f)
    with open(
        procpath / f"{twbase}_allcond.pkl",
        "wb",
    ) as f:
        pickle.dump(allcond_spectra_tw, f)
    with open(
        procpath / f"{owbase}_percond.pkl",
        "wb",
    ) as f:
        pickle.dump(percond_spectra_ow, f)
    with open(
        procpath / f"{twbase}_percond.pkl",
        "wb",
    ) as f:
        pickle.dump(percond_spectra_tw, f)

    print("Done.\n")

    print("Plotting SNR and SNR topos for oneword+twoword, all conditions combined...")
    plot_freqs = (1, 15)
    topofig_kw = dict(figsize=(8, 8), dpi=200)

    for name, spectra in {"oneword": allcond_spectra_ow, "twoword": allcond_spectra_tw}.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey="row")
        for i, (tag, data) in enumerate(spectra.items()):
            ax = axes[:, i]
            if name == "twoword":
                # Vertical lines at tag frequencies and f2-f1, f1+f2 IMs
                tagfreq = [
                    7.05882353 - 6.0,
                    6.0,
                    7.05882353,
                    2 * 6,
                    6 + 7.05882353,
                    2 * 7.05882353,
                ]
                titlestr = f"{tag} Two-Word SNR"
            else:
                tagfreq = {"F1": 6.0, "F2": 7.05882353}[tag]
                titlestr = f"{tag} One-Word SNR"
            imp.plot_snr(
                data["psds"],
                data["snrs"],
                data["freqs"],
                fmin=plot_freqs[0],
                fmax=plot_freqs[1],
                fig=fig,
                axes=ax,
                titleannot=titlestr,
                tagfreq=tagfreq,
                plotpsd=True,
            )
            # Topomap plots
            topofig = imp.snr_topo(
                data["snrs"].mean(axis=0),
                epochs.pick("data", exclude="bads"),
                data["freqs"],
                fmin=plot_freqs[0],
                fmax=plot_freqs[1],
                ymin=0.0,
                ymax=8.0,
                vlines=[tagfreq] if name == "oneword" else tagfreq,
                fig_kwargs=topofig_kw,
            )
            topofig.suptitle(titlestr + ": All conditions", color="w")
            topofig.savefig(
                plotpath
                / f"sub-{args.subject}_ses-{args.session}_task-syntaxIM_{name}_allconds_{tag}_snrtopo.pdf"
            )
            plt.close(topofig)
        axes[1, 0].set_ylim([-0.5, 4.0])
        fig.savefig(
            plotpath
            / f"sub-{args.subject}_ses-{args.session}_task-syntaxIM_{name}_allconds_snr.pdf"
        )
        plt.close(fig)
    print("Done.")

    print("Plotting SNR and SNR topos for oneword+twoword, per condition...")
    for name, spectra in {"oneword": percond_spectra_ow, "twoword": percond_spectra_tw}.items():
        ncond = len(spectra.keys())
        fig, axes = plt.subplots(2, ncond, figsize=(ncond * 5, 11), sharex=True, sharey="row")
        for i, (tag, data) in enumerate(spectra.items()):
            cond = tag.split("/")[1]
            freq = tag.split("/")[-1]
            if name == "twoword":
                # Vertical lines at tag frequencies and f2-f1, f1+f2 IMs
                tagfreq = [
                    7.05882353 - 6.0,
                    6.0,
                    7.05882353,
                    2 * 6,
                    6 + 7.05882353,
                    2 * 7.05882353,
                ]
                titlestr = f"{cond} SNR"
            else:
                tagfreq = {"F1": 6.0, "F2": 7.05882353}[freq]
                titlestr = f"{cond} SNR"
            ax = axes[:, i]
            imp.plot_snr(
                data["psds"],
                data["snrs"],
                data["freqs"],
                fmin=plot_freqs[0],
                fmax=plot_freqs[1],
                fig=fig,
                axes=ax,
                titleannot=titlestr,
                tagfreq=tagfreq,
                plotpsd=True,
            )
            topofig = imp.snr_topo(
                data["snrs"].mean(axis=0),
                epochs.pick("data", exclude="bads"),
                data["freqs"],
                fmin=plot_freqs[0],
                fmax=plot_freqs[1],
                ymin=0.0,
                ymax=8.0,
                vlines=[tagfreq] if name == "oneword" else tagfreq,
                fig_kwargs=topofig_kw,
            )
            topofig.suptitle(f"{name}: {freq} {cond} trials SNR", color="w")
            topofig.savefig(
                plotpath
                / f"sub-{args.subject}_ses-{args.session}_task-syntaxIM_{name}_{cond}_{freq}_snrtopo.pdf"
            )
            plt.close(topofig)
        axes[1, 0].set_ylim([-0.5, 4.0])
        fig.suptitle(f"{name} SNR per condition", fontsize=16)
        fig.tight_layout()
        fig.savefig(
            plotpath
            / f"sub-{args.subject}_ses-{args.session}_task-syntaxIM_{name}_percond_snr.pdf"
        )
    print("Done.")
