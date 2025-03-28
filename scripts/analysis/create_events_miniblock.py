from argparse import ArgumentParser

import mne
import mne_bids
import numpy as np
import pandas as pd

from intermodulation.freqtag_spec import LUT_TRIGGERS

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bids_root", type=str, required=True)
    parser.add_argument("--subject_id", type=str, required=True)
    parser.add_argument("--session_id", type=str, required=True)
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--task", type=str, default="syntaxIM")
    parser.add_argument("--proc", type=str, default=None)
    parser.add_argument("--stim_channel", type=str, default="STI102")
    parser.add_argument("--ev-offset", type=int, default=0)
    parser.add_argument("--miniblock-events", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--show-channels", action="store_true")
    parser.add_argument("--overwrite-tsv", action="store_true")
    parser.add_argument("--overwrite-fif", action="store_true")
    parser.add_argument("--stepfix", action="store_true")
    args = parser.parse_args()

    if args.show_channels and not args.interactive:
        raise ValueError("Cannot show channels without interactive mode!")

    deriv = args.proc is not None

    bids_path = mne_bids.BIDSPath(
        subject=args.subject_id,
        session=args.session_id,
        task=args.task,
        run=args.run,
        processing=args.proc,
        root=args.bids_root,
        datatype="meg",
        suffix="meg" if not deriv else "raw",
        check=True if not deriv else False,
    )

    raw = mne.io.read_raw_fif(bids_path.copy().update(extension=".fif", split="01"), preload=True)
    events = mne.find_events(
        raw,
        stim_channel=args.stim_channel,
        output="onset",
        consecutive="increasing",
        shortest_event=1,
    )
    evdf = pd.DataFrame(events, columns=["sample", "v1", "v2"])
    if args.stepfix:
        for i in evdf.index:
            if evdf.at[i, "v1"] != 0:
                evdf.at[i - 1, "v2"] = evdf.at[i, "v2"]
        evdf = evdf[evdf["v1"] == 0].copy().reset_index()
    lut_triggers = {k: "/".join(v) for k, v in LUT_TRIGGERS.items()}
    if args.miniblock_events:
        for k, v in lut_triggers.items():
            if v.split("/")[0] in ("ONEWORD", "TWOWORD"):
                lut_triggers[k + 100] = "MINIBLOCK/" + v

    evdf["label1"] = evdf["v1"].map(lut_triggers)
    evdf["label2"] = evdf["v2"].map(lut_triggers)
    evdf["last"] = np.roll(evdf["label2"].values, 1)

    records = []
    offset = raw.first_samp / raw.info["sfreq"]
    newevs = []
    miniblock_onset = None
    miniblock_sample = None
    for row in evdf.itertuples():
        if row.label2 in ("STATEEND", "TRIALEND", "BLOCKEND"):
            continue
        try:
            end = evdf.at[row.Index + 1, "sample"] / raw.info["sfreq"] + offset
        except KeyError:
            end = raw.times[-1]
        onset = row.sample / raw.info["sfreq"] + offset
        if args.miniblock_events and row.label2.split("/")[0] in ("ONEWORD", "TWOWORD"):
            if row.last == "FIXATION":
                miniblock_onset = onset
                miniblock_sample = row.sample
            elif row.last == row.label2 and evdf.at[row.Index + 1, "label2"] != row.label2:
                if miniblock_sample is None:
                    raise ValueError("MINIBLOCK start not found!")
                records.append({
                    "onset": miniblock_onset,
                    "duration": onset - miniblock_onset,
                    "trial_type": "MINIBLOCK/" + row.label2,
                    "value": 100 + row.v2,
                    "sample": miniblock_sample,
                    "channel": args.stim_channel,
                })
                newevs.append(
                    np.array([miniblock_sample + args.ev_offset, 0, 100 + row.v2]).reshape(1, 3)
                )
                miniblock_onset = None
                miniblock_sample = None

        records.append({
            "onset": onset,
            "duration": end - onset,
            "trial_type": row.label2,
            "value": row.v2,
            "sample": row.sample,
            "channel": args.stim_channel,
        })
        newevs.append(np.array([row.sample + args.ev_offset, 0, row.v2]).reshape(1, 3))
    newevs = np.concatenate(newevs, axis=0)
    bidsevs = pd.DataFrame.from_records(records)
    bidevpath = bids_path.copy().update(suffix="events", extension=".tsv")

    if args.interactive:
        annot = mne.annotations_from_events(
            events=newevs,
            event_desc=lut_triggers,
            sfreq=raw.info["sfreq"],
            first_samp=raw.first_samp,
        )
        raw.set_annotations(annot)
        raw.plot(
            picks=["MISC010", args.stim_channel] if not args.show_channels else "all",
            events=newevs,
            event_id={v: k for k, v in lut_triggers.items()},
            use_opengl=True,
        )
        good = input("Do the new events look right? Y/N")
        if good.lower() != "y":
            raise ValueError("Events not approved!")

    if not len(bidevpath.match()) == 0 and not args.overwrite_fif:
        oldev = pd.read_csv(bidevpath.fpath, sep="\t")
        catevs = pd.concat([oldev, bidsevs], axis=0).reset_index(drop=True)
        if catevs.duplicated().any():
            if not args.overwrite_tsv:
                raise ValueError("Duplicate events found after concatenation!")
            else:
                catevs.drop_duplicates(inplace=True)
        catevs.sort_values("onset", inplace=True)
        catevs.to_csv(bidevpath.fpath, sep="\t", index=False)
    if args.overwrite_fif:
        if deriv:
            raw.save(
                bids_path.copy().update(extension=".fif").fpath,
                split_naming="bids",
                overwrite=True,
            )
        else:
            mne_bids.write_raw_bids(
                raw, bids_path, overwrite=True, allow_preload=True, format="FIF"
            )
