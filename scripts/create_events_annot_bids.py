from argparse import ArgumentParser

import mne
import mne_bids
import numpy as np
import pandas as pd

from intermodulation.freqtag_spec import LUT_TRIGGERS, TRIGGERS, VALID_TRANS, nested_deepkeys

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bids_root", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--session", type=str, required=True)
    parser.add_argument("--task", type=str, default="syntaxIM")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Fully overwrite existing events. If not set, then old events will be written "
        "to a file ending in annotbackup.fif in the same directory as the original file.",
    )
    parser.add_argument("--stim_channel", type=str, default="STI102")
    parser.add_argument("--tmpfix", action="store_true")
    args = parser.parse_args()

    bids_path = mne_bids.BIDSPath(
        subject=args.subject,
        session=args.session,
        task=args.task,
        root=args.bids_root,
        datatype="meg",
        split="01",
    )

    raw = mne.io.read_raw_fif(bids_path.copy().update(suffix="meg", extension=".fif", split="01"))
    events = mne.find_events(
        raw,
        stim_channel=args.stim_channel,
        output="onset",
        consecutive="increasing",
        shortest_event=1,
    )
    evdf = pd.DataFrame(events, columns=["sample", "v1", "v2"])
    lut_triggers = {k: "/".join(v) for k, v in LUT_TRIGGERS.items()}
    evdf["label1"] = evdf["v1"].map(lut_triggers)
    evdf["label2"] = evdf["v2"].map(lut_triggers)
    evdf["last"] = np.roll(evdf["label2"].values, 1)

    if args.tmpfix:
        VALID_TRANS = list(VALID_TRANS)
        VALID_TRANS.extend(
            (("/".join(k), "ITI") for k in nested_deepkeys(TRIGGERS) if k[0] == "ONEWORD")
        )

    evdf["valid"] = evdf.apply(lambda x: (x["last"], x["label2"]) in VALID_TRANS, axis=1)
    remidx = []
    for row in evdf[~evdf["valid"]].itertuples():
        i = row.Index
        if evdf["valid"].loc[i - 1] and evdf["valid"].loc[i + 1]:
            remidx.append(i)
        elif ~evdf["valid"].loc[i + 1] and evdf["valid"].loc[i - 1]:
            remidx.append(i)
        elif ~evdf["valid"].loc[i - 1]:
            pass
        else:
            raise ValueError(f"Unhandled transition error at index {i}")
    goodevs = evdf.drop(remidx)
    goodevs["last"] = np.roll(goodevs["label2"].values, 1)
    goodevs["valid"] = goodevs.apply(lambda x: (x["last"], x["label2"]) in VALID_TRANS, axis=1)
    if not goodevs["valid"].all():
        raise ValueError(
            "Invalid transitions remain after cleanup. Remaining bad indices: ",
            np.argwhere(~goodevs["valid"]),
        )

    goodevs.reset_index(inplace=True, drop=True)
    records = []
    metadata = []
    offset = raw.first_samp / raw.info["sfreq"]
    for row in goodevs.itertuples():
        if row.label2 in ("STATEEND", "TRIALEND", "BLOCKEND"):
            continue
        if goodevs.at[row.Index + 1, "label2"] in ("STATEEND", "TRIALEND", "BLOCKEND"):
            end = goodevs.at[row.Index + 1, "sample"] / raw.info["sfreq"] - offset
        else:
            if args.tmpfix and row.label2.split("/")[0] == "ONEWORD":
                end = goodevs.at[row.Index + 1, "sample"] / raw.info["sfreq"] - offset
            else:
                raise ValueError(
                    f"Expected end of trial at index {row.Index + 1}, found "
                    f"{goodevs.at[row.Index + 1, 'label2']}"
                )
        onset = row.sample / raw.info["sfreq"] - offset
        records.append([row.sample - raw.first_samp, 0, row.v2])
        labels = row.label2.split("/")
        complexstate = len(labels) > 1
        stimstate = len(labels) > 2
        metadata.append(
            {
                "onset": onset,
                "duration": end - onset,
                "label": row.label2,
                "orig_samp": row.sample,
                "state": labels[0],
                "cond": labels[1] if complexstate else None,
                "freq": labels[2] if stimstate else None,
            }
        )
    # bidsevs = pd.DataFrame.from_records(records)
    bidsevs = np.array(records)
    ev_meta = pd.DataFrame.from_records(metadata)
    if not args.overwrite:
        raw.annotations.save(
            bids_path.copy().update(suffix="annotbackup", extension=".fif", check=False).fpath,
        )
    mne_bids.write_raw_bids(
        raw,
        bids_path,
        bidsevs,
        event_id={v: k for k, v in lut_triggers.items()},
        event_metadata=ev_meta,
        extra_columns_descriptions={
            "onset": "onset time of state",
            "duration": "duration of state",
            "label": "label sent to MNE for that state",
            "orig_samp": "original sample number before applying first sample correction",
            "state": "top-level identity of the state",
            "cond": "sub-condition of the state, such as type of stimulus or query ground truth",
            "freq": "frequency of the stimulus",
        },
        overwrite=True,
        format="FIF",
    )
