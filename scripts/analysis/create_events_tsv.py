from argparse import ArgumentParser

import mne
import mne_bids
import numpy as np
import pandas as pd

from intermodulation.freqtag_spec import LUT_TRIGGERS, TRIGGERS, VALID_TRANS, nested_deepkeys

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bids_root", type=str, required=True)
    parser.add_argument("--subject_id", type=str, required=True)
    parser.add_argument("--session_id", type=str, required=True)
    parser.add_argument("--task", type=str, default="syntaxIM")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stim_channel", type=str, default="STI102")
    parser.add_argument("--tmpfix", action="store_true")
    args = parser.parse_args()

    bids_path = mne_bids.BIDSPath(
        subject=args.subject_id,
        session=args.session_id,
        task=args.task,
        root=args.bids_root,
        datatype="meg",
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
    offset = raw.first_samp / raw.info["sfreq"]
    for row in goodevs.itertuples():
        if row.label2 in ("STATEEND", "TRIALEND", "BLOCKEND"):
            continue
        if goodevs.at[row.Index + 1, "label2"] in ("STATEEND", "TRIALEND", "BLOCKEND"):
            end = goodevs.at[row.Index + 1, "sample"] / raw.info["sfreq"] + offset
        else:
            if args.tmpfix and row.label2.split("/")[0] == "ONEWORD":
                end = goodevs.at[row.Index + 1, "sample"] / raw.info["sfreq"] + offset
            else:
                raise ValueError(
                    f"Expected end of trial at index {row.Index + 1}, found "
                    f"{goodevs.at[row.Index + 1, 'label2']}"
                )
        onset = row.sample / raw.info["sfreq"] + offset
        records.append(
            {
                "onset": onset,
                "duration": end - onset,
                "trial_type": row.label2,
                "value": row.v2,
                "sample": row.sample,
                "channel": args.stim_channel,
            }
        )
    bidsevs = pd.DataFrame.from_records(records)
    bidevpath = bids_path.copy().update(suffix="events", extension=".tsv")
    if not len(bidevpath.match()) == 0:
        oldev = pd.read_csv(bidevpath.fpath, sep="\t")
        catevs = pd.concat([oldev, bidsevs], axis=0).reset_index(drop=True)
        if catevs.duplicated().any():
            if not args.overwrite:
                raise ValueError("Duplicate events found after concatenation!")
            else:
                catevs.drop_duplicates(inplace=True)
        catevs.sort_values("onset", inplace=True)

    bidsevs.to_csv(bidevpath.fpath, sep="\t", index=False)
