"""
Script to add whole-miniblock events to the data, instead of only the detected trigger signals
showing the start of each stimuli within the mini-block.
"""

from pathlib import Path

import mne
import mne_bids as mnb

from intermodulation.analysis import miniblock_events

SUBJECT = "PilotC"
SESSION = "01"
TASK = "syntaxIM"

bids_root = Path("/Volumes/BerkStorage/datasets/syntax_im/syntax_dataset")
deriv_root = bids_root / "derivatives/mne-bids-pipeline/"

basepath_raw = mnb.BIDSPath(
    subject=SUBJECT,
    session=SESSION,
    task=TASK,
    root=bids_root,
    datatype="meg",
    suffix="meg",
)
basepath_deriv = basepath_raw.copy().update(root=deriv_root, suffix="raw", check=False)

# We will add these events to the raw data, SSS processed data, and "clean" data
rawpath = basepath_raw.copy().update(split="01", extension=".fif")
ssspath = basepath_deriv.copy().update(processing="sss", split="01", extension=".fif")
filtpath = basepath_deriv.copy().update(processing="filt", extension=".fif")
cleanpath = basepath_deriv.copy().update(processing="clean", extension=".fif")

for rawpath in [
    filtpath,
]:
    if rawpath.root == deriv_root:
        deriv = True
        raw = mne.io.read_raw_fif(rawpath.fpath, preload=True)
    else:
        deriv = False
        raw = mnb.read_raw_bids(rawpath)
    miniblock_events(raw)
    raw.plot()
    input("Press Enter if the new events look good, Ctrl+C to stop")
    if deriv:
        raw.save(rawpath.copy().update(split=None).fpath, split_naming="bids", overwrite=True)
    else:
        pass
        # mnb.write_raw_bids(raw, rawpath, overwrite=True)