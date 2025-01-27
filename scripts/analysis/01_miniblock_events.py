"""
Script to add whole-miniblock events to the data, instead of only the detected trigger signals
showing the start of each stimuli within the mini-block.
"""

from pathlib import Path

import mne_bids as mnb

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
cleanpath = basepath_deriv.copy().update(processing="clean", extension=".fif")
