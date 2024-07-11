"""
General task-related utility functions.
"""

import asyncio
from copy import deepcopy
import pickle
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psychopy.constants
import psychopy.core
import psychopy.visual

LOGGABLES = {
    "per_trial": [
        "trial_number",
        "block_number",
        "block_trial",
        "trial_start",
        "fixation_on",
        "stim_on",
        "iti_start",
        "trial_end",
        "word_1",
        "word_2",
        "word_1_freq",
        "word_2_freq",
        "condition",
        "randomized",
        "trial_cond",
    ],
    "continuous_per_trial": [
        "word_1_switches",
        "word_2_switches",
        "word_1_states",
        "word_2_states",
    ],
}


@dataclass
class ExperimentLog:
    loggables: Mapping[str, list[str]] = field(default_factory=LOGGABLES.copy)

    def __post_init__(self):
        if "trial_number" not in self.loggables["per_trial"]:
            self.loggables["per_trial"].insert(0, "trial_number")
        trial_template = tuple(zip(
            self.loggables["per_trial"], np.ones(len(self.loggables["per_trial"])) * np.nan
        ))
        self.trials = defaultdict(lambda: deepcopy(dict(trial_template)))
        cont_template = tuple(zip(
            self.loggables["continuous_per_trial"],
            [list().copy() for i in range(len(self.loggables["continuous_per_trial"]))],
        ))
        self.continuous = defaultdict(lambda: deepcopy(dict(cont_template)))
        self.log_on_flip = []

    def log(self, trial_number, key, value):
        if trial_number not in self.trials:
            self.trials[trial_number]["trial_number"] = trial_number
        if trial_number not in self.continuous:
            _ = self.continuous[trial_number]
        
        if asyncio.coroutines.iscoroutine(value):
            self.log_on_flip.append(self._lazylog(trial_number, key, value))
        else:
            asyncio.run(self._lazylog(trial_number, key, value))

    def log_flip(self):
        if len(self.log_on_flip) > 0:
            asyncio.run(self._process_flip_logs())
            self.log_on_flip = []

    def save(self, fn: str):
        trial_nums = list(self.trials.keys())
        trialsdf = pd.DataFrame.from_records([self.trials[tn] for tn in sorted(trial_nums)])
        with open(fn, "wb") as fw:
            pickle.dump({"continuous": self.continuous, "trials": trialsdf}, fw)

    async def _lazylog(self, trial_number, key, value):
        if asyncio.iscoroutine(value):
            value = await value
        if key in self.loggables["per_trial"]:
            self.trials[trial_number][key] = value
        elif key in self.loggables["continuous_per_trial"]:
            self.continuous[trial_number][key].append(value)
        else:
            raise ValueError(f"Key {key} not in loggables.")

    async def _process_flip_logs(self):
        for log in self.log_on_flip:
            await log
        return


def quit_experiment(
    device_manager,
    window,
):
    """
    Function to quit the experiment on keypress of 'q' or 'escape'. Only closes the window,
    but doesn't kill the python process (for debugging and testing).

    Parameters
    ----------
    device_manager : psychopy.hardware.DeviceManager
        The device manager object that contains the ioServer object.
    window : psychopy.visual.Window
        The window object to close.
    """
    keyboard = device_manager.getDevice("defaultKeyboard")
    if keyboard.getKeys(keyList=["q", "escape"]):
        window.close()
        return True
    return False
