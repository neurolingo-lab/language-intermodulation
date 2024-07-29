import asyncio
import pickle
from collections import defaultdict
from collections.abc import Collection, Hashable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

LOGGABLES = {
    "per_state": [
        "state_number",
        "trial_number",
        "block_number",
        "block_trial",
        "state_start",
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
    "continuous_per_state": [
        "word_1_switches",
        "word_2_switches",
        "word_1_states",
        "word_2_states",
    ],
}


@dataclass
class ExperimentLog:
    # TODO: Checking whether logged items are in loggables aggressively, documentation
    loggables: Mapping[str, list[str]] = field(default_factory=LOGGABLES.copy)

    def __post_init__(self):
        if "state_number" not in self.loggables["per_state"]:
            self.loggables["per_state"].insert(0, "state_number")
        state_template = tuple(
            zip(self.loggables["per_state"], np.ones(len(self.loggables["per_state"])) * np.nan)
        )
        self.states = defaultdict(lambda: deepcopy(dict(state_template)))
        cont_template = tuple(
            zip(
                self.loggables["continuous_per_state"],
                [list().copy() for i in range(len(self.loggables["continuous_per_state"]))],
            )
        )
        self.continuous = defaultdict(lambda: deepcopy(dict(cont_template)))
        self.log_on_flip = []

    def log(self, state_number, key, value):
        if state_number not in self.states:
            self.states[state_number]["state_number"] = state_number
        if state_number not in self.continuous:
            _ = self.continuous[state_number]

        if asyncio.coroutines.iscoroutine(value):
            self.log_on_flip.append(self._lazylog(state_number, key, value))
        else:
            asyncio.run(self._lazylog(state_number, key, value))

    def log_flip(self):
        if len(self.log_on_flip) > 0:
            asyncio.run(self._process_flip_logs())
            self.log_on_flip = []

    def save(self, fn: str | Path):
        if isinstance(fn, Path):
            fn = fn.resolve()
        state_nums = list(self.states.keys())
        statesdf = pd.DataFrame.from_records([self.states[tn] for tn in sorted(state_nums)])
        with open(fn, "wb") as fw:
            pickle.dump({"continuous": dict(self.continuous), "states": statesdf}, fw)
        self.statesdf = statesdf

    async def _lazylog(self, state_number, key, value):
        if asyncio.iscoroutine(value):
            value = await value
        if key in self.loggables["per_state"]:
            self.states[state_number][key] = value
        elif key in self.loggables["continuous_per_state"]:
            self.continuous[state_number][key].append(value)
        else:
            raise ValueError(f"Key {key} not in loggables.")

    async def _process_flip_logs(self):
        for log in self.log_on_flip:
            await log
        return
