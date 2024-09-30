import asyncio
import pickle
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LOGGABLES = {
    "per_state": [
        "state_number",
        "state",
        "next_state",
        "state_start",
        "target_end",
        "state_end",
        "trial_number",
        "block_number",
        "block_trial",
        "trial_end",
        "block_end",
        "condition",
    ],
    "continuous_per_state": [
        "stim1",
        "stim2",
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
        statesdf = self.statesdf()
        contdf = self.contdf()
        with open(fn, "wb") as fw:
            pickle.dump({"continuous": contdf, "states": statesdf}, fw)

    def statesdf(self):
        state_nums = list(self.states.keys())
        statesdf = pd.DataFrame.from_records([self.states[sn] for sn in sorted(state_nums)])
        return statesdf.convert_dtypes()

    def contdf(self):
        state_nums = list(self.continuous.keys())
        maxkeylen = max(map(len, [k for sn in state_nums for k in self.continuous[sn].keys()]))
        data = {
            "state_number": np.empty(0, dtype=int),
            "value": np.empty(0),
            **{f"key{n}": [] for n in range(maxkeylen)},
        }
        for sn in state_nums:
            for key, values in self.continuous[sn].items():
                data["state_number"] = np.append(data["state_number"], np.ones(len(values)) * sn)
                data["value"] = np.append(data["value"], values)
                if isinstance(key, str):
                    data["key0"] = np.append(data["key0"], np.ones(len(values), dtype=int) * key)
                else:
                    for i, subk in enumerate(key):
                        data[f"key{i}"].extend([subk] * len(values))

        contdf = pd.DataFrame.from_dict(data)
        return contdf.convert_dtypes()

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
