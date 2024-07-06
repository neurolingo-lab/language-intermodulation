"""
General task-related utility functions.
"""

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import psychopy.constants
import psychopy.core
import psychopy.visual

import intermodulation.stimuli as stimuli


@dataclass
class ExperimentLog:
    clock: psychopy.core.Clock
    trials: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    trial_states: dict[int, dict[int, list[tuple[float, bool]]]] = field(
        default_factory=lambda: {0: defaultdict(list), 1: defaultdict(list)}
    )

    async def update(self, attrib, key, value, exp, key2=None):
        value = self.parse_value(value, exp)

        if key2 is None:
            getattr(self, attrib)[key].append(value)
        else:
            getattr(self, attrib)[key][key2].append(value)

    def parse_value(self, value, exp):
        twoval = False
        if isinstance(value, tuple):
            twoval = True
            secondval = value[1]
            value = value[0]
            if not isinstance(value, str):
                raise ValueError(f"Invalid value string. Must have string in first position.")
        if isinstance(value, str):
            match value.split("."):
                case ["fliptime"]:
                    return1 = exp.last_flip
                case ["state", subkey]:
                    return1 = getattr(exp.state, subkey)
                case [other]:
                    return1 = other
        else:
            return1 = value
        if twoval:
            return return1, secondval
        else:
            return return1

    def save(self, fn: str):
        with open(fn, "wb") as fw:
            pickle.dump({"trial_states": self.trial_states, "trials": self.trials}, fw)


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
