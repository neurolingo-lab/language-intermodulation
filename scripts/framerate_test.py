from pathlib import Path

import numpy as np
import pandas as pd
import psychopy.visual
import psychtoolbox as ptb
from psychopy.visual.rect import Rect

import intermodulation.core as imc
import intermodulation.utils as imu

# constants
WINDOW_CONFIG = {
    "screen": 0,  # 0 is the primary monitor
    "fullscr": True,
    "winType": "pyglet",
    "allowStencil": False,
    "monitor": "testMonitor",
    "color": [0, 0, 0],
    "colorSpace": "rgb",
    "units": "pix",
    "checkTiming": False,
}
LOGGABLES = {
    "per_trial": [
        "trial_number",
    ],
    "continuous_per_trial": [
        "frame_flips",
    ],
}

window = psychopy.visual.Window(**WINDOW_CONFIG)
framerate = window.getActualFrameRate()
clock = psychopy.core.Clock()
logger = imc.ExperimentLog(loggables=LOGGABLES)

flipt = []


def ptb_get_secs():
    flipt.append(ptb.GetSecs())


window.flip()
clock.reset()
logger.log(1, "trial_number", 1)
window.callOnFlip(ptb_get_secs)
while clock.getTime() < 120:
    logger.log(1, "frame_flips", imu.lazy_time(clock))
    window.flip()
    logger.log_flip()
    window.callOnFlip(ptb_get_secs)

logger.save(Path("../data/framerate_test.pkl"))
pd.Series(flipt).to_csv(Path("../data/framerate_test_fliptimes.csv").resolve(), index=False)
