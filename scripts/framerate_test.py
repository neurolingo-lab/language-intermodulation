from pathlib import Path

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
    "color": [-1, -1, -1],
    "colorSpace": "rgb",
    "units": "pix",
    "checkTiming": False,
}
LOGGABLES = {
    "per_state": [
        "state_number",
    ],
    "continuous_per_state": [
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


# Run and save the pure reported frame flip test
window.flip()
clock.reset()
logger.log(1, "state_number", 1)
window.callOnFlip(ptb_get_secs)
while clock.getTime() < 120:
    logger.log(1, "frame_flips", imu.lazy_time(clock))
    window.flip()
    logger.log_flip()
    window.callOnFlip(ptb_get_secs)

logger.save(Path("../data/framerate_test.pkl"))
pd.Series(flipt).to_csv(Path("../data/framerate_test_fliptimes.csv").resolve(), index=False)

# Run and save the frame rate test with the addition of a flickering whole-screen white box
# which will produce a framerate / 2 Hz flicker
screenbox = Rect(window, units="pix", size=window.size, fillColor=[1, 1, 1], opacity=1.0)
logger = imc.ExperimentLog(loggables=LOGGABLES)
flipt = []

window.flip()
clock.reset()
logger.log(1, "state_number", 1)
window.callOnFlip(ptb_get_secs)
i = 0  # Frame counter
while clock.getTime() < 120:
    logger.log(1, "frame_flips", imu.lazy_time(clock))
    if i % 2 == 0:
        screenbox.draw()
    window.flip()
    logger.log_flip()
    window.callOnFlip(ptb_get_secs)
    i += 1

logger.save(Path("../data/framerate_flicker_test.pkl"))
pd.Series(flipt).to_csv(
    Path("../data/framerate_flicker_test_fliptimes.csv").resolve(), index=False
)
