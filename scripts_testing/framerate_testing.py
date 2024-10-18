from pathlib import Path

import pandas as pd
import psychopy.visual
import psychtoolbox as ptb
from byte_triggers import ParallelPortTrigger
from psychopy.visual.rect import Rect

import intermodulation.core as imc
import intermodulation.utils as imu

# constants
TESTING_TIME = 15
WINDOW_CONFIG = {
    "screen": 1,  # 0 is the primary monitor
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
SAVEPATH = Path(__file__).parents[1] / "data"
print(SAVEPATH)

trigger = ParallelPortTrigger("/dev/parport0")
window = psychopy.visual.Window(**WINDOW_CONFIG)
framerate = window.getActualFrameRate()
clock = psychopy.core.Clock()
logger = imc.ExperimentLog(loggables=LOGGABLES)

flipt = []


def ptb_get_secs():
    flipt.append(ptb.GetSecs())


# Run and save the pure reported frame flip test
screenbox = Rect(window, units="pix", size=window.size, fillColor=[1, 1, 1], opacity=1.0)
window.flip()
clock.reset()
window.callOnFlip(ptb_get_secs)
i = 0
while clock.getTime() < TESTING_TIME:
    if i % 2 == 0:
        screenbox.draw()
    window.flip()
    window.callOnFlip(ptb_get_secs)
    i += 1

pd.Series(flipt).to_csv(SAVEPATH / "framerate_test_fliptimes_ptb.csv", index=False)


# Now test using only the intermodulation logger
logger = imc.ExperimentLog(loggables=LOGGABLES)

trigger.signal(1)
window.flip()
clock.reset()
logger.log(1, "state_number", 1)
i = 0  # Frame counter
while clock.getTime() < TESTING_TIME:
    logger.log(1, "frame_flips", imu.lazy_time(clock))
    if i % 2 == 0:
        screenbox.draw()
    window.flip()
    logger.log_flip()
    i += 1

logger.save(SAVEPATH / "framerate_flicker_test_logger.pkl")


# Run a combination of the two logging methods
logger = imc.ExperimentLog(loggables=LOGGABLES)
flipt = []

trigger.signal(1)
window.flip()
clock.reset()
logger.log(1, "state_number", 1)
window.callOnFlip(ptb_get_secs)
i = 0  # Frame counter
while clock.getTime() < TESTING_TIME:
    logger.log(1, "frame_flips", imu.lazy_time(clock))
    if i % 2 == 0:
        screenbox.draw()
    window.flip()
    logger.log_flip()
    window.callOnFlip(ptb_get_secs)
    i += 1

logger.save(SAVEPATH / "framerate_flicker_test_bothlogs.pkl")
pd.Series(flipt).to_csv(SAVEPATH / "framerate_flicker_test_fliptimes_bothlogs.csv", index=False)
