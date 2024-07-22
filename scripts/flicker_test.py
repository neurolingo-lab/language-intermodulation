from pathlib import Path

import numpy as np
import pandas as pd
import psychopy.visual
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
        "flicker_freq",
        "duration",
        "trial_start",
        "trial_end",
    ],
    "continuous_per_trial": [
        "flicker_switches",
    ],
}
FREQUENCIES = np.arange(2, 60, step=0.2)
TRIAL_DUR = 3.0


# initialize components
window = psychopy.visual.Window(**WINDOW_CONFIG)
framerate = window.getActualFrameRate()
clock = psychopy.core.Clock()
logger = imc.ExperimentLog(loggables=LOGGABLES)
stim = imc.StatefulStim(
    window=window,
    constructors={"fullscreen": Rect},
)
constructor_kwargs = {
    "fullscreen": {
        "units": "pix",
        "size": window.size,
        "fillColor": [1, 1, 1],
        "opacity": 1.0,
    }
}
# State definitions
state = imc.FlickerStimState(
    next="flicker",
    dur=TRIAL_DUR,
    framerate=framerate,
    frequencies={"fullscreen": 5.0},
    window=window,
    stim=stim,
    logger=logger,
    clock=clock,
)


# Run flickering
window.flip()
clock.reset()
for i, freq in enumerate(FREQUENCIES):
    state.frequencies = {"fullscreen": freq}
    flipt = window.getFutureFlipTime(clock=clock)
    state.start_state(flipt, constructor_kwargs=constructor_kwargs)
    logger.log(i, "flicker_freq", freq)
    logger.log(i, "duration", 2.0)
    logger.log(i, "flicker_switches", imu.lazy_time(clock))
    logger.log(i, "trial_start", imu.lazy_time(clock))
    window.flip()
    logger.log_flip()
    while (fft := window.getFutureFlipTime(clock=clock)) < state.stimon_t + state.dur:
        state.update_state(fft)
        if len(state.log_onflip) > 0:
            logger.log(i, "flicker_switches", imu.lazy_time(clock))
        window.flip()
        logger.log_flip()
    state.end_state(window.getFutureFlipTime(clock=clock))
    logger.log(i, "trial_end", imu.lazy_time(clock))
    window.flip()
    logger.log_flip()
window.close()
logger.save(Path("../data/flicker_test_logs.pkl"))

stats = []
for trial, logs in logger.continuous.items():
    diff = pd.Series(np.diff(a=logs["flicker_switches"]))
    f_diff = 1 / (2 * diff)
    diffstats = diff.describe().to_dict()
    fstats = f_diff.describe().to_dict()
    screenstats = {k + "_interval": v for k, v in diffstats.items()}
    fscreenstats = {k + "_f": v for k, v in fstats.items()}
    screenstats["mean_f_old"] = 1 / (2 * screenstats["mean_interval"])
    finalstats = {"target_f": FREQUENCIES[trial], **screenstats, **fscreenstats}
    stats.append(finalstats)
statsdf = logger.trialsdf.join(pd.DataFrame(stats))
statsdf.to_csv(Path("../data/flicker_test_stats.csv").resolve(), index=False)
