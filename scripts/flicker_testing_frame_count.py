from pathlib import Path

import numpy as np
import pandas as pd
import psychopy.visual
from psychopy.visual.rect import Rect
from byte_triggers import ParallelPortTrigger

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
    "checkTiming": True,
}
LOGGABLES = {
    "per_state": [
        "state_number",
        "flicker_freq",
        "duration",
        "state_start",
        "state_end",
    ],
    "continuous_per_state": [
        "flicker_switches",
    ],
}
TRIAL_DUR = 2.0


# initialize components
window = psychopy.visual.Window(**WINDOW_CONFIG)
framerate = window.getActualFrameRate()
print(f"Exact framerate of {framerate} Hz")
print(f"Rounded to {np.round(framerate).astype(int)} for choosing test freqs")
f_vals = np.linspace(4, 120, 1000)
nearest = imu.get_nearest_f(f_vals, np.round(framerate).astype(int))
FREQUENCIES = np.unique(nearest)
FREQUENCIES = FREQUENCIES[np.isfinite(FREQUENCIES)]
print(f"Total of {len(FREQUENCIES)} unique possible frequencies between 4 and 120 Hz")
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
    clock=clock,
    flicker_handler="frame_count",
    stim_constructor_kwargs=constructor_kwargs,
)


# Run flickering
trigger = ParallelPortTrigger("/dev/parport0")
window.flip()
clock.reset()
for i, freq in enumerate(FREQUENCIES):
    state.frequencies = {"fullscreen": freq}
    flipt = window.getFutureFlipTime(clock=clock)
    state.start_state(flipt)
    logger.log(i, "flicker_freq", freq)
    logger.log(i, "duration", 2.0)
    logger.log(i, "flicker_switches", imu.lazy_time(clock))
    logger.log(i, "state_start", imu.lazy_time(clock))
    trigger.signal(i + 1)
    window.flip()
    logger.log_flip()
    while (fft := window.getFutureFlipTime(clock=clock)) < state.stimon_t + state.dur:
        state.update_state(fft)
        if len(state.log_onflip) > 0:
            logger.log(i, "flicker_switches", imu.lazy_time(clock))
        window.flip()
        logger.log_flip()
    state.end_state(window.getFutureFlipTime(clock=clock))
    logger.log(i, "state_end", imu.lazy_time(clock))
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
statsdf = logger.statesdf.join(pd.DataFrame(stats))
statsdf.to_csv(Path("../data/flicker_test_stats.csv").resolve(), index=False)
