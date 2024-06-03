import numpy as np
import pandas as pd

from intermodulation import experiments, stimuli

# constants
RANDOM_SEED = 42  # CHANGE IF NOT DEBUGGING!!

# WINDOW_CONFIG = {
#     "screen": 0,  # 0 is the primary monitor
#     "fullscr": True,
#     "winType": "pyglet",
#     "allowStencil": False,
#     "monitor": "testMonitor",
#     "color": [0, 0, 0],
#     "colorSpace": "rgb",
#     "units": "deg",
#     "checkTiming": False,
# }
WINDOW_CONFIG = stimuli.DEBUG_WINDOW_CONFIG
FLICKER_RATES = np.array([17.0, 19.0])  # Hz
WORDS = pd.read_csv("words_v1.csv")[:10]
ITI_BOUNDS = [0.2, 0.5]  # seconds
FIXATION_DURATION = 1.0  # seconds
WORD_DURATION = 1.0  # seconds
N_BLOCKS = 1  # number of blocks of stimuli to run (each block is the full word list, permuted)
WORD_SEP: int = 5  # word separation in degrees
TEXT_CONFIG = {
    "font": "Arial",
    "height": 2.0,
    "wrapWidth": None,
    "ori": 0.0,
    "color": "white",
    "colorSpace": "rgb",
    "opacity": None,
    "languageStyle": "LTR",
    "depth": 0.0,
}
DOT_CONFIG = {
    "size": (0.05, 0.05),
    "vertices": "circle",
    "anchor": "center",
    "colorSpace": "rgb",
    "lineColor": "white",
    "fillColor": "white",
    "interpolate": True,
}

expspec = experiments.ExperimentSpec(
    flicker_rates=FLICKER_RATES,
    flicker_map=(0, 1),
    words=WORDS,
    iti_bounds=ITI_BOUNDS,
    fixation_t=FIXATION_DURATION,
    word_t=WORD_DURATION,
    n_blocks=N_BLOCKS,
    word_sep=WORD_SEP,
)

experiment = experiments.WordFreqTagging(
    window_config=WINDOW_CONFIG,
    expspec=expspec,
    text_config=TEXT_CONFIG,
    dot_config=DOT_CONFIG,
    random_seed=RANDOM_SEED,
)

log = experiment.run()
log.trial_states["flip"] = experiment.flip_times
log.save("word_tagging_log.pkl")
