from pathlib import Path

import numpy as np
import pandas as pd
import psychopy.core
import psychopy.visual

from intermodulation.core import ExperimentController
from intermodulation.core.events import ExperimentLog
from intermodulation.states import FixationState, InterTrialState, TwoWordState
from intermodulation.stimuli import TwoWordStim

# constants
RANDOM_SEED = 42  # CHANGE IF NOT DEBUGGING!!

WINDOW_CONFIG = {
    "screen": 0,  # 0 is the primary monitor
    "fullscr": True,
    "winType": "pyglet",
    "allowStencil": False,
    "monitor": "testMonitor",
    "color": [0, 0, 0],
    "colorSpace": "rgb",
    "units": "deg",
    "checkTiming": False,
}
FLICKER_RATES = np.array([20.0, 30.0])  # Hz
WORDS = pd.read_csv("words_v1.csv")
ITI_BOUNDS = [0.2, 0.5]  # seconds
FIXATION_DURATION = 1.0  # seconds
WORD_DURATION = 1.0  # seconds
N_BLOCKS = 3  # number of blocks of stimuli to run (each block is the full word list, permuted)
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


rng = np.random.default_rng(RANDOM_SEED)
window = psychopy.visual.Window(**WINDOW_CONFIG)
framerate = window.getActualFrameRate()
clock = psychopy.core.Clock()
logger = ExperimentLog()
states = {
    "fixation": FixationState(
        next="words",
        dur=FIXATION_DURATION,
        window=window,
        clock=clock,
        framerate=framerate,
        dot_kwargs=DOT_CONFIG,
    ),
    "words": TwoWordState(
        next="ITI",
        dur=WORD_DURATION,
        window=window,
        stim=TwoWordStim(
            win=window,
            word1=WORDS.iloc[0]["word1"],
            word2=WORDS.iloc[0]["word2"],
            separation=WORD_SEP,
            fixation_dot=True,
            text_config=TEXT_CONFIG,
            dot_config=DOT_CONFIG,
        ),
        frequencies={"words": {"word1": FLICKER_RATES[0], "word2": FLICKER_RATES[1]}},
        clock=clock,
        framerate=framerate,
    ),
    "ITI": InterTrialState(
        next="fixation",
        duration_bounds=ITI_BOUNDS,
        rng=rng,
    ),
}

states["words"].wordcount = 0


def update_words():
    words = WORDS.iloc[states["words"].wordcount]
    states["words"].stim.word1 = words["word1"]
    states["words"].stim.word2 = words["word2"]
    states["words"].wordcount += 1
    return


controller = ExperimentController(
    states=states,
    window=window,
    start="fixation",
    logger=logger,
    clock=clock,
    trial_endstate="ITI",
    N_blocks=N_BLOCKS,
    K_blocktrials=len(WORDS),
    trial_calls=[update_words],
)

controller.run_experiment()
logger.save(Path("../data/word_experiment_test.csv").resolve())
