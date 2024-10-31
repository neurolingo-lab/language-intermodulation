from pathlib import Path

import numpy as np
import pandas as pd
import psychopy.logging
import psychopy.visual
from psyquartz import Clock

import intermodulation.core as core
import intermodulation.utils as spec
from intermodulation.core.controller import ExperimentController
from intermodulation.core.events import ExperimentLog
from intermodulation.states import TwoWordState
from intermodulation.stimuli import TwoWordStim

parent_path = Path(core.__file__).parents[2]

# constants
RANDOM_SEED = 42  # CHANGE IF NOT DEBUGGING!!

WINDOW_CONFIG = {
    "screen": 0,  # 0 is the primary monitor
    "fullscr": False,
    "winType": "pyglet",
    "allowStencil": False,
    "monitor": "testMonitor",
    "color": [-1, -1, -1],
    "colorSpace": "rgb",
    "units": "deg",
    "checkTiming": False,
}
FLICKER_RATES = np.array([6.25, 25])  # Hz
WORDS = pd.read_csv(parent_path / "words_v1.csv")
ITI_BOUNDS = [0.2, 0.5]  # seconds
FIXATION_DURATION = 1.0  # seconds
WORD_DURATION = 0.5  # seconds
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

clock = Clock()
psychopy.logging.setDefaultClock(clock)

rng = np.random.default_rng(RANDOM_SEED)
wordsdf = spec.assign_frequencies_to_words(WORDS, *FLICKER_RATES, rng)
window = psychopy.visual.Window(**WINDOW_CONFIG)
framerate = np.round(window.getActualFrameRate())
logger = ExperimentLog()
states = {
    "words": TwoWordState(
        next="words",
        dur=WORD_DURATION,
        window=window,
        word_list=WORDS,
        stim=TwoWordStim(
            win=window,
            word1="experiment",
            word2="start",
            separation=WORD_SEP,
            fixation_dot=True,
            text_config=TEXT_CONFIG,
            dot_config=DOT_CONFIG,
        ),
        frequencies={"words": {"word1": FLICKER_RATES[0], "word2": FLICKER_RATES[1]}},
        clock=clock,
        framerate=framerate,
        flicker_handler="frame_count",
    ),
}
state_calls = {  # What to call at different points (start, update, end) of each state
    "words": {
        "end": states["words"].update_words,
    }
}

controller = ExperimentController(
    states=states,
    window=window,
    start="words",
    logger=logger,
    clock=clock,
    trial_endstate="words",
    N_blocks=N_BLOCKS,
    K_blocktrials=10,
    state_calls=state_calls,
)
controller.add_loggable(
    "words", "start", "condition", object=states["words"].stim, attribute="word1"
)

controller.run_experiment()
logger.save(Path("../data/word_experiment_test.pkl").resolve())

window.close()
