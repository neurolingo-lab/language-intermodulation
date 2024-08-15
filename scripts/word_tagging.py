from pathlib import Path

import numpy as np
import pandas as pd
import psychopy.core
import psychopy.visual

from intermodulation.core import ExperimentController
from intermodulation.core.events import ExperimentLog
from intermodulation.states import FixationState, InterTrialState, TwoWordState
from intermodulation.stimuli import TwoWordStim
import intermodulation.core as core

parent_path = Path(core.__file__).parents[2]

# constants
RANDOM_SEED = 42  # CHANGE IF NOT DEBUGGING!!

WINDOW_CONFIG = {
    "screen": 0,  # 0 is the primary monitor
    "fullscr": True,
    "winType": "pyglet",
    "allowStencil": False,
    "monitor": "testMonitor",
    "color": [-1, -1, -1],
    "colorSpace": "rgb",
    "units": "deg",
    "checkTiming": False,
}
FLICKER_RATES = np.array([6., 10.0])  # Hz
WORDS = pd.read_csv(parent_path / "words_v1.csv")
ITI_BOUNDS = [0.2, 0.5]  # seconds
FIXATION_DURATION = 1.0  # seconds
WORD_DURATION = 2.0  # seconds
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


rng = np.random.default_rng(RANDOM_SEED)
window = psychopy.visual.Window(**WINDOW_CONFIG)
framerate = np.round(window.getActualFrameRate())
clock = psychopy.core.Clock()
logger = ExperimentLog()
states = {
    "words": TwoWordState(
        next="words",
        dur=WORD_DURATION,
        window=window,
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
        flicker_handler="frame_count"
    ),
}

states["words"].wordcount = 0


def update_words():
    words = WORDS.iloc[states["words"].wordcount]
    states["words"].stim.word1 = words["w1"]
    states["words"].stim.word2 = words["w2"]
    if states["words"].wordcount == len(WORDS):
        states["words"].wordcount = 0
    else:
        states["words"].wordcount += 1
    return


controller = ExperimentController(
    states=states,
    window=window,
    start="words",
    logger=logger,
    clock=clock,
    trial_endstate="words",
    N_blocks=N_BLOCKS,
    K_blocktrials=10,
    trial_calls=[update_words],
)

controller.run_experiment()
logger.save(Path("../data/word_experiment_test.pkl").resolve())

window.close()