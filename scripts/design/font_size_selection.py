from pathlib import Path

import numpy as np
import pandas as pd
import psychopy.core
import psychopy.event
import psychopy.monitors
import psychopy.visual
import psychopy.visual.circle
from psyquartz import Clock

import intermodulation.core as core
from intermodulation.freqtag_spec import (
    DOT_CONFIG,
    REPORT_PIX,
    REPORT_PIX_SIZE,
    TEXT_CONFIG,
    WINDOW_CONFIG,
    WORD_SEP,
)
from intermodulation.states import (
    TwoWordState,
)
from intermodulation.stimuli import TwoWordStim

parent_path = Path(core.__file__).parents[2]

###################################
# Targets for font size selection #
###################################
MAX_SEP = 5.0  # Maximum separation of words in degrees from leftmost to rightmost edges
FUDGE_FACTOR = 0.5  # Psychopy "size" returns the size from roughly middle of first to last letters. This is to correct.
TWOWORDS = pd.read_csv(parent_path / "two_word_stimuli.csv", index_col=0)
###################################

# Set up dummy flicker rates for testing
flicker_rates = np.array([0.0, 0.0])  # Hz
seed = 42
rng = np.random.default_rng(seed)

# Use the psyquartz clock for platform stability
clock = Clock()
psychopy.logging.setDefaultClock(clock)

# Choose which monitor to use for the experiment (uncomment if not testing)
desktop = psychopy.monitors.Monitor(name="desktop", width=80.722, distance=60)
desktop.setSizePix((3440, 1440))
desktop.save()
WINDOW_CONFIG["monitor"] = "desktop"

WINDOW_CONFIG["fullscr"] = False
window = psychopy.visual.Window(**WINDOW_CONFIG)
framerate = window.getActualFrameRate()
if framerate is None:
    raise ValueError("Could not determine window framerate")
framerate = np.round(framerate)

# Setup of experiment components
TWOWORDS["w1_freq"], TWOWORDS["w2_freq"] = 0.0, 0.0
wordsdf = TWOWORDS.copy()
states_2word = {
    "words": TwoWordState(
        next="words",
        dur=1.0,
        window=window,
        stim=TwoWordStim(
            win=window,
            word1="experiment",
            word2="start",
            separation=WORD_SEP,
            fixation_dot=True,
            reporting_pix=REPORT_PIX,
            reporting_pix_size=REPORT_PIX_SIZE,
            text_config=TEXT_CONFIG,
            dot_config=DOT_CONFIG,
        ),
        word_list=wordsdf,
        frequencies={"words": {"word1": None, "word2": None}},
        clock=clock,
        framerate=framerate,
        flicker_handler="frame_count",
    ),
}

fivedeg = psychopy.visual.circle.Circle(
    win=window,
    radius=MAX_SEP / 2,
    units="deg",
    lineColor=(1, 1, 1),
    fillColor=None,
    autoDraw=True,
)


def get_max_sep_words(wordstate: TwoWordState, window: psychopy.visual.Window):
    wordstate.update_words()
    wordstate.start_state(0.0)
    window.flip()
    edgeL = (
        wordstate.stim.stim["words"]["word1"].pos[0]
        - wordstate.stim.stim["words"]["word1"].width / 2
    )
    edgeR = (
        wordstate.stim.stim["words"]["word2"].pos[0]
        + wordstate.stim.stim["words"]["word2"].width / 2
    )
    wordstate.end_state(1.0)
    return (edgeR - edgeL) + WORD_SEP


def cycle_all_words(wordstate: TwoWordState, window: psychopy.visual.Window):
    for i in range(len(wordstate.word_list)):
        wordstate.update_words()
        wordstate.start_state(0.0)
        window.flip()
        psychopy.core.wait(0.3333333333)
        wordstate.end_state(1.0)
    return


states_2word["words"].start_state(0.0)
window.flip()
states_2word["words"].end_state(1.0)
currmaxsep = 100
while (currmaxsep + FUDGE_FACTOR) >= MAX_SEP:
    innermax = 0
    for w in ["word1", "word2"]:
        states_2word["words"].stim.stim_constructor_kwargs["words"][w]["height"] -= 0.01
    for i in range(len(states_2word["words"].word_list)):
        currsep = get_max_sep_words(states_2word["words"], window)
        if currsep > innermax:
            innermax = currsep
    currmaxsep = innermax

print(
    "chosen font height is ",
    states_2word["words"].stim.stim_constructor_kwargs["words"]["word1"]["height"],
)

cycle_all_words(states_2word["words"], window)
