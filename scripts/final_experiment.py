from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psychopy.core
import psychopy.event
import psychopy.visual
from psyquartz import Clock

try:
    from byte_triggers import ParallelPortTrigger
except ImportError:
    pass

import intermodulation.core as core
import intermodulation.freqtag_spec as spec
from intermodulation.core.events import ExperimentLog
from intermodulation.states import FixationState, InterTrialState, QueryState, TwoWordState
from intermodulation.stimuli import FixationStim, QueryStim, TwoWordStim

parent_path = Path(core.__file__).parents[2]

# constants
RANDOM_SEED = 42  # CHANGE IF NOT DEBUGGING!! SET TO NONE FOR RANDOM SEED

# EXPERIMENT PARAMETERS
LOGPATH = parent_path / "logs"
PARALLEL_PORT = "/dev/parport0"  # Set to None if no parallel port
FLICKER_RATES = np.array([6.25, 25])  # Hz
WORDS = pd.read_csv(parent_path / "words_v1.csv").iloc[:50]
FIXATION_DURATION = 0.2  # seconds
WORD_DURATION = 0.5  # seconds
QUERY_DURATION = 0.5  # seconds
ITI_BOUNDS = [0.05, 0.2]  # seconds
QUERY_P = 0.9  # probability of a query appearing after stimulus
N_BLOCKS = 1  # number of blocks of stimuli to run (each block is the full word list, permuted)
WORD_SEP: int = 5  # word separation in degrees

# Detailed display parameters
DISPLAY_DISTANCE = 120  # cm
DISPLAY_WIDTH = 55  # cm
DISPLAY_HEIGHT = 30.5
FOVEAL_ANGLE = 5.0  # degrees

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

# Use the psyquartz clock for platform stability
clock = Clock()
psychopy.logging.setDefaultClock(clock)


# Setup of RNG and display distance/width
rng = np.random.default_rng(RANDOM_SEED)
# propixx = Monitor(name="propixx", width=DISPLAY_WIDTH, distance=DISPLAY_DISTANCE)
# WINDOW_CONFIG["monitor"] = "propixx"

window = psychopy.visual.Window(**WINDOW_CONFIG)
framerate = np.round(window.getActualFrameRate())
if framerate is None:
    raise ValueError("Could not determine window framerate")

## FOR DEBUGGING ONLY!!! ##
framerate = (
    100  # Change to whatever you *know* your monitor to refresh at. Avoids measurement errors.
)
###########################

logger = ExperimentLog(loggables=spec.LOGGABLES)

# Setup of experiment components
wordsdf = spec.assign_frequencies_to_words(WORDS, *FLICKER_RATES, rng)
states = {
    "intertrial": InterTrialState(
        next="fixation",
        duration_bounds=ITI_BOUNDS,
        rng=rng,
    ),
    "fixation": FixationState(
        next="words",
        dur=FIXATION_DURATION,
        stim=FixationStim(win=window, dot_config=DOT_CONFIG),
        window=window,
        clock=clock,
        framerate=framerate,
    ),
    "query": QueryState(
        next="intertrial",
        dur=QUERY_DURATION,
        stim=QueryStim(win=window, rng=rng, query_config=TEXT_CONFIG),
        window=window,
        clock=clock,
        framerate=framerate,
        rng=rng,
    ),
    "words": TwoWordState(
        next=["query", "fixation"],
        transition=lambda: rng.choice([0, 1], p=[QUERY_P, 1 - QUERY_P]),
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
        word_list=wordsdf,
        frequencies={"words": {"word1": None, "word2": None}},
        clock=clock,
        framerate=framerate,
        flicker_handler="frame_count",
    ),
}

try:
    trigger = ParallelPortTrigger(PARALLEL_PORT, delay=2)
except RuntimeError:
    trigger = None
controller = core.ExperimentController(
    states=states,
    window=window,
    start="fixation",
    logger=logger,
    clock=clock,
    trial_endstate="intertrial",
    N_blocks=N_BLOCKS,
    K_blocktrials=len(WORDS),
)

controller.state_calls = {
    "words": {
        "end": [
            (
                states["words"].update_words,
                (states["query"],),
            )
        ]
    },
}

spec.add_logging_to_controller(controller, states, "query", "words")
spec.add_triggers_to_controller(
    controller, trigger, FLICKER_RATES, states, "intertrial", "fixation", "query"
)


# Set up CTRL + C handling for graceful exit with logs
def save_logs_quit():
    logger.save("final_experiment.pkl")
    window.close()
    return


psychopy.event.globalKeys.add(
    key="q",
    modifiers=["ctrl"],
    func=save_logs_quit,
)
psychopy.event.globalKeys.add(
    key="p",
    modifiers=["ctrl"],
    func=controller.toggle_pause,
)
controller.state_calls["all"] = {"end": []}
controller.run_experiment()

# Save logs
date = datetime.now().isoformat(timespec="minutes")
logger.save(LOGPATH / f"{date}_experiment.pkl")
