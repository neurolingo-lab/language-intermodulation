from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psychopy.core
import psychopy.event
import psychopy.monitors
import psychopy.visual
from psyquartz import Clock

try:
    from byte_triggers import ParallelPortTrigger
except ImportError:
    pass

import intermodulation.core as core
import intermodulation.freqtag_spec as spec
from intermodulation.core.events import ExperimentLog
from intermodulation.states import (
    FixationState,
    InterTrialState,
    OneWordState,
    QueryState,
    TwoWordState,
)
from intermodulation.stimuli import FixationStim, OneWordStim, QueryStim, TwoWordStim

parent_path = Path(core.__file__).parents[2]

# constants
RANDOM_SEED = 42  # CHANGE IF NOT DEBUGGING!! SET TO NONE FOR RANDOM SEED
rng = np.random.default_rng(RANDOM_SEED)

# EXPERIMENT PARAMETERS
LOGPATH = parent_path / "logs"
PARALLEL_PORT = "/dev/parport0"  # Set to None if no parallel port
FLICKER_RATES = np.array([0.5, 0.25])  # Hz
TWOWORDS = pd.read_csv(parent_path / "two_word_stimuli.csv", index_col=0).sample(
    frac=1, random_state=rng
)
ONEWORDS = pd.read_csv(parent_path / "one_word_stimuli.csv", index_col=0).sample(
    frac=1, random_state=rng
)
FIXATION_DURATION = 0.5  # seconds
WORD_DURATION = 2.0  # seconds
QUERY_DURATION = 2.0  # seconds
ITI_BOUNDS = [0.05, 0.2]  # seconds
QUERY_P = 0.1  # probability of a query appearing after stimulus
N_BLOCKS_2W = 1  # number of blocks of stimuli to run (each block is the full word list, permuted)
N_BLOCKS_1W = 1  # number of blocks of stimuli to run for the one-word task
WORD_SEP: int = 5  # word separation in degrees

# Detailed display parameters
DISPLAY_RES = (1280, 720)
DISPLAY_DISTANCE = 120  # cm
DISPLAY_WIDTH = 36.666666  # cm
DISPLAY_HEIGHT = 20.333333333333333333  # cm
FOVEAL_ANGLE = 5.0  # degrees

REPORT_PIX = True
REPORT_PIX_SIZE = 50

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
    "font": "Ubuntu mono",
    "height": 0.95,
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


# Make the log folder if it doesn't exist
LOGPATH.mkdir(exist_ok=True)

# Choose which monitor to use for the experiment (uncomment if not testing)
desktop = psychopy.monitors.Monitor(name="desktop", width=80.722, distance=60)
desktop.setSizePix((3440, 1440))
desktop.save()
WINDOW_CONFIG["monitor"] = "desktop"

# propixx = psychopy.monitors.Monitor(name="propixx", width=DISPLAY_WIDTH, distance=DISPLAY_DISTANCE)
# propixx.setSizePix(DISPLAY_RES)
# propixx.save()
# WINDOW_CONFIG["monitor"] = "propixx"


window = psychopy.visual.Window(**WINDOW_CONFIG)
framerate = window.getActualFrameRate()
if framerate is None:
    raise ValueError("Could not determine window framerate")
framerate = np.round(framerate)

try:
    trigger = ParallelPortTrigger(PARALLEL_PORT, delay=2)
except RuntimeError:

    class DummyTrigger:
        def __init__(self):
            pass

        def signal(self, value: int):
            print(f"Trigger would be sent with value {value}")

    trigger = DummyTrigger()

## FOR DEBUGGING ONLY!!! ##
framerate = (
    100  # Change to whatever you *know* your monitor to refresh at. Avoids measurement errors.
)
###########################

logger = ExperimentLog(loggables=spec.LOGGABLES)

# Setup of experiment components
wordsdf = spec.assign_frequencies_to_words(TWOWORDS, *FLICKER_RATES, rng)
states_2word = {
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
        next=["query", "intertrial"],
        transition=lambda: rng.choice([0, 1], p=[QUERY_P, 1 - QUERY_P]),
        dur=WORD_DURATION,
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

controller = core.ExperimentController(
    states=states_2word,
    window=window,
    start="fixation",
    logger=logger,
    clock=clock,
    trial_endstate="intertrial",
    N_blocks=N_BLOCKS_2W,
    K_blocktrials=len(TWOWORDS),
)

controller.state_calls = {
    "words": {
        "end": [
            (
                states_2word["words"].update_words,
                (states_2word["query"],),
            )
        ]
    },
}

spec.add_logging_to_controller(controller, states_2word, "query", twoword="words")
spec.add_triggers_to_controller(
    controller,
    trigger,
    FLICKER_RATES,
    states_2word,
    "intertrial",
    "fixation",
    "query",
    twoword="words",
)


# Set up CTRL + C handling for graceful exit with logs
def save_logs_quit():
    logger.save("final_experiment.pkl")
    controller.quit()
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
logger.save(LOGPATH / f"{date}_2word_experiment.pkl")

########################################################
#         TASK 1 ABOVE THIS LINE, TASK 2 BELOW         #
########################################################

psychopy.event.globalKeys.remove("all")

# Set up the one-word task while we display a break message
pause_text = psychopy.visual.TextStim(
    window,
    text="Time for a break!",
    **TEXT_CONFIG,
)
pause_text.setAutoDraw(True)
window.flip()

# New states, controller, and logger for the one-word task
clock.reset()
worddf = spec.assign_frequencies_to_words(ONEWORDS, *FLICKER_RATES, rng)
states_1word = {
    "intertrial": InterTrialState(
        next="fixation",
        duration_bounds=ITI_BOUNDS,
        rng=rng,
    ),
    "fixation": FixationState(
        next="word",
        dur=FIXATION_DURATION,
        stim=FixationStim(win=window, dot_config=DOT_CONFIG),
        window=window,
        clock=clock,
        framerate=framerate,
    ),
    "word": OneWordState(
        next="intertrial",
        dur=WORD_DURATION,
        window=window,
        stim=OneWordStim(
            win=window,
            word1="experiment",
            text_config=TEXT_CONFIG,
        ),
        word_list=worddf,
        frequencies={"words": {"word1": None}},
        clock=clock,
        framerate=framerate,
        flicker_handler="frame_count",
    ),
}

logger = ExperimentLog(loggables=spec.LOGGABLES)

controller = core.ExperimentController(
    states=states_1word,
    window=window,
    start="fixation",
    logger=logger,
    clock=clock,
    trial_endstate="intertrial",
    N_blocks=N_BLOCKS_1W,
    K_blocktrials=len(ONEWORDS),
)

controller.state_calls = {
    "word": {
        "end": [states_1word["word"].update_word],
    },
}

spec.add_logging_to_controller(controller, states_1word, oneword="word")
spec.add_triggers_to_controller(
    controller,
    trigger,
    FLICKER_RATES,
    states_1word,
    "intertrial",
    "fixation",
    oneword="word",
)


def save_logs_quit():
    logger.save("final_experiment.pkl")
    controller.quit()
    window.close()
    exit()
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

# # Wait for input and run the one-word task
# while True:
#     keys = psychopy.event.getKeys(["enter"])
#     if len(keys) > 0:
#         break

pause_text.setAutoDraw(False)
window.flip()
clock.reset()
controller.run_experiment()
