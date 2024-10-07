from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psychopy.core
import psychopy.event
import psychopy.monitors
import psychopy.visual
from psyquartz import Clock

from intermodulation.freqtag_spec import (
    DOT_CONFIG,
    REPORT_PIX,
    REPORT_PIX_SIZE,
    TEXT_CONFIG,
    WINDOW_CONFIG,
    WORD_SEP,
)

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
FLICKER_RATES = np.array([15, 17.14286])  # Hz
TWOWORDS = pd.read_csv(parent_path / "two_word_stimuli.csv", index_col=0).sample(
    frac=1, random_state=rng
)
ONEWORDS = pd.read_csv(parent_path / "one_word_stimuli.csv", index_col=0).sample(
    frac=1, random_state=rng
)
FIXATION_DURATION = 0.5  # seconds
WORD_DURATION = 2.0  # seconds
QUERY_DURATION = 2.0  # seconds
ITI_BOUNDS = [0.5, 1.5]  # seconds
QUERY_P = 0.1  # probability of a query appearing after stimulus
N_BLOCKS_2W = 3  # number of blocks of stimuli to run (each block is the full word list, permuted)
N_BLOCKS_1W = 2  # number of blocks of stimuli to run for the one-word task
#############################################################
#         DEBUGGING PARAMETER CHANGES HERE, IF ANY!         #
#############################################################
# FLICKER_RATES = np.array([5.55555555555555, 16.666666666666])  # Hz
# WORD_DURATION = 2.0  # seconds
#############################################################

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
# framerate = 100

try:
    trigger = ParallelPortTrigger(PARALLEL_PORT, delay=2)
except RuntimeError:

    class DummyTrigger:
        def __init__(self):
            pass

        def signal(self, value: int):
            print(f"Trigger would be sent with value {value}")

    trigger = DummyTrigger()


########################################################
#                     TASK 1 BELOW                     #
########################################################

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
#                     TASK 2 BELOW                     #
########################################################

psychopy.event.globalKeys.remove("all")

# New states, controller, and logger for the one-word task
clock.reset()
worddf = spec.assign_frequencies_to_words(ONEWORDS, *FLICKER_RATES, rng)
states_1word = {
    "pause": OneWordState(
        next="fixation",
        dur=np.inf,
        window=window,
        stim=OneWordStim(
            win=window,
            word1="Time for a break!",
            text_config=TEXT_CONFIG,
            reporting_pix=REPORT_PIX,
            reporting_pix_size=REPORT_PIX_SIZE,
        ),
        word_list=pd.DataFrame(
            {
                "w1": ["Time for a break!"],
                "w2": [
                    None,
                ],
                "w1_freq": [0],
                "condition": ["pause"],
            }
        ),
        frequencies={"words": {"word1": None}},
        clock=clock,
        framerate=framerate,
        flicker_handler="frame_count",
    ),
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
            reporting_pix=REPORT_PIX,
            reporting_pix_size=REPORT_PIX_SIZE,
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
    current="pause",
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
    try:
        logger.save("final_experiment.pkl")
    except Exception as e:
        print(f"Failed to save logs: {e}")
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


window.flip()
clock.reset()
controller.toggle_pause()
controller.run_experiment()
