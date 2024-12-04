from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psychopy.core
import psychopy.event
import psychopy.monitors
import psychopy.visual
from psyquartz import Clock

import intermodulation.core.controller
import intermodulation.utils
from intermodulation.freqtag_spec import (
    TRIGGERS,
    WINDOW_CONFIG,
)
from intermodulation.utils import generate_1w_states, generate_discrete_states

try:
    from byte_triggers import ParallelPortTrigger
except ImportError:
    pass

import intermodulation.core as core
import intermodulation.freqtag_spec as spec
from intermodulation.core.events import ExperimentLog

parent_path = Path(core.__file__).parents[3]

#############################################################
#                   EXPERIMENT_PARAMETERS                   #
#############################################################
# RANDOM_SEED = 42  # CHANGE IF NOT DEBUGGING!! SET TO NONE FOR RANDOM SEED
RANDOM_SEED = None
rng = np.random.default_rng(RANDOM_SEED)

# PC parameters: Paths and hardware info
LOGPATH = parent_path / "logs"
PARALLEL_PORT = "/dev/parport0"  # Set to None if no parallel port

# Stimulus parameters
FLICKER_RATES = np.array([17.14286, 20.0])  # Hz
TWOWORDS = pd.read_csv(parent_path / "two_word_stimuli_nina.csv", index_col=0).sample(
    frac=1, random_state=rng
)
ONEWORDS = pd.read_csv(parent_path / "one_word_stimuli.csv", index_col=0).sample(
    frac=1, random_state=rng
)

# Task parameters
PAUSE_KEY = "4"
TRIGGER_DUR = 4  # milliseconds
FIXATION_DURATION = 0.5  # seconds
WORD_DURATION = 2.0  # seconds
QUERY_DURATION = 2.0  # seconds
ITI_BOUNDS = [0.5, 1.5]  # seconds
QUERY_P = 0.5  # probability of a query appearing after stimulus
N_BLOCKS_2W = 3  # number of blocks of stimuli to run (each block is the full word list, permuted)
N_BLOCKS_1W = 2  # number of blocks of stimuli to run for the one-word task
FORCE_FR = None
#############################################################
#         DEBUGGING PARAMETER CHANGES HERE, IF ANY!         #
#############################################################
# FLICKER_RATES = np.array([5.55555555555555, 16.666666666666])  # Hz
# WORD_DURATION = 2.0  # seconds
# TWOWORDS = TWOWORDS.head(15)
# ONEWORDS = ONEWORDS.head(15)
# QUERY_P = 1.0
# N_BLOCKS_2W = 1  # number of blocks of stimuli to run (each block is the full word list, permuted)
# N_BLOCKS_1W = 1  # number of blocks of stimuli to run for the one-word task
# FORCE_FR = 100  # Force the frame rate to 100 for testing
#############################################################

# Use the psyquartz clock for platform stability
clock = Clock()
psychopy.logging.setDefaultClock(clock)


# Make the log folder if it doesn't exist
LOGPATH.mkdir(exist_ok=True)

# Choose which monitor to use for the experiment (uncomment if not testing)
# desktop = psychopy.monitors.Monitor(name="desktop", width=80.722, distance=60)
# desktop.setSizePix((3440, 1440))
# desktop.save()
# WINDOW_CONFIG["monitor"] = "desktop"

propixx = psychopy.monitors.Monitor(
    name="propixx", width=spec.DISPLAY_WIDTH, distance=spec.DISPLAY_DISTANCE
)
propixx.setSizePix(spec.DISPLAY_RES)
propixx.save()
WINDOW_CONFIG["monitor"] = "propixx"


# Create the window and check the frame rate. Raise an error if the frame rate is not detected.
window = psychopy.visual.Window(**WINDOW_CONFIG)
if FORCE_FR is not None:
    framerate = FORCE_FR
else:
    framerate = window.getActualFrameRate()
    framerate = np.round(framerate)

# Create a trigger object for sending triggers to the parallel port, and if there's no working port
# create a dummy trigger object that prints the value that would be sent.
try:
    trigger = ParallelPortTrigger(PARALLEL_PORT, delay=TRIGGER_DUR)
except RuntimeError:

    class DummyTrigger:
        def __init__(self):
            pass

        def signal(self, value: int):
            print(f"Trigger would be sent with value {value}")

    trigger = DummyTrigger()


########################################################
#                        TASK 1                        #
########################################################

logger = ExperimentLog(loggables=spec.LOGGABLES)

# Setup of experiment components: Final stimulus df with randomly assigned flicker rates,
# states for the 2-word task, and the controller
wordsdf = intermodulation.utils.assign_frequencies_to_words(TWOWORDS, *FLICKER_RATES, rng)
states_2word = generate_discrete_states(
    rng,
    FIXATION_DURATION,
    WORD_DURATION,
    QUERY_DURATION,
    ITI_BOUNDS,
    QUERY_P,
    clock,
    window,
    framerate,
    wordsdf,
)
controller = intermodulation.core.controller.ExperimentController(
    states=states_2word,
    window=window,
    start="fixation",
    logger=logger,
    clock=clock,
    trial_endstate="intertrial",
    N_blocks=N_BLOCKS_2W,
    K_blocktrials=len(TWOWORDS),
)
controller.state_calls = {  # Make sure that at the end of each word stimulus we update the words
    "words": {
        "end": [
            (
                states_2word["words"].update_words,
                (states_2word["query"],),
            )
        ]
    },
}

# Adding logging of various variables to the controller, and trigger outputs when needed
intermodulation.utils.add_logging_to_controller(controller, states_2word, "query", twoword="words")
intermodulation.utils.add_triggers_to_controller(
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
psychopy.event.globalKeys.add(
    key=PAUSE_KEY,
    func=controller.toggle_pause,
)
controller.state_calls["all"] = {"end": []}

expltext = psychopy.visual.TextStim(
    window,
    text="This experiment will begin with a dot on the screen.\n\n Stare at the dot when you see it, "
    "and continue to stare where the dot was when words appear.\n\nSometimes these words will be "
    "followed by a question.\n\nIf the question word was present in the last set of word(s) you just saw, "
    "press yes, otherwise press no.",
    anchorHoriz="center",
    alignText="center",
    pos=(0, 0),
    color="white",
    height=1.0,
)
expltext.draw()
window.flip()

psychopy.event.waitKeys(keyList=["1"])
del expltext


controller.run_experiment()

# Save logs
date = datetime.now().strftime("%Y-%m-%d_%H-%M")
logger.save(LOGPATH / f"{date}_2word_experiment.pkl")

# ########################################################
# #                        TASK 2                        #
# ########################################################

# Reset our global keypress events so we can reassign them to the new controller
psychopy.event.globalKeys.remove("all")

# New words, states, controller, and logger for the one-word task
clock.reset()
worddf = intermodulation.utils.assign_frequencies_to_words(ONEWORDS, *FLICKER_RATES, rng)
states_1word = generate_1w_states(
    rng, FIXATION_DURATION, WORD_DURATION, ITI_BOUNDS, clock, window, framerate, worddf
)
logger = ExperimentLog(loggables=spec.LOGGABLES)
controller = intermodulation.core.controller.ExperimentController(
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
# Triggers and logging for the one-word task
intermodulation.utils.add_logging_to_controller(controller, states_1word, oneword="word")
intermodulation.utils.add_triggers_to_controller(
    controller,
    trigger,
    FLICKER_RATES,
    states_1word,
    "intertrial",
    "fixation",
    oneword="word",
)


# Set up CTRL + C handling for graceful exit with logs
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
psychopy.event.globalKeys.add(
    key=PAUSE_KEY,
    func=controller.toggle_pause,
)

window.flip()
clock.reset()
controller._resume = "fixation"
controller.toggle_pause()
controller.run_experiment()

# Save logs
date = datetime.now().strftime("%Y-%m-%d_%H-%M")
logger.save(LOGPATH / f"{date}_1word_experiment.pkl")

########################################################
#                        TASK 2.5                      #
########################################################

# Reset our global keypress events so we can reassign them to the new controller
psychopy.event.globalKeys.remove("all")

# New words, states, controller, and logger for the one-word task
clock.reset()
worddf = intermodulation.utils.assign_frequencies_to_words(ONEWORDS, *FLICKER_RATES, rng)
states_1word = generate_1w_states(
    rng, FIXATION_DURATION, WORD_DURATION, ITI_BOUNDS, clock, window, framerate, worddf
)
intermodulation.utils.add_masked_1w_states(states_1word, worddf)

logger = ExperimentLog(loggables=spec.LOGGABLES)
controller = intermodulation.core.controller.ExperimentController(
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
    "mask": {
        "end": [states_1word["mask"].update_word],
    },
    "word": {
        "end": [states_1word["word"].update_word],
    },
}

# Triggers and logging for the one-word task
intermodulation.utils.add_logging_to_controller(controller, states_1word, oneword="word")
intermodulation.utils.add_triggers_to_controller(
    controller,
    trigger,
    FLICKER_RATES,
    states_1word,
    "intertrial",
    "fixation",
    oneword="word",
)
controller.state_calls["mask"]["start"].append((trigger.signal, TRIGGERS.MASK))
controller.state_calls["mask"]["end"].append((trigger.signal, TRIGGERS.STATEEND))


# Set up CTRL + C handling for graceful exit with logs
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
psychopy.event.globalKeys.add(
    key=PAUSE_KEY,
    func=controller.toggle_pause,
)

window.flip()
clock.reset()
controller._resume = "fixation"
controller.toggle_pause()
controller.run_experiment()

# Save logs
date = datetime.now().strftime("%Y-%m-%d_%H-%M")
logger.save(LOGPATH / f"{date}_1word_experiment.pkl")
