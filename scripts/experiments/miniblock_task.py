from datetime import datetime

import numpy as np
import pandas as pd
import psychopy.logging as psylog
import psychopy.visual as psyv
import psyquartz as pq
import psystate.controller as pc
import psystate.events as pe
import psystate.states as ps
from mnemonic import Mnemonic
from psychopy.gui import DlgFromDict

import intermodulation.freqtag_spec as spec
import intermodulation.states as ims
import intermodulation.stimuli as imst

# Get subject ID and session number
gen = Mnemonic("english")
default_id = gen.generate(128).replace(" ", "")
subinfo = {
    "subject": default_id,
    "session": 1,
    "date": datetime.now().strftime("%Y-%m-%d_%H-%M"),
    "seed": np.random.randint(0, 255),
}
dialog = DlgFromDict(subinfo, fixed=["date"], title="Subject Information")
dialog.show()
if dialog.OK:
    pass
else:
    exit()

rng = np.random.default_rng(subinfo["seed"])
# Prepare word stimuli
twowords = pd.read_csv(spec.TWOWORDPATH)
n_mini = int(len(twowords) / spec.MINIBLOCK_LEN)
freqidxs = np.repeat(rng.choice(2, n_mini), spec.MINIBLOCK_LEN)

onewords = pd.read_csv(spec.ONEWORDPATH)

clock = pq.Clock()
psylog.setDefaultClock(clock)
window = psyv.Window(
    fullscr=False,
    monitor="testMonitor",
    units="deg",
    color=[-1, -1, -1],
    winType="pyglet",
    checkTiming=False,
)
framerate = window.getActualFrameRate()
if framerate is None:
    framerate = 100.0
else:
    framerate = np.round(framerate)

wordstim = imst.TwoWordStim(window, "test1", "test2", separation=0.3)
fixstim = imst.FixationStim(window)


word_dur = 2.0
wordframes = int(np.round(word_dur / (1 / framerate)))


def update_words(state: ims.TwoWordState):
    if state.frame_num % wordframes == 0:
        state.update_words()


states = {
    "words": ims.TwoWordState(
        next="fixation",
        dur=20.0,
        window=window,
        framerate=framerate,
        stim=wordstim,
        clock=clock,
        frequencies={"w1": 12.5, "w2": 25, "fixdot": None},
        word_list=wordlist,
        loggables=pe.Loggables(
            start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
            end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
        ),
        log_updates=True,
        strict_freqs="allow",
    ),
    "fixation": ps.StimulusState(
        next="words",
        dur=2.0,
        stim=fixstim,
        window=window,
        clock=clock,
        loggables=pe.Loggables(
            start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
            end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
        ),
    ),
}
states["words"].start_calls.append(
    states["words"].update_words,
)
states["words"].update_calls.insert(0, (update_words, (states["words"],)))

controller = pc.ExperimentController(
    states=states,
    window=window,
    start="fixation",
    logger=pe.ExperimentLog(clock),
    clock=clock,
    trial_endstate="words",
    N_blocks=1,
    K_blocktrials=3,
)

clock.reset()
controller.run_experiment()
print(controller.logger.statesdf)
controller.logger.contdf.to_csv("testcont.csv")
print(controller.logger.contdf)
controller.logger.statesdf.to_csv("teststates.csv")
window.close()
