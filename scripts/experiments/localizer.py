from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import psychopy.event as psyev
import psychopy.logging as psylog
import psychopy.visual as psyv
import psyquartz as pq
import psystate.controller as pc
import psystate.events as pe
from byte_triggers import MockTrigger, ParallelPortTrigger
from mnemonic import Mnemonic
from psychopy.gui import DlgFromDict

import intermodulation.freqtag_spec as spec
import intermodulation.states as ims
import intermodulation.stimuli as imst

##################################
##  Dialog box for subject info ##
##################################

gen = Mnemonic("english")
default_id = "".join(gen.generate(128).split(" ")[:3])
subinfo = {
    "task": "evlab_localizer",
    "subject": default_id,
    "session": 1,
    "date": datetime.now().strftime("%Y-%m-%d_%H-%M"),
    "seed": np.random.randint(0, 255),
    "debug": False,
}
dialog = DlgFromDict(subinfo, fixed=["date"], title="Subject Information")

if dialog.OK:
    if subinfo["debug"]:
        stimpars = {
            "fullscr": spec.debug.FULLSCR,
            "framerate": spec.debug.FRAMERATE,
        }
    else:
        stimpars = {
            "fullscr": spec.FULLSCR,
            "framerate": spec.FRAMERATE,
        }

    if subinfo["debug"]:
        debugdialog = DlgFromDict(stimpars, title="Debug Display Parameter Settings")
else:
    exit()

if subinfo["debug"]:  # Override the experiment parameters with the debug ones if needed
    for attr in spec.debug.__dict__.keys():
        if attr not in ["FULLSCR", "FRAMERATE", "FREQUENCIES"]:
            setattr(spec, attr, spec.debug.__dict__[attr])
    subinfo["seed"] = 0

#####################################
##         Load in stimuli         ##
#####################################

onewordpath = spec.WORDSPATH / "evlab_fast_localizer.csv"

rng = np.random.default_rng(subinfo["seed"])
setidx = rng.choice(np.arange(10))
allstim = (
    pd.read_csv(onewordpath)
    .rename(columns={"origidx": "miniblock", "word": "w1", "category": "condition"})
    .set_index(["seqset", "miniblock"])
)
locwords = allstim.loc[setidx].copy().reset_index()
locwords["w1_freq"] = 0.0
locwords["miniblock"] -= 1

blocktrials = locwords["miniblock"].max() + 1
groups = [df for _, df in locwords.groupby("miniblock")]
rng.shuffle(groups)
locwords = pd.concat(groups).reset_index(drop=True)

print(f"Loaded {len(locwords)} words for localizer to be presented in {blocktrials} miniblocks.")
locdur = blocktrials * (
    spec.LOCALIZER_MINIBLOCK_LEN * spec.LOCALIZER_WORD_DUR + np.mean(spec.LOCALIZER_ITI_BOUNDS) + 1
)
dur_min = np.floor(2 * locdur / 60)
dur_sec = np.round((2 * locdur) % 60)
print(
    f"Should take {locdur} seconds per run, for a total of {dur_min} minutes, {dur_sec} seconds."
)

########################################
## Initialize the window and triggers ##
########################################

clock = pq.Clock()
psylog.setDefaultClock(clock)
spec.WINDOW_CONFIG.update(fullscr=stimpars["fullscr"])
window = psyv.Window(**spec.WINDOW_CONFIG)
# If we have a debug framerate, use that. Otherwise, measure the actual framerate
if not hasattr(spec, "FRAMERATE") and "framerate" not in stimpars:
    framerate = window.getActualFrameRate()
    if framerate is None:
        raise ValueError("Couldn't accurately measure framerate")
    else:
        framerate = np.round(framerate)
else:
    framerate = stimpars["framerate"]

if not hasattr(spec, "TRIGGER") or spec.TRIGGER is None:
    trigger = MockTrigger()
else:
    try:
        trigger = ParallelPortTrigger(spec.TRIGGER)
    except RuntimeError:
        trigger = MockTrigger()


###########################################
## Generate stimuli and update functions ##
###########################################

onewordstim = imst.OneWordStim(
    window,
    "test1",
    reporting_pix=False,
    text_config=spec.TEXT_CONFIG,
)
fixstim = imst.FixationStim(window)


#################################################
## Generate states to use for both experiments ##
#################################################

localizerwords = ims.OneWordMiniblockState(
    next="iti",
    dur=spec.LOCALIZER_WORD_DUR * spec.LOCALIZER_MINIBLOCK_LEN,
    window=window,
    framerate=framerate,
    stim=onewordstim,
    stim_dur=spec.LOCALIZER_WORD_DUR,
    clock=clock,
    frequencies={"word1": None, "fixdot": None},
    word_list=locwords,
    loggables=pe.Loggables(
        start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
        end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
    ),
    log_updates=True,
    strict_freqs=False,
)
fixation = ims.FixationState(
    next="words",
    dur=1.0,
    stim=fixstim,
    window=window,
    clock=clock,
    loggables=pe.Loggables(
        start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
        end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
    ),
    trigger=trigger,
    trigger_val=spec.TRIGGERS.FIXATION,
)

iti = ims.InterTrialState(
    next="fixation",
    duration_bounds=spec.LOCALIZER_ITI_BOUNDS,
    rng=rng,
    loggables=pe.Loggables(
        start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
        end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
    ),
    trigger=trigger,
    trigger_val=spec.TRIGGERS.ITI,
)
states = {
    "words": localizerwords,
    "fixation": fixation,
    "iti": iti,
}


############################################################
## Add logs to send a trigger and tell us when we sent it ##
############################################################


def trigger_val_oneword(state: ims.OneWordMiniblockState, triggers):
    if state.condition == "S":
        return triggers.LOCALIZER.SENTENCE
    elif state.condition == "N":
        return triggers.LOCALIZER.NONWORD
    else:
        raise ValueError(f"Invalid condition {state.condition}")


def trigger_cond_oneword(state: ims.OneWordMiniblockState):
    if any([upd[1] == "text" for upd in state._update_log]):
        return True
    else:
        return False


oneword_starttrig = pe.TriggerTimeLogItem(
    "trigger_time",
    True,
    trigger=trigger,
    value=partial(trigger_val_oneword, triggers=spec.TRIGGERS, state=localizerwords),
)
localizerwords.loggables.add("start", oneword_starttrig)
oneword_updatetrig = pe.TriggerTimeLogItem(
    "update_trigger_t",
    False,
    trigger=trigger,
    value=partial(trigger_val_oneword, triggers=spec.TRIGGERS, state=localizerwords),
    cond=partial(trigger_cond_oneword, state=localizerwords),
)
localizerwords.loggables.add("update", oneword_updatetrig)


def newblock_trig(trigger, triggers):
    trigger.signal(triggers.BLOCKEND)
    return


########################################################
## Create controller, then add pause and quit hotkeys ##
########################################################


def save_and_quit():
    subj = subinfo["subject"]
    date = subinfo["date"]
    controller.logger.save(f"interrupted_{subj}_{date}.pkl")
    controller.quit()
    window.close()
    exit()

def debug_miniblock(state):
    print("miniblock:", state.miniblock_idx)
    print("word:", state.wordset_idx)
    return

controller = pc.ExperimentController(
    states=states,
    window=window,
    start="fixation",
    logger=pe.ExperimentLog(clock),
    clock=clock,
    trial_endstate="iti",
    N_blocks=2,
    K_blocktrials=blocktrials,
    block_calls=[partial(newblock_trig, trigger=trigger, triggers=spec.TRIGGERS)],
    trial_calls=[partial(debug_miniblock, localizerwords)],
)

starting = False


def startnew():
    global starting
    starting = True
    return


def waitloop(text):
    global starting
    while not starting:
        text.draw()
        window.flip()
    return


clock.reset()
expltext = psyv.TextStim(window, **spec.LOCALIZER_EXPL)
psyev.globalKeys.add(key=spec.PAUSE_KEY, func=startnew)
waitloop(expltext)
del expltext
psyev.globalKeys.clear()

psyev.globalKeys.add(key="p", modifiers=["ctrl"], func=controller.toggle_pause)
psyev.globalKeys.add(key=spec.PAUSE_KEY, func=controller.toggle_pause)
psyev.globalKeys.add(key="q", modifiers=["ctrl"], func=save_and_quit)

controller.run_experiment()
if subinfo["debug"]:
    controller.logger.contdf.to_csv("testcont.csv")
    controller.logger.statesdf.to_csv("teststates.csv")
else:
    controller.logger.save(f"localizer_{subinfo['subject']}_{subinfo['date']}.pkl")

trigger.signal(spec.TRIGGERS.EXPEND)
window.close()
