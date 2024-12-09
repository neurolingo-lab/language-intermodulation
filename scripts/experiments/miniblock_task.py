from datetime import datetime
from functools import partial

import numpy as np
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
import intermodulation.utils as imu

##################################
##  Dialog box for subject info ##
##################################

gen = Mnemonic("english")
default_id = "".join(gen.generate(128).split(" ")[:3])
subinfo = {
    "task": "miniblockIM",
    "subject": default_id,
    "session": 1,
    "date": datetime.now().strftime("%Y-%m-%d_%H-%M"),
    "seed": np.random.randint(0, 255),
    "debug": False,
}
dialog = DlgFromDict(subinfo, fixed=["date"], title="Subject Information")

if dialog.OK:
    pass
else:
    exit()

if subinfo["debug"]:  # Override the experiment parameters with the debug ones if needed
    for attr in spec.debug.__dict__.keys():
        setattr(spec, attr, spec.debug.__dict__[attr])
    subinfo["seed"] = 0


####################################
## Load in stimuli for both tasks ##
####################################

rng = np.random.default_rng(subinfo["seed"])
# Prepare word stimuli by first shuffling, then assigning frequencies
onewords, twowords, allwords = imu.load_prep_words(
    path_1w=spec.ONEWORDPATH,
    path_2w=spec.TWOWORDPATH,
    rng=rng,
    miniblock_len=spec.MINIBLOCK_LEN,
    freqs=spec.FREQUENCIES,
)

########################################
## Initialize the window and triggers ##
########################################

clock = pq.Clock()
psylog.setDefaultClock(clock)
window = psyv.Window(**spec.WINDOW_CONFIG)
# If we have a debug framerate, use that. Otherwise, measure the actual framerate
if not hasattr(spec, "FRAMERATE"):
    framerate = window.getActualFrameRate()
    if framerate is None:
        raise ValueError("Couldn't accurately measure framerate")
    else:
        framerate = np.round(framerate)
else:
    framerate = spec.FRAMERATE

if not hasattr(spec, "TRIGGER") or spec.TRIGGER is None:
    trigger = MockTrigger()
else:
    trigger = ParallelPortTrigger(spec.TRIGGER)

###########################################
## Generate stimuli and update functions ##
###########################################

wordstim = imst.TwoWordStim(
    window,
    "test1",
    "test2",
    reporting_pix=spec.REPORT_PIX,
    reporting_pix_size=spec.REPORT_PIX_SIZE,
    separation=0.3,
    text_config=spec.TEXT_CONFIG,
)
onewordstim = imst.OneWordStim(
    window,
    "test1",
    reporting_pix=spec.REPORT_PIX,
    reporting_pix_size=spec.REPORT_PIX_SIZE,
    text_config=spec.TEXT_CONFIG,
)
fixstim = imst.FixationStim(window)

query_tracker = {
    "miniblock": 0,
    "last_words": twowords.query("miniblock == 0"),
    "query_idx": 0,
    "categories": [
        ("word", "seen"),
        ("word", "unseen"),
        ("nonword", "seen"),
        ("nonword", "unseen"),
    ],
    "query_order": rng.permutation(4),
}
query_tracker_1w = {
    "miniblock": 0,
    "last_words": onewords.query("miniblock == 0"),
    "query_idx": 0,
    "categories": [
        ("word", "seen"),
        ("word", "unseen"),
        ("nonword", "seen"),
        ("nonword", "unseen"),
    ],
    "query_order": rng.permutation(4),
}

qchoice = partial(imu.set_next_query_miniblock, allwords=allwords, rng=rng)

qnext = partial(imu.next_state_query_miniblock, query_tracker)
qupdate = partial(imu.update_tracker_miniblock, query_tracker)

qnext_1w = partial(imu.next_state_query_miniblock, query_tracker_1w)
qupdate_1w = partial(imu.update_tracker_miniblock, query_tracker_1w)

#################################################
## Generate states to use for both experiments ##
#################################################

twoword = ims.TwoWordMiniblockState(
    next="query",
    dur=spec.WORD_DUR * spec.MINIBLOCK_LEN,
    window=window,
    framerate=framerate,
    stim=wordstim,
    stim_dur=spec.WORD_DUR,
    clock=clock,
    frequencies={"w1": spec.FREQUENCIES[0], "w2": spec.FREQUENCIES[1], "fixdot": None},
    word_list=twowords,
    loggables=pe.Loggables(
        start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
        end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
    ),
    log_updates=True,
    strict_freqs="allow",
)
oneword = ims.OneWordMiniblockState(
    next="query",
    dur=spec.WORD_DUR * spec.MINIBLOCK_LEN,
    window=window,
    framerate=framerate,
    stim=onewordstim,
    stim_dur=spec.WORD_DUR,
    clock=clock,
    frequencies={"word1": spec.FREQUENCIES[0], "fixdot": None},
    word_list=onewords,
    loggables=pe.Loggables(
        start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
        end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
    ),
    log_updates=True,
    strict_freqs="allow",
)
fixation = ims.FixationState(
    next="words",
    dur=2.0,
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
query = ims.QueryState(
    next=["query", "iti"],
    dur=2.0,
    transition=qnext,
    window=window,
    stim=imst.QueryStim(window),
    clock=clock,
    query_tracker=query_tracker,
    update_fn=qchoice,
)
query_1w = ims.QueryState(
    next=["query", "iti"],
    dur=2.0,
    transition=qnext_1w,
    window=window,
    stim=imst.QueryStim(window),
    clock=clock,
    query_tracker=query_tracker_1w,
    update_fn=qchoice,
)
iti = ims.InterTrialState(
    next="fixation",
    duration_bounds=spec.ITI_BOUNDS,
    rng=rng,
    trigger=trigger,
    trigger_val=spec.TRIGGERS.ITI,
)
twoword.end_calls.insert(0, (qupdate, (twoword,)))
oneword.end_calls.insert(0, (qupdate_1w, (oneword,)))
states_2w = {
    "words": twoword,
    "fixation": fixation,
    "query": query,
    "iti": iti,
}
states_1w = {
    "words": oneword,
    "fixation": fixation,
    "query": query_1w,
    "iti": iti,
}

############################################################
## Add logs to send a trigger and tell us when we sent it ##
############################################################


def trigger_val_query(state: ims.QueryState, triggers):
    if state.truth:
        return triggers.QUERY.TRUE
    else:
        return triggers.QUERY.FALSE


def trigger_val_twoword(state: ims.TwoWordMiniblockState, triggers):
    leftword_f = state.frequencies["word1"]
    f1left = leftword_f == spec.FREQUENCIES[0]
    if state.condition == "phrase":
        if f1left:
            return triggers.TWOWORD.PHRASE.F1LEFT
        else:
            return triggers.TWOWORD.PHRASE.F1RIGHT
    elif state.condition == "non-phrase":
        if f1left:
            return triggers.TWOWORD.NONPHRASE.F1LEFT
        else:
            return triggers.TWOWORD.NONPHRASE.F1RIGHT
    elif state.condition == "non-word":
        if f1left:
            return triggers.TWOWORD.NONWORD.F1LEFT
        else:
            return triggers.TWOWORD.NONWORD.F1RIGHT
    else:
        raise ValueError(
            f"Invalid cond/freq pair: {state.condition}, {f1left}, freq left = {leftword_f}"
        )


def trigger_cond_twoword(state: ims.TwoWordMiniblockState):
    if any([upd[1] == "text" for upd in state._update_log]):
        return True
    else:
        return False


twoword_starttrig = pe.TriggerTimeLogItem(
    "trigger_time",
    True,
    trigger=trigger,
    value=partial(trigger_val_twoword, triggers=spec.TRIGGERS, state=twoword),
)
twoword.loggables.add("start", twoword_starttrig)
twoword_updatetrig = pe.TriggerTimeLogItem(
    "update_trigger_t",
    False,  # Not unique, since it trigger multiple times per state
    trigger=trigger,
    value=partial(trigger_val_twoword, triggers=spec.TRIGGERS, state=twoword),
    cond=partial(trigger_cond_twoword, state=twoword),
)
twoword.loggables.add("update", twoword_updatetrig)
querytrig = pe.TriggerTimeLogItem(
    "trigger_time",
    True,
    trigger=trigger,
    value=partial(trigger_val_query, state=query, triggers=spec.TRIGGERS),
)
query.loggables.add("start", querytrig)

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


controller_2w = pc.ExperimentController(
    states=states_2w,
    window=window,
    start="fixation",
    logger=pe.ExperimentLog(clock),
    clock=clock,
    trial_endstate="words",
    N_blocks=spec.N_BLOCKS,
    K_blocktrials=twowords["miniblock"].max(),
)
controller_1w = pc.ExperimentController(
    states=states_1w,
    window=window,
    start="fixation",
    logger=pe.ExperimentLog(clock),
    clock=clock,
    trial_endstate="words",
    N_blocks=spec.N_BLOCKS,
    K_blocktrials=onewords["miniblock"].max(),
)

controller = controller_2w
psyev.globalKeys.add(key="p", modifiers=["ctrl"], func=controller_2w.toggle_pause)
psyev.globalKeys.add(key="q", modifiers=["ctrl"], func=save_and_quit)

clock.reset()
# controller_2w.run_experiment()
# controller_2w.logger.contdf.to_csv("testcont.csv")
# controller_2w.logger.statesdf.to_csv("teststates.csv")

psyev.globalKeys.clear()
controller = controller_1w
psyev.globalKeys.add(key="p", modifiers=["ctrl"], func=controller_1w.toggle_pause)
psyev.globalKeys.add(key="q", modifiers=["ctrl"], func=save_and_quit)

controller_1w.run_experiment()
controller_1w.logger.contdf.to_csv("testcont_1w.csv")
controller_1w.logger.statesdf.to_csv("teststates_1w.csv")
window.close()
