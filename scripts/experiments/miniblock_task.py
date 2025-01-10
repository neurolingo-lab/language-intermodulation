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
    "even_group": bool(np.random.binomial(1, 0.5)),
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
            "f1": spec.debug.FREQUENCIES[0],
            "f2": spec.debug.FREQUENCIES[1],
            "n_mini": None,
            "skip_twoword": False,
            "skip_oneword": False,
        }
    else:
        stimpars = {
            "fullscr": spec.FULLSCR,
            "framerate": spec.FRAMERATE,
            "f1": spec.FREQUENCIES[0],
            "f2": spec.FREQUENCIES[1],
            "skip_twoword": False,
            "skip_oneword": False,
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


####################################
## Load in stimuli for both tasks ##
####################################

group = "even" if subinfo["even_group"] == 0 else "odd"
onewordpath = spec.WORDSPATH / f"{group}_one_word_stimuli.csv"
twowordpath = spec.WORDSPATH / f"{group}_two_word_stimuli.csv"

rng = np.random.default_rng(subinfo["seed"])
# Prepare word stimuli by first shuffling, then assigning frequencies
onewords, twowords, allwords = imu.load_prep_words(
    path_1w=onewordpath,
    path_2w=twowordpath,
    rng=rng,
    miniblock_len=spec.MINIBLOCK_LEN,
    freqs=[stimpars["f1"], stimpars["f2"]],
)
if subinfo["debug"] and stimpars["n_mini"] is not None:
    maxmini = stimpars["n_mini"]
    twowords = twowords.query(f"miniblock < {maxmini}")
    onewords = onewords.query(f"miniblock < {maxmini}")

blocktrials_2w = twowords["miniblock"].max() + 1
blocktrials_1w = onewords["miniblock"].max() + 1


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

if not subinfo["debug"]:
    wordframes = spec.WORD_DUR * framerate
    assert wordframes % 1 == 0, "Word duration must produce a whole number of frames"
    f1_frames = int(np.round(framerate / stimpars["f1"]))
    f2_frames = int(np.round(framerate / stimpars["f2"]))
    assert f1_frames % 2 == 0 and f2_frames % 2 == 0, "Frames per cycle for each freq must be even"


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

query_cats = [
    ("word", "seen"),
    ("word", "unseen"),
    ("nonword", "seen"),
    ("nonword", "unseen"),
]

query_tracker_2w = imu.QueryTracker(
    miniblock=0,
    last_words=twowords.query("miniblock == 0"),
    allwords=allwords,
    categories=query_cats.copy(),
    rng=rng,
)
query_tracker_1w = imu.QueryTracker(
    miniblock=0,
    last_words=onewords.query("miniblock == 0"),
    allwords=allwords,
    categories=query_cats.copy(),
    rng=rng,
)

#################################################
## Generate states to use for both experiments ##
#################################################

twoword = ims.TwoWordMiniblockState(
    next="query",
    dur=spec.WORD_DUR * spec.MINIBLOCK_LEN + spec.WORD_DUR / 1.333333,
    window=window,
    framerate=framerate,
    stim=wordstim,
    stim_dur=spec.WORD_DUR,
    clock=clock,
    frequencies={"w1": stimpars["f1"], "w2": stimpars["f2"], "fixdot": None},
    word_list=twowords,
    loggables=pe.Loggables(
        start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
        end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
    ),
    log_updates=True,
    strict_freqs=False,
)
oneword = ims.OneWordMiniblockState(
    next="query",
    dur=spec.WORD_DUR * spec.MINIBLOCK_LEN + spec.WORD_DUR / 1.333333,
    window=window,
    framerate=framerate,
    stim=onewordstim,
    stim_dur=spec.WORD_DUR,
    clock=clock,
    frequencies={"word1": stimpars["f1"], "fixdot": None},
    word_list=onewords,
    loggables=pe.Loggables(
        start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
        end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
    ),
    log_updates=True,
    strict_freqs=False,
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
    dur=spec.WORD_DUR,
    transition=query_tracker_2w.next_state,
    window=window,
    stim=imst.QueryStim(window),
    clock=clock,
    update_fn=query_tracker_2w.set_next_query,
)
query_1w = ims.QueryState(
    next=["query", "iti"],
    dur=spec.WORD_DUR,
    transition=query_tracker_1w.next_state,
    window=window,
    stim=imst.QueryStim(window),
    clock=clock,
    update_fn=query_tracker_1w.set_next_query,
)
iti = ims.InterTrialState(
    next="fixation",
    duration_bounds=spec.ITI_BOUNDS,
    rng=rng,
    trigger=trigger,
    trigger_val=spec.TRIGGERS.ITI,
)
twoword.end_calls.insert(0, (query_tracker_2w.update_miniblock, (twoword,)))
oneword.end_calls.insert(0, (query_tracker_1w.update_miniblock, (oneword,)))
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
    f1left = leftword_f == stimpars["f1"]
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


def trigger_val_oneword(state: ims.OneWordMiniblockState, triggers):
    f1 = state.frequencies["word1"] == stimpars["f1"]
    if state.condition == "word":
        if f1:
            return triggers.ONEWORD.WORD.F1
        else:
            return triggers.ONEWORD.WORD.F2
    elif state.condition == "non-word":
        if f1:
            return triggers.ONEWORD.NONWORD.F1
        else:
            return triggers.ONEWORD.NONWORD.F2
    else:
        freq = state.frequencies["word1"]
        f1def, f2def = stimpars["f1"], stimpars["f2"]
        raise ValueError(
            f"Invalid cond/freq pair: {state.condition}, {freq:0.3f} Hz.\n"
            f"Possible tags are {f1def:0.3f} and {f2def:0.3f} "
        )


def trigger_cond_oneword(state: ims.OneWordMiniblockState):
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

oneword_starttrig = pe.TriggerTimeLogItem(
    "trigger_time",
    True,
    trigger=trigger,
    value=partial(trigger_val_oneword, triggers=spec.TRIGGERS, state=oneword),
)
oneword.loggables.add("start", oneword_starttrig)
oneword_updatetrig = pe.TriggerTimeLogItem(
    "update_trigger_t",
    False,
    trigger=trigger,
    value=partial(trigger_val_oneword, triggers=spec.TRIGGERS, state=oneword),
    cond=partial(trigger_cond_oneword, state=oneword),
)
oneword.loggables.add("update", oneword_updatetrig)

querytrig = pe.TriggerTimeLogItem(
    "trigger_time",
    True,
    trigger=trigger,
    value=partial(trigger_val_query, state=query, triggers=spec.TRIGGERS),
)
query.loggables.add("start", querytrig)
querytrig_1w = pe.TriggerTimeLogItem(
    "trigger_time",
    True,
    trigger=trigger,
    value=partial(trigger_val_query, state=query_1w, triggers=spec.TRIGGERS),
)
query_1w.loggables.add("start", querytrig_1w)


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


controller_2w = pc.ExperimentController(
    states=states_2w,
    window=window,
    start="fixation",
    logger=pe.ExperimentLog(clock),
    clock=clock,
    trial_endstate="iti",
    N_blocks=spec.N_BLOCKS,
    K_blocktrials=blocktrials_2w,
    block_calls=[partial(newblock_trig, trigger=trigger, triggers=spec.TRIGGERS)],
)
controller_1w = pc.ExperimentController(
    states=states_1w,
    window=window,
    start="fixation",
    logger=pe.ExperimentLog(clock),
    clock=clock,
    trial_endstate="iti",
    N_blocks=spec.N_1W_BLOCKS,
    K_blocktrials=blocktrials_1w,
    block_calls=[partial(newblock_trig, trigger=trigger, triggers=spec.TRIGGERS)],
)

controller = controller_2w

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
if not subinfo["debug"] or not stimpars["skip_twoword"]:
    expltext = psyv.TextStim(window, **spec.TASK1_EXPL)
    psyev.globalKeys.add(key=spec.PAUSE_KEY, func=startnew)
    waitloop(expltext)
    del expltext
    psyev.globalKeys.clear()

    psyev.globalKeys.add(key="p", modifiers=["ctrl"], func=controller_2w.toggle_pause)
    psyev.globalKeys.add(key=spec.PAUSE_KEY, func=controller_2w.toggle_pause)
    psyev.globalKeys.add(key="q", modifiers=["ctrl"], func=save_and_quit)

    controller_2w.run_experiment()
    if subinfo["debug"]:
        controller_2w.logger.contdf.to_csv("testcont.csv")
        controller_2w.logger.statesdf.to_csv("teststates.csv")
    else:
        controller_2w.logger.save(f"twoword_{subinfo['subject']}_{subinfo['date']}.pkl")

psyev.globalKeys.clear()
pausetxt = psyv.TextStim(window, text=spec.INTERTASK_TEXT, pos=(0, 0), height=0.4)
pausetxt.draw()
window.flip()

psyev.globalKeys.add(key=spec.PAUSE_KEY, func=startnew)
waitloop(pausetxt)

starting = False
pausetxt.text = spec.INTERTASK_TEXT2
pausetxt._needSetText = True
pausetxt.draw()
window.flip()

waitloop(pausetxt)
del pausetxt
psyev.globalKeys.clear()

controller = controller_1w
psyev.globalKeys.add(key="p", modifiers=["ctrl"], func=controller_1w.toggle_pause)
psyev.globalKeys.add(key=spec.PAUSE_KEY, func=controller_2w.toggle_pause)
psyev.globalKeys.add(key="q", modifiers=["ctrl"], func=save_and_quit)


if not subinfo["debug"] or not stimpars["skip_oneword"]:
    controller_1w.run_experiment()
    if subinfo["debug"]:
        controller_2w.logger.contdf.to_csv("testcont_1w.csv")
        controller_2w.logger.statesdf.to_csv("teststates_1w.csv")
    else:
        controller_2w.logger.save(f"oneword_{subinfo['subject']}_{subinfo['date']}.pkl")
window.close()
