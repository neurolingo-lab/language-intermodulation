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
import intermodulation.utils as imu


def get_match_params(
    iti,
    fix,
    qpause,
    qdur,
    q_n,
    mini,
    n_oneword,
    n_twoword,
    miniblock_len,
    n_tw_blocks,
    n_ow_blocks,
):
    trial_dur = iti + fix + mini + qpause + q_n * qdur
    sweeptrial_dur = iti + fix + mini
    miniblock_len = 10
    n_ow_mini = n_oneword / miniblock_len
    n_tw_mini = n_twoword / miniblock_len
    n_ow_blocks = 2
    n_tw_blocks = 3
    ow_ft_time = mini * n_ow_mini * n_ow_blocks
    tw_ft_time = mini * n_tw_mini * n_tw_blocks
    ow_total_t = trial_dur * n_ow_mini * n_ow_blocks
    tw_total_t = trial_dur * n_tw_mini * n_tw_blocks
    sweep_ow_match_nmini = n_ow_mini * n_ow_blocks / 2
    print(
        f"Two word trials take a total of {tw_total_t / 60} min yielding {tw_ft_time / 60} min "
        "of FT data"
    )
    print(
        f"One word trials take a total of {ow_total_t / 60} min yielding {ow_ft_time / 60} min "
        f"of FT data, {ow_ft_time / 60 / 2} min per tag"
    )
    print(
        f"To match one-word FT data we need {sweep_ow_match_nmini} miniblocks for a total of "
        f"{sweep_ow_match_nmini * sweeptrial_dur / 60} min of recording per frequency"
    )
    return sweep_ow_match_nmini


########################################
##   Parameters for frequency sweep   ##
########################################

FREQS = np.array([
    4.0,
    5.0,
    6.0,
    7.05882353,
    7.5,
    13.33333333,
    15.0,
    17.14285714,
    20.0,
    24.0,
    30.0,
    40.0,
    60.0,
])

##################################
##  Dialog box for subject info ##
##################################

gen = Mnemonic("english")
default_id = "".join(gen.generate(128).split(" ")[:3])
subinfo = {
    "task": "FTsweep",
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
            "n_mini": None,
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


#########################################################
## Generate shuffled one word lists, one per frequency ##
#########################################################

rng = np.random.default_rng(subinfo["seed"])
group = "even" if rng.choice([True, False]) else "odd"
onewordpath = spec.WORDSPATH / f"{group}_one_word_stimuli.csv"
# Prepare word stimuli by first shuffling, then assigning frequencies
onewords = pd.read_csv(onewordpath, index_col=0)
freqwords = []
start_mini = 0
for f in FREQS:
    conddf = imu.shuffle_condition(onewords, rng)
    conddf = imu.split_miniblocks(conddf, spec.MINIBLOCK_LEN, rng)
    neworder = rng.permutation(conddf["miniblock"].unique())
    conddf["miniblock"] = conddf["miniblock"].map(
        dict(zip(conddf["miniblock"].unique(), neworder))
    )
    conddf = conddf.sort_values("miniblock")
    if stimpars["n_mini"] is not None:
        conddf = conddf.query(f"miniblock < {int(stimpars['n_mini'])}")
    conddf["miniblock"] += start_mini
    start_mini = conddf["miniblock"].max() + 1
    conddf["w1_freq"] = f
    freqwords.append(conddf)
onewords = pd.concat(freqwords, ignore_index=True)

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

onewordstim = imst.OneWordStim(
    window,
    "test1",
    reporting_pix=spec.REPORT_PIX,
    reporting_pix_size=spec.REPORT_PIX_SIZE,
    text_config=spec.TEXT_CONFIG,
)
fixstim = imst.FixationStim(window)

#################################################
## Generate states to use for both experiments ##
#################################################

oneword = ims.OneWordMiniblockState(
    next="iti",
    dur=spec.WORD_DUR * spec.MINIBLOCK_LEN + spec.WORD_DUR / 1.333333,
    window=window,
    framerate=framerate,
    stim=onewordstim,
    stim_dur=spec.WORD_DUR,
    clock=clock,
    frequencies={"word1": onewords.iloc[0]["w1_freq"], "fixdot": None},
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
iti = ims.InterTrialState(
    next="fixation",
    duration_bounds=spec.ITI_BOUNDS,
    rng=rng,
    loggables=pe.Loggables(
        start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
        end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
    ),
    trigger=trigger,
    trigger_val=spec.TRIGGERS.ITI,
)
states_1w = {
    "words": oneword,
    "fixation": fixation,
    "iti": iti,
}

############################################################
## Add logs to send a trigger and tell us when we sent it ##
############################################################


def trigger_val_oneword(state: ims.OneWordMiniblockState):
    blockstart = state.wordset_idx == 0
    freqidx = np.argwhere(FREQS == state.frequencies["word1"]).flatten()[0]
    if blockstart:
        return 30 + freqidx
    else:
        return 40 + freqidx


def trigger_cond_oneword(state: ims.OneWordMiniblockState):
    if any([upd[1] == "text" for upd in state._update_log]):
        return True
    else:
        return False


oneword_starttrig = pe.TriggerTimeLogItem(
    "trigger_time",
    True,
    trigger=trigger,
    value=partial(trigger_val_oneword, state=oneword),
)
oneword.loggables.add("start", oneword_starttrig)
oneword_updatetrig = pe.TriggerTimeLogItem(
    "update_trigger_t",
    False,
    trigger=trigger,
    value=partial(trigger_val_oneword, state=oneword),
    cond=partial(trigger_cond_oneword, state=oneword),
)
oneword.loggables.add("update", oneword_updatetrig)


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


controller = pc.ExperimentController(
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
expltext = psyv.TextStim(window, **spec.TASK1_EXPL)
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
    controller.logger.save(f"{subinfo['task']}_{subinfo['subject']}_{subinfo['date']}.pkl")
