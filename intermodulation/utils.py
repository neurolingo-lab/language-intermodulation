from collections.abc import Sequence
from typing import Hashable

import numpy as np
import pandas as pd
from byte_triggers import ParallelPortTrigger

import intermodulation.core.controller as core
from intermodulation.freqtag_spec import (
    DOT_CONFIG,
    REPORT_PIX,
    REPORT_PIX_SIZE,
    TEXT_CONFIG,
    TRIGGERS,
    WORD_SEP,
)
from intermodulation.states import (
    FixationState,
    InterTrialState,
    OneWordState,
    QueryState,
    TwoWordState,
)
from intermodulation.stimuli import FixationStim, OneWordStim, QueryStim, TwoWordStim


def flip_state(t, target_t, keymask, framerate):
    close_enough = np.isclose(t, target_t, rtol=0.0, atol=1 / (2 * framerate) - 1e-6)
    past_t = t > target_t
    goodclose = (close_enough & keymask) | (past_t & keymask)
    # breakpoint()
    if np.any(goodclose):
        ts_idx = np.argwhere(goodclose).flatten()[-1]
        keymask[ts_idx] = False
        return True, keymask
    return False, keymask


# def infer_states(events, triggers, first_samp=0):
#     """
#     Infer the state start/end times from a set of events produced by `mne.find_events` and a
#     set of known trigger values.

#     Parameters
#     ----------
#     events : np.ndarray
#         N x 3 array of events produced by `mne.find_events`.
#     triggers : AttriDict
#         A nested AttriDict of different trigger types and their corresponding values.
#     first_samp : int
#         The index of the first sample in the data. Usually taken from mne.Raw.first_samp. Default 0
#     """
#     lut = {v: k for k, v in nested_iteritems(triggers)}

#     endmask = events[:, 2] == triggers.STATEEND
#     starttimes = events[~endmask]
#     endtimes = events[endmask]
#     records = []
#     for tidx, prev, trig in starttimes:
#         if trig in lut:
#             stidx = np.searchsorted(endtimes[:, 0], tidx)
#             next_end = np.argwhere(endtimes[:, 2][stidx:] == triggers.STATEEND).flatten()[0]


def assign_frequencies_to_words(wordsdf, freq1: float, freq2: float, rng: np.random.Generator):
    """
    Counterbalanced assignment of one of two frequencies to each word in a DataFrame, within
    condition. Returns a new DataFrame with the columns "w1_freq" and "w2_freq" added. Each pair
    will have one of each frequency.

    Parameters
    ----------
    wordsdf : pd.DataFrame
        Dataframe of word pairs to assign frequencies to.
    freq1 : float
        Value to assign to half of the words.
    freq2 : float
        Value to assign to the other half of the words.
    rng : np.random.Generator
        RNG object to use
    """
    outdf = wordsdf.copy()
    outdf["w1_freq"] = np.nan
    if "w2" in wordsdf.columns:
        outdf["w2_freq"] = np.nan
    for condition in wordsdf["condition"].unique():
        condidx = wordsdf.query("condition == @condition").index
        mask = np.zeros(len(condidx), dtype=bool)
        f1_choices = rng.choice(np.arange(len(condidx)), size=len(condidx) // 2, replace=False)
        mask[f1_choices] = True
        w1_f = np.where(mask, freq1, freq2)
        outdf.loc[condidx, "w1_freq"] = w1_f
        if "w2" in wordsdf.columns:
            w2_f = np.where(mask, freq2, freq1)
            outdf.loc[condidx, "w2_freq"] = w2_f

    return outdf


def generate_1w_states(
    rng, FIXATION_DURATION, WORD_DURATION, ITI_BOUNDS, clock, window, framerate, worddf
):
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

    return states_1word


def add_masked_1w_states(states_1word, word_list, mask_char="+"):
    maskdf = word_list.copy()
    maskdf["w1"] = maskdf["w1"].apply(lambda x: mask_char * len(x))
    states_1word["fixation"].next = "mask"
    states_1word["mask"] = OneWordState(
        window=states_1word["fixation"].window,
        next="word",
        dur=states_1word["fixation"].dur,
        stim=OneWordStim(
            win=states_1word["fixation"].window,
            word1="testing",
            text_config=TEXT_CONFIG,
            reporting_pix=REPORT_PIX,
            reporting_pix_size=REPORT_PIX_SIZE,
        ),
        word_list=maskdf,
        frequencies={"words": {"word1": None}},
        clock=states_1word["fixation"].clock,
        framerate=states_1word["fixation"].framerate,
        flicker_handler="frame_count",
    )
    return


def generate_2w_states(
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
):
    states_2word = {
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

    return states_2word


def add_logging_to_controller(
    controller: core.ExperimentController,
    states: dict,
    query: Hashable | None = None,
    twoword: Hashable | None = None,
    oneword: Hashable | None = None,
):
    if twoword is None and oneword is None:
        raise ValueError("At least one of twoword or oneword must be provided.")
    elif twoword is not None and oneword is not None:
        raise ValueError("Only one of twoword or oneword can be provided.")
    if twoword is not None:
        controller.add_loggable(
            twoword, "start", "word1", object=states[twoword].stim, attribute="word1"
        )
        controller.add_loggable(
            twoword, "start", "word2", object=states[twoword].stim, attribute="word2"
        )
        controller.add_loggable(
            twoword,
            "start",
            "word1_freq",
            object=states[twoword],
            attribute=("frequencies", "words", "word1"),
        )
        controller.add_loggable(
            twoword,
            "start",
            "word2_freq",
            object=states[twoword],
            attribute=("frequencies", "words", "word2"),
        )
        controller.add_loggable(
            twoword, "start", "condition", object=states[twoword], attribute="phrase_cond"
        )
        controller.add_loggable(
            query, "start", "word1", object=states[query], attribute="test_word"
        )
        controller.add_loggable(query, "start", "truth", object=states[query], attribute="truth")

    elif oneword is not None:
        controller.add_loggable(
            oneword, "start", "word1", object=states[oneword].stim, attribute="word1"
        )
        controller.add_loggable(
            oneword,
            "start",
            "word1_freq",
            object=states[oneword],
            attribute=("frequencies", "words", "word1"),
        )
        controller.add_loggable(
            oneword, "start", "condition", object=states[oneword], attribute="word_cond"
        )
    return


def add_triggers_to_controller(
    controller: core.ExperimentController,
    trigger: ParallelPortTrigger | None,
    freqs: Sequence,
    states: dict,
    iti: Hashable,
    fixation: Hashable,
    query: Hashable | None = None,
    twoword: Hashable | None = None,
    oneword: Hashable | None = None,
):
    # Check that only one of twoword or oneword is provided, as they need to be run in separate
    # experiment controllers to avoid conflicts
    if twoword is None and oneword is None:
        if trigger is None:
            return
        raise ValueError("At least one of twoword or oneword must be provided.")
    elif twoword is not None and oneword is not None:
        raise ValueError("Only one of twoword or oneword can be provided.")

    # Make sure we have exactly two frequencies
    if len(freqs) != 2:
        raise ValueError("freqs must be a sequence of two frequencies.")
    # Make sure every passed state is in the states dictionary
    wordstate = twoword if twoword is not None else oneword
    stateslist = [iti, fixation, "pause", wordstate]
    if query is not None:
        stateslist.append(query)

    missing = []
    for state in stateslist:
        if state not in states:
            missing.append(state)
    if len(missing) > 0:
        raise ValueError(f"Missing states in the passed states dictionary: {missing}")

    if trigger is None:
        return

    # Initialize the call lists for each state if they don't exist, and add the state end triggers
    for state in stateslist:
        if state not in controller.state_calls:
            controller.state_calls[state] = {}
        if "end" not in controller.state_calls[state]:
            controller.state_calls[state]["end"] = []
        if "start" not in controller.state_calls[state]:
            controller.state_calls[state]["start"] = []
        controller.state_calls[state]["end"].append(
            (
                trigger.signal,
                (TRIGGERS.STATEEND,),
            )
        )
        if state == "pause":
            trig = TRIGGERS.BREAK
        elif state == iti:
            trig = TRIGGERS.ITI
        elif state == fixation:
            trig = TRIGGERS.FIXATION
        elif state == query:
            continue
        else:
            continue
        controller.state_calls[state]["start"].append(
            (
                trigger.signal,
                (trig,),
            )
        )

    # Add the universal block and trial end triggers, and the pause state trigger
    controller.trial_calls.append(
        (
            trigger.signal,
            (TRIGGERS.TRIALEND,),
        )
    )
    controller.block_calls.append(
        (
            trigger.signal,
            (TRIGGERS.BLOCKEND,),
        )
    )
    controller.state_calls["pause"]["start"].append(
        (
            trigger.signal,
            (TRIGGERS.BREAK,),
        )
    )

    # Define the functions that will be called to report exactly which condition and freq tag is
    # used for each state in one word and two word conditions
    def choose_2word_trigger(state, freqs, trigger):
        match state.phrase_cond:
            case "phrase":
                st_trig = TRIGGERS.TWOWORD.PHRASE
            case "non-phrase":
                st_trig = TRIGGERS.TWOWORD.NONPHRASE
            case "non-word":
                st_trig = TRIGGERS.TWOWORD.NONWORD
            case _:
                raise ValueError(f"Unexpected condition: {state.phrase_cond}")

        if np.isclose(state.frequencies["words"]["word1"], freqs[0]):
            trigval = st_trig.F1LEFT
        elif np.isclose(state.frequencies["words"]["word2"], freqs[0]):
            trigval = st_trig.F1RIGHT
        else:
            raise ValueError("No tagging frequency matched the passed frequencies.")

        trigger.signal(trigval)
        return

    def choose_1word_trigger(state, freqs, trigger):
        match state.word_cond:
            case "word":
                st_trig = TRIGGERS.ONEWORD.WORD
            case "non-word":
                st_trig = TRIGGERS.ONEWORD.NONWORD
            case _:
                raise ValueError(f"Unexpected condition: {state.word_cond}")
        if np.isclose(state.frequencies["words"]["word1"], freqs[0]):
            trigval = st_trig.F1
        elif np.isclose(state.frequencies["words"]["word1"], freqs[1]):
            trigval = st_trig.F2
        else:
            raise ValueError("No tagging frequency matched the passed frequencies.")
        trigger.signal(trigval)
        return

    def choose_query_trigger(state, freqs, trigger):
        if state.truth:
            trigger.signal(TRIGGERS.QUERY.TRUE)
        else:
            trigger.signal(TRIGGERS.QUERY.FALSE)
        return

    # Add the trigger calls to the controller for the given word state
    if twoword is not None:
        controller.state_calls[twoword]["start"].append(
            (
                choose_2word_trigger,
                (states[twoword], freqs, trigger),
            )
        )
    else:
        controller.state_calls[oneword]["start"].append(
            (
                choose_1word_trigger,
                (states[oneword], freqs, trigger),
            )
        )

    if query is not None:
        controller.state_calls[query]["start"].append(
            (
                choose_query_trigger,
                (states[query], freqs, trigger),
            )
        )
    return
