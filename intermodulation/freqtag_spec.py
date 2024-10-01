from types import SimpleNamespace
from typing import Hashable, Sequence

import numpy as np
from byte_triggers import ParallelPortTrigger

import intermodulation.core as core

LOGGABLES = {
    "per_state": [
        "state_number",
        "state",
        "next_state",
        "state_start",
        "target_end",
        "state_end",
        "trial_number",
        "block_number",
        "block_trial",
        "trial_end",
        "block_end",
        "word1",
        "word2",
        "word1_freq",
        "word2_freq",
        "condition",
    ],
    "continuous_per_state": [
        ("words", "word1"),
        ("words", "word2"),
    ],
}

# Trigger codes for the experiment to send via the parallel port
# Nested logic is used to group triggers by condition and sub-condition
# Organization is as follows:
# - 10-20: General triggers
# - 20-30: Query condition triggers (nexted namespace for query word being in
#      previous trial (TRUE) or not (FALSE))
# - 30-40: Two-word stimulus condition triggers (nested namespace for phrase, non-phrase, non-word)
#      And within each condition (P, NP, NW) another nested namespace for frequency tag 1 being on
#      the left or right
# - 40-50: One-word stimulus condition triggers (nested namespace for word, non-word)
#      And within each condition (W, NW) another nested namespace for frequency tag 1 being on L/R
TRIGGERS = SimpleNamespace(
    STATEEND=10,
    TRIALEND=11,
    BLOCKEND=12,
    ITI=13,
    FIXATION=14,
    BREAK=15,
    INTERBLOCK=16,
    ABORT=17,
    ERROR=18,
    EXPEND=255,
    # 20-30 are reserved for the query condition
    QUERY=SimpleNamespace(
        TRUE=SimpleNamespace(
            F1LEFT=20,
            F1RIGHT=21,
        ),
        FALSE=SimpleNamespace(
            F1LEFT=22,
            F1RIGHT=23,
        ),
    ),
    # 30-40 are reserved for the two-word stimulus condition
    TWOWORD=SimpleNamespace(
        PHRASE=SimpleNamespace(
            F1LEFT=30,
            F1RIGHT=31,
        ),
        NONPHRASE=SimpleNamespace(
            F1LEFT=32,
            F1RIGHT=33,
        ),
        NONWORD=SimpleNamespace(
            F1LEFT=34,
            F1RIGHT=35,
        ),
    ),
    # 40-50 are reserved for the one-word stimulus condition
    ONEWORD=SimpleNamespace(
        WORD=SimpleNamespace(
            F1=40,
            F2=41,
        ),
        NONWORD=SimpleNamespace(
            F1=42,
            F2=43,
        ),
    ),
)


def assign_frequencies_to_words(wordsdf, freq1: float, freq2: float, rng: np.random.Generator):
    """
    Counterbalanced assignment of one of two frequencies to each word in a DataFrame. Returns
    a new DataFrame with the columns "w1_freq" and "w2_freq" added. Each pair will have one of
    each frequency.

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
    mask = np.zeros(len(wordsdf), dtype=bool)
    f_choices = rng.choice(np.arange(len(wordsdf)), size=len(wordsdf) // 2, replace=False)
    mask[f_choices] = True
    w1_f = np.where(mask, freq1, freq2)
    w2_f = np.where(mask, freq2, freq1)

    outdf["w1_freq"] = w1_f
    outdf["w2_freq"] = w2_f
    return outdf


def add_logging_to_controller(
    controller: core.ExperimentController,
    states: dict,
    query: Hashable,
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
    controller.add_loggable(query, "start", "word1", object=states[query], attribute="test_word")


def add_triggers_to_controller(
    controller: core.ExperimentController,
    trigger: ParallelPortTrigger | None,
    freqs: Sequence,
    states: dict,
    iti: Hashable,
    fixation: Hashable,
    query: Hashable,
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
    wordstate = twoword if twoword is not None else oneword

    # Make sure we have exactly two frequencies
    if len(freqs) != 2:
        raise ValueError("freqs must be a sequence of two frequencies.")
    # Make sure every passed state is in the states dictionary
    missing = []
    for state in [iti, fixation, query, wordstate]:
        if state not in states:
            missing.append(state)
    if len(missing) > 0:
        raise ValueError(f"Missing states in the passed states dictionary: {missing}")

    if trigger is None:
        return

    # Initialize the call lists for each state if they don't exist, and add the state end triggers
    for state in [iti, fixation, query, wordstate, "pause"]:
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
    return
