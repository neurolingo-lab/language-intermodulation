from collections.abc import Sequence
from pathlib import Path
from typing import Hashable, Literal, Mapping

import numpy as np
import pandas as pd
import psystate.controller as psycon
from byte_triggers import ParallelPortTrigger

import intermodulation.freqtag_spec as spec
from intermodulation.freqtag_spec import (
    TRIGGERS,
)
from intermodulation.states import (
    QueryState,
    TwoWordMiniblockState,
)


def update_tracker_miniblock(q_track: Mapping, state: TwoWordMiniblockState):
    """
    Update the query tracker (see `set_next_query_miniblock`) with the current miniblock
    number and set of words.

    Parameters
    ----------
    q_track : Mapping
        Dictionary of query tracking information. See `set_next_query_miniblock` for details.
    state : TwoWordMiniblockState
        The miniblock stimulus state that should be used to update the query tracker.
    """
    if q_track["miniblock"] != state.miniblock_idx:
        q_track["miniblock"] = state.miniblock_idx
        q_track["last_words"] = state.wordset
    return


def next_state_query_miniblock(
    q_track: Mapping,
    nexts: Sequence[Hashable] = ["query", "iti"],
    query_id: Hashable = "query",
    iti_id="iti",
):
    """
    Set whether the next state should be a query or the ITI state based on the current miniblock

    Parameters
    ----------
    q_track : Mapping
        Tracking dictionary for the query states. See `set_next_query_miniblock` for details.
    nexts : Sequence[Hashable], optional
        A sequence of the next possible state IDs, by default ["query", "iti"]
    query_id : Hashable, optional
        The ID of the query state, to be returned if we haven't reached our target number of
        queries, by default "query"
    iti_id : str, optional
        The ID of the ITI state, to be returned if we have queried all categories, by default "iti"
    """
    if (q_track["query_idx"] + 1) % (len(q_track["categories"])) == 0:
        return nexts.index(iti_id)
    return nexts.index(query_id)


def set_next_query_miniblock(
    q_track: Mapping,
    state: QueryState,
    allwords: pd.DataFrame,
    rng: np.random.RandomState,
):
    """
    Function to get the next query following the miniblock for the miniblock task. Will use the
    query tracker to keep track of which categories are desired (see q_track below), and set the
    test word and truth value in the passed QueryState object.

    Parameters
    ----------
    q_track : Mapping
        Dictionary storing the information about queries between state runs. Must have keys
        'query_idx', 'query_order', 'categories', 'last_words', 'candidates', 'miniblock'.
         - 'query_idx' is an integer tracking the number of queries so far
         - 'query_order' is a list of indices defining the order of categories to query, permuted by
            this function once all categories have been used
         - 'categories' is a list of 2-tuples defining the categories to query, which are
            combinations of 'word' or 'non-word' and 'seen' or 'unseen'.
         - 'last_words' is a DataFrame storing the last words presented in the previous miniblock,
            and should be updated after each miniblock is completed by another function.
         - 'candidates' is a dictionary storing the candidate words for each category, sampled from
            the 'allwords' DataFrame
         - 'miniblock' is an integer tracking the current miniblock number
    state : QueryState
        The query state that should be updated with the next test word
    allwords : pd.DataFrame
        All words used in the current task, to define the 'seen'/'unseen' status of the words
    rng : np.random.RandomState
        Random generator to use
    """
    # Store some variables locally for easier access
    n_queries = len(q_track["categories"])
    qidx = q_track["query_idx"] % n_queries
    catnum = q_track["query_order"][qidx]
    if qidx == 0:  # If we have queried all categories, permute the order again and get new cands
        q_track["query_order"] = rng.permutation(n_queries)
        candidates = {}
        for cat in q_track["categories"]:
            # Set the mask of words that meet our word/non-word choices
            if cat[0] == "word":
                wordmask = allwords["cond"] == cat[0]
            else:
                wordmask = allwords["cond"] != "word"
            # Set the columns containing our last seen words. If we have a second word, use it
            columns = ["w1"]
            if "w2" in q_track["last_words"].columns:
                columns.append("w2")
            # Set the mask of words that were seen or unseen in the last miniblock
            seenmask = allwords["word"].isin(q_track["last_words"][columns].values.flat)
            if cat[1] == "unseen":
                seenmask = ~seenmask
            # Sample the candidates for this category
            candidates[cat] = allwords[wordmask & seenmask].copy().sample(frac=1, random_state=rng)
        q_track["candidates"] = candidates  # Store the candidates for the next queries
    qcat = q_track["categories"][catnum]
    # Set the next word to be tested and whether the correct response is True (seen) or False
    state.test_word = q_track["candidates"][qcat].sample(random_state=rng)["word"].values[0]
    seen = qcat[1]
    state.truth = True if seen == "seen" else False

    q_track["query_idx"] += 1
    return


def assign_miniblock_freqs(
    task: Literal["twoword", "oneword"],
    rng: np.random.Generator,
    df: pd.DataFrame,
    miniblock_len: int,
    freqs: Sequence[float],
) -> pd.DataFrame:
    df = df.copy()
    if task not in ["twoword", "oneword"]:
        raise ValueError("task must be either 'twoword' or 'oneword'.")
    n_mini = len(df) / miniblock_len  # Number of mini-blocks
    if n_mini % 1 != 0:  # make sure we're not dropping stimuli
        raise ValueError("Number of mini-blocks must be an integer. Miniblock length is wrong.")
    else:
        n_mini = int(n_mini)
    freqidxs = np.concat((np.zeros(n_mini // 2), np.ones(n_mini // 2)))  # Balance number F1/F2
    miniblock_nums = np.arange(n_mini)
    rng.shuffle(freqidxs)  # Randomize order
    freqidxs = np.repeat(freqidxs, miniblock_len).reshape(-1, 1)
    miniblock_nums = np.repeat(miniblock_nums, miniblock_len).reshape(-1, 1)
    tilefreqs = np.tile(spec.FREQUENCIES, len(freqidxs)).reshape(-1, 2)
    freqs = np.where(freqidxs == 0, tilefreqs, tilefreqs[:, ::-1])
    if task == "twoword":
        df["w1_freq"] = freqs[:, 0]
        df["w2_freq"] = freqs[:, 1]
    else:
        df["w1_freq"] = freqs[:, 0]
    df["miniblock"] = miniblock_nums

    # Check that the frequencies are balanced within each miniblock
    assert all(df["w1_freq"].value_counts() == len(df) / 2)
    twoword_minis = [
        df.query(f"miniblock == {mini}") for mini in range(int(len(df) / spec.MINIBLOCK_LEN))
    ]
    assert all([all(minib["w1_freq"] == minib.iloc[0]["w1_freq"]) for minib in twoword_minis])
    if task == "twoword":
        assert all(df["w2_freq"].value_counts() == len(df) / 2)
        assert all([all(minib["w2_freq"] == minib.iloc[0]["w2_freq"]) for minib in twoword_minis])
    return df


def load_prep_words(
    path_1w: str | Path,
    path_2w: str | Path,
    rng: np.random.Generator,
    miniblock_len: int,
    freqs: Sequence[float],
) -> pd.DataFrame:
    # Prepare word stimuli by first shuffling, then assigning frequencies
    twowords = pd.read_csv(path_2w, index_col=0)
    twowords = twowords.sample(frac=1, random_state=rng)
    twowords = assign_miniblock_freqs("twoword", rng, twowords, miniblock_len, freqs)
    onewords = pd.read_csv(path_1w, index_col=0)
    onewords = onewords.sample(frac=1, random_state=rng)
    onewords = assign_miniblock_freqs("oneword", rng, onewords, miniblock_len, freqs)

    # Generate a list of all used words together with their categories, for the query task
    all_2w = pd.melt(
        twowords,
        id_vars=["w1_type", "w2_type"],
        value_vars=["w1", "w2"],
        var_name="position",
        value_name="word",
    )
    all_2w["cond"] = all_2w["w1_type"].where(all_2w["position"] == "w1", all_2w["w2_type"])
    all_2w = all_2w[["word", "cond"]]
    allwords = pd.concat(
        [
            all_2w,
            onewords[["w1", "condition"]].rename(columns={"w1": "word", "condition": "cond"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    return onewords, twowords, allwords


def add_triggers_to_controller(
    controller: psycon.ExperimentController,
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
