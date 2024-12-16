from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Literal

import numpy as np
import pandas as pd
import psystate.controller as psycon
from byte_triggers import ParallelPortTrigger

from intermodulation.freqtag_spec import (
    TRIGGERS,
)
from intermodulation.states import (
    QueryState,
)


@dataclass
class QueryTracker:
    miniblock: int
    categories: list[tuple[Literal["word", "non-word"], Literal["seen", "unseen"]]]
    last_words: pd.DataFrame
    allwords: pd.DataFrame
    rng: np.random.RandomState

    def __post_init__(self):
        self.candidates: dict[tuple[str, str], pd.DataFrame] = {}
        self.remaining_cat: list[tuple[Literal["word", "non-word"], Literal["seen", "unseen"]]] = (
            self._get_valid_cats()
        )

    def update_miniblock(self, state):
        if self.miniblock != state.miniblock_idx:
            self.miniblock = state.miniblock_idx
            self.last_words = state.wordset
        return

    def next_state(
        self, nexts: Sequence[Hashable] = ["query", "iti"], query_id="query", iti_id="iti"
    ):
        # If we're on the last of our categories, return the index of the ITI state (end queries).
        # Otherwise return the index of the query state (keep querying).
        if len(self.remaining_cat) == 1:
            return nexts.index(iti_id)
        return nexts.index(query_id)

    def set_next_query(self, state: QueryState):
        # See how many categories are left to query. If there's none, we're at the start of a new
        # query set and need to regenerate the categories and candidates.
        rem_cat = len(self.remaining_cat)
        first_q = self.miniblock == 0 and rem_cat == 4
        if rem_cat == 0 or first_q:
            if not first_q:
                self.remaining_cat = self._get_valid_cats()
            candidates = self._set_candidates(self.allwords)
            self.candidates = candidates

        # Pop the next category to query and set the test word and truth value in the passed state
        qcat = self.remaining_cat.pop()
        catdf = self.candidates[qcat]
        state.test_word = catdf.sample(random_state=self.rng)["word"].values[0]
        seen = qcat[1]

        wordcols = ["w1", "w2"] if "w2" in self.last_words.columns else ["w1"]
        if seen == "seen":
            assert state.test_word in self.last_words[wordcols].values.flat
        else:
            assert state.test_word not in self.last_words[wordcols].values.flat

        state.truth = True if seen == "seen" else False
        return

    def _set_candidates(self, allwords: pd.DataFrame):
        candidates = {}
        for cat in self.remaining_cat:
            if cat in candidates:
                continue
            if cat[0] == "word":
                wordmask = allwords["cond"] == cat[0]
            else:
                wordmask = allwords["cond"] != "word"
            columns = ["w1"]
            if "w2" in self.last_words.columns:
                columns.append("w2")
            seenmask = allwords["word"].isin(self.last_words[columns].values.flat)
            if cat[1] == "unseen":
                seenmask = ~seenmask
            candidates[cat] = (
                allwords[wordmask & seenmask].copy().sample(frac=1, random_state=self.rng)
            )
        return candidates

    def _get_valid_cats(self):
        if any(self.last_words["condition"] == "non-word"):
            if "w2" in self.last_words.columns:
                return [self.categories[i] for i in np.random.permutation(4)]
            else:
                valid_qidx = np.array(
                    [self.categories.index(cat) for cat in self.categories if cat[0] == "nonword"]
                )
        else:
            valid_qidx = np.array(
                [self.categories.index(cat) for cat in self.categories if cat[0] == "word"]
            )
        qidx = np.repeat(valid_qidx, 2)
        return [self.categories[qidx[i]] for i in self.rng.permutation(4)]


def shuffle_condition(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Shuffle the condition column of a DataFrame, while keeping the number of each condition the same.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a column named 'condition' that should be shuffled
    rng : np.random.Generator
        Random number generator to use for shuffling

    Returns
    -------
    pd.DataFrame
        The input DataFrame with the 'condition' column shuffled
    """
    df = df.copy()
    grp = df.groupby("condition")
    return grp.sample(frac=1, random_state=rng).reset_index(drop=True)


def split_miniblocks(
    df: pd.DataFrame,
    miniblock_len: int,
    rng: np.random.Generator = np.random.default_rng(),
    dup_extra: bool = False,
) -> pd.DataFrame:
    """
    Split a dataframe into mini-blocks of a given length, and assign a miniblock number to each row.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of stimuli which must have a 'condition' column. The number of elements per
        condition must be divisible by the miniblock length if `dup_extra` is False.
    miniblock_len : int
        Length of each mini-block
    rng : np.random.Generator
        Random state to pass to the sampling function.
    dup_extra : bool, optional
        If the number of stimuli in a condition are not a whole-number multiple of `miniblock_len`,
        whether to duplicate some elements to reach the whole-number. By default False

    Returns
    -------
    pd.DataFrame
        Dataframe with new column `miniblock` containing the miniblock number for each row
    """
    df = df.copy()
    n_mini = len(df) / miniblock_len  # Number of mini-blocks
    if n_mini % 1 != 0:  # make sure we're not dropping stimuli
        if not dup_extra:
            raise ValueError("The miniblock length does not evently divide the number of stimuli.")
        else:
            condrem = {
                cond: len(df.query(f"condition == '{cond}'")) % miniblock_len
                for cond in df["condition"].unique()
            }
            dups = []
            for cond, n in condrem.items():
                dups.append(df.query(f"condition == '{cond}'").sample(n, random_state=rng))
            dupdf = pd.concat(dups, ignore_index=True)
            df = pd.concat([df, dupdf], ignore_index=True)
    sorted_df = df.sort_values("condition")
    miniblocks = np.repeat(np.arange(int(n_mini)), miniblock_len)
    sorted_df["miniblock"] = miniblocks
    return sorted_df


def assign_miniblock_freqs(
    df: pd.DataFrame, freqs: Sequence[float], rng: np.random.Generator = np.random.default_rng()
) -> pd.DataFrame:
    df = df.copy()
    df["w1_freq"] = np.nan
    if "w2" in df.columns:
        df["w2_freq"] = np.nan
    minis = df.groupby("condition").value_counts(["miniblock"])
    # We want to balance the number of F1 and F2 tags in each condition, so we will create a
    # balanced number of F1/F2 blocks and shuffle the indices. This biases our miniblocks to have
    # one more F1 tag (idx 0) if there are an uneven number.
    freqids = []
    for cond in minis.index.levels[0]:
        halfcondmini = len(minis[cond]) // 2
        freqids = np.concatenate(
            [np.zeros(halfcondmini), np.ones(len(minis[cond]) - halfcondmini)]
        )
        freqidxs = rng.permutation(freqids)
        stim_freqidxs = np.repeat(freqidxs, minis[cond]).reshape(-1, 1)
        condfreqs = np.where(stim_freqidxs == 0, freqs, freqs[::-1])
        df.loc[df["condition"] == cond, "w1_freq"] = condfreqs[:, 0]
        if "w2" in df.columns:
            df.loc[df["condition"] == cond, "w2_freq"] = condfreqs[:, 1]
    return df


def prep_miniblocks(
    task: Literal["twoword", "oneword"],
    rng: np.random.Generator,
    df: pd.DataFrame,
    miniblock_len: int,
    freqs: Sequence[float],
) -> pd.DataFrame:
    # Shuffle within conditions and split into miniblocks.
    shuf_df = shuffle_condition(df, rng)
    miniblock_df = split_miniblocks(shuf_df, miniblock_len, rng)
    # Assign frequencies to each miniblock
    outdf = assign_miniblock_freqs(miniblock_df, freqs, rng)
    return outdf


def load_prep_words(
    path_1w: str | Path,
    path_2w: str | Path,
    rng: np.random.Generator,
    miniblock_len: int,
    freqs: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Prepare word stimuli by first shuffling, then assigning frequencies
    twowords = pd.read_csv(path_2w, index_col=0)
    twowords = twowords.sample(frac=1, random_state=rng)
    twowords = prep_miniblocks("twoword", rng, twowords, miniblock_len, freqs)
    onewords = pd.read_csv(path_1w, index_col=0)
    onewords = onewords.sample(frac=1, random_state=rng)
    onewords = prep_miniblocks("oneword", rng, onewords, miniblock_len, freqs)

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
