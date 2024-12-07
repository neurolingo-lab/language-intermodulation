from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Number
from typing import Hashable

import numpy as np
import pandas as pd
import psystate.states as ps

import intermodulation.stimuli as ims

DOT_DEFAULT = {
    "size": (0.05, 0.05),
    "vertices": "circle",
    "anchor": "center",
    "colorSpace": "rgb",
    "lineColor": "white",
    "fillColor": "white",
    "interpolate": True,
}
REPORT_PIX_VALS = {
    (False, False): (-1, -1, -1),
    (True, False): (-(1 / 3), -(1 / 3), -(1 / 3)),
    (False, True): (1 / 3, 1 / 3, 1 / 3),
    (True, True): (1, 1, 1),
}


@dataclass
class TwoWordState(ps.FrameFlickerStimState):
    stim: ims.TwoWordStim = field(kw_only=True)
    word_list: pd.DataFrame = field(kw_only=True)

    def __post_init__(self):
        self.pair_idx = 0

        # Ignore the initial passed words and use the list
        words = self.word_list.iloc[self.pair_idx]
        self.phrase_cond = words["condition"]
        self.word1 = words["w1"]
        self.word2 = words["w2"]
        self.frequencies["word1"] = words["w1_freq"]
        self.frequencies["word2"] = words["w2_freq"]

        self.stim_constructor_kwargs = {}

        super().__post_init__()
        if self.stim.reporting_pix:
            self.update_calls.append(self._set_pixreport)
        # Debug calls
        # self.start_calls.append((lambda t: print(self.pair_idx),))
        # self.end_calls.append((lambda t: print(self.pair_idx),))

    def update_words(self, query_state=None):
        if query_state is not None and isinstance(query_state, QueryState):
            query_state.word_list = self.word_list
            query_state.stim_idx = int(self.pair_idx)

        if self.pair_idx == (len(self.word_list) - 1):
            self.pair_idx = 0
        else:
            self.pair_idx += 1

        words = self.word_list.iloc[self.pair_idx]
        self.phrase_cond = words["condition"]
        self.stim.word1 = words["w1"]
        self.stim.word2 = words["w2"]

        self.frequencies["word1"] = words["w1_freq"]
        self.frequencies["word2"] = words["w2_freq"]

        return

    def _set_pixreport(self, *args, **kwargs):
        word_states = (self.stim.states["word1"], self.stim.states["word2"])
        self.stim.stim["reporting_pix"].fillColor = REPORT_PIX_VALS[word_states]


@dataclass
class OneWordState(ps.FrameFlickerStimState):
    stim: ims.OneWordStim = field(kw_only=True)
    word_list: pd.DataFrame = field(kw_only=True)

    def __post_init__(self):
        self.word_idx = 0

        # Ignore the initial passed words and use the list
        words = self.word_list.iloc[self.word_idx]
        self.word_cond = words["condition"]
        self.stim.word1 = words["w1"]
        self.frequencies["words"]["word1"] = words["w1_freq"]
        self.frequencies["reporting_pix"] = words["w1_freq"]

        self.stim_constructor_kwargs = {}
        super().__post_init__()

    def update_word(self, query_state=None):
        if query_state is not None and isinstance(query_state, QueryState):
            query_state.word_list = self.word_list
            query_state.stim_idx = int(self.word_idx)

        if self.word_idx == (len(self.word_list) - 1):
            self.word_idx = 0
        else:
            self.word_idx += 1

        words = self.word_list.iloc[self.word_idx]
        self.word_cond = words["condition"]
        self.stim.word1 = words["w1"]

        self.frequencies["words"]["word1"] = words["w1_freq"]
        if self.stim.reporting_pix:
            self.frequencies["reporting_pix"] = words["w1_freq"]
        return


@dataclass
class FixationState(ps.StimulusState):
    stim: ims.FixationStim = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()


class InterTrialState(ps.MarkovState):
    def __init__(self, next, duration_bounds=(1.0, 3.0), rng=np.random.default_rng()):
        def duration_callable():
            return rng.uniform(*duration_bounds)

        super().__init__(next=next, dur=duration_callable)


@dataclass
class QueryState(ps.FrameFlickerStimState):
    stim: ims.QueryStim = field(kw_only=True)
    query_kwargs: Mapping = field(kw_only=True, default_factory=dict)
    word_list: pd.DataFrame = field(kw_only=True, default=None)
    stim_idx: int = field(kw_only=True, default=0)
    rng: np.random.Generator = field(kw_only=True, default=np.random.default_rng)
    frequencies: Mapping[Hashable, Number | Mapping] = field(
        init=False, kw_only=True, default_factory={"query": None}.copy
    )

    def __post_init__(self):
        self.stim_constructor_kwargs = {}
        self.test_word = None
        super().__post_init__()
        self.start_calls.insert(0, (self._choose_query,))

    def _choose_query(self, t):
        if len(self.word_list) == 0 or self.stim_idx is None:
            raise ValueError(
                "QueryStim requires a word_list and stim_idx to be set. Make sure "
                "it isn't being run before TwoWordState.update_words is called."
            )
        widx = self.word_list.iloc[self.stim_idx].name
        if "w1" in self.word_list.columns and "w2" in self.word_list.columns:
            words = self.word_list.loc[widx][["w1", "w2"]]
            correct_word = self.rng.choice(words)
            other_words = self.word_list.drop(index=widx)[["w1", "w2"]].values.flatten()
        elif "word" in self.word_list.columns:
            correct_word = self.word_list.loc[widx]["word"]
            other_words = self.word_list.drop(index=widx)["word"]
        else:
            raise ValueError(
                "word_list must have a column named 'word' (single word stim) or 'w1' and 'w2' "
                "(two-word stim) to use QueryStim."
            )
        incorr_word = self.rng.choice(other_words)
        self.test_word = self.rng.choice([correct_word, incorr_word])
        self.truth = self.test_word == correct_word
        self.stim.stim_constructor_kwargs["query"]["text"] = f"{self.test_word}?"
