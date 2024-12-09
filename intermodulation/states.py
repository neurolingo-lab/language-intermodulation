from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd
import psystate.events as pe
import psystate.states as ps
from byte_triggers._base import BaseTrigger

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
class StartStopTriggerLogMixin:
    trigger: BaseTrigger | None = field(kw_only=True, default=None)
    trigger_val: int | None = field(kw_only=True, default=None)

    def attach_trigger(self):
        if self.trigger is not None:
            if not isinstance(self.trigger_val, int):
                raise ValueError("Must provide an integer trigger value with a trigger")
            itemclass = partial(
                pe.TriggerTimeLogItem, trigger=self.trigger, value=self.trigger_val
            )
        else:
            itemclass = pe.TimeLogItem
        self.loggables = pe.Loggables(
            start=[
                itemclass(
                    name="start",
                    unique=True,
                )
            ],
            end=[
                pe.TimeLogItem(
                    name="end",
                    unique=True,
                ),
            ],
        )


@dataclass
class TwoWordState(ps.FrameFlickerStimState, StartStopTriggerLogMixin):
    stim: ims.TwoWordStim = field(kw_only=True)
    word_list: pd.DataFrame = field(kw_only=True)

    def __post_init__(self):
        super().attach_trigger()
        super().__post_init__()
        self.pair_idx = 0

        # Ignore the initial passed words and use the list
        words = self.word_list.iloc[self.pair_idx]
        self.phrase_cond = words["condition"]
        self.word1 = words["w1"]
        self.word2 = words["w2"]
        self.frequencies["word1"] = words["w1_freq"]
        self.frequencies["word2"] = words["w2_freq"]
        if self.stim.reporting_pix:
            self.update_calls.append(self._set_pixreport)

    def update_words(self):
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
class TwoWordMiniblockState(ps.FrameFlickerStimState, StartStopTriggerLogMixin):
    stim: ims.TwoWordStim = field(kw_only=True)
    stim_dur: float = field(kw_only=True)
    word_list: pd.DataFrame = field(kw_only=True)

    def __post_init__(self):
        super().attach_trigger()
        super().__post_init__()
        self.miniblock_idx = 0
        self.wordset_idx = 0
        self.wordframes = int(np.round(self.stim_dur / (1 / self.framerate)))

        # Ignore the initial passed words and use the list
        self.wordset = self.word_list.query("miniblock == 0")
        self._init_miniblock()
        self.update_calls.insert(1, self.check_word_update)
        self.end_calls.append(self._inc_miniblock)
        if self.stim.reporting_pix:
            self.update_calls.append(self._set_pixreport)

    def check_word_update(self):
        if self.frame_num % self.wordframes == 0 and self.frame_num > 0:
            self._inc_wordidx()
            self.word1 = self.wordset.iloc[self.wordset_idx]["w1"]
            self.word2 = self.wordset.iloc[self.wordset_idx]["w2"]
            changed = self.stim.update_stim({})
            if changed is not None:
                changed = [(*v, self.frame_num) for v in changed]
                self._update_log.extend(changed)

    @property
    def word1(self):
        return self.stim.word1

    @word1.setter
    def word1(self, value):
        self.stim.word1 = value

    @property
    def word2(self):
        return self.stim.word2

    @word2.setter
    def word2(self, value):
        self.stim.word2 = value

    def _inc_wordidx(self):
        if self.wordset_idx == (len(self.wordset) - 1):
            pass
        else:
            self.wordset_idx += 1
        self.condition = self.wordset.iloc[self.wordset_idx]["condition"]

    def _inc_miniblock(self):
        self.wordset_idx = 0
        self.miniblock_idx += 1
        self.wordset = self.word_list.query(f"miniblock == {self.miniblock_idx}")
        self._init_miniblock()

    def _init_miniblock(self):
        initial = self.wordset.iloc[0]
        self.word1 = initial["w1"]
        self.word2 = initial["w2"]
        self.frequencies["word1"] = initial["w1_freq"]
        self.frequencies["word2"] = initial["w2_freq"]
        self.condition = initial["condition"]

    def _set_pixreport(self, *args, **kwargs):
        word_states = (
            bool(self.stim.stim["word1"].opacity),
            bool(self.stim.stim["word2"].opacity),
        )
        self.stim.stim["reporting_pix"].fillColor = REPORT_PIX_VALS[word_states]


@dataclass
class OneWordState(ps.FrameFlickerStimState, StartStopTriggerLogMixin):
    stim: ims.OneWordStim = field(kw_only=True)
    word_list: pd.DataFrame = field(kw_only=True)

    def __post_init__(self):
        super().attach_trigger()
        super().__post_init__()
        self.word_idx = 0

        # Ignore the initial passed words and use the list
        words = self.word_list.iloc[self.word_idx]
        self.word_cond = words["condition"]
        self.stim.word1 = words["w1"]
        self.frequencies["word1"] = words["w1_freq"]
        self.frequencies["reporting_pix"] = words["w1_freq"]

        self.stim_constructor_kwargs = {}

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

        self.frequencies["word1"] = words["w1_freq"]
        if self.stim.reporting_pix:
            self.frequencies["reporting_pix"] = words["w1_freq"]
        return


@dataclass
class OneWordMiniblockState(ps.FrameFlickerStimState, StartStopTriggerLogMixin):
    stim: ims.TwoWordStim = field(kw_only=True)
    stim_dur: float = field(kw_only=True)
    word_list: pd.DataFrame = field(kw_only=True)

    def __post_init__(self):
        super().attach_trigger()
        super().__post_init__()
        self.miniblock_idx = 0
        self.wordset_idx = 0
        self.wordframes = int(np.round(self.stim_dur / (1 / self.framerate)))

        # Ignore the initial passed words and use the list
        self.wordset = self.word_list.query("miniblock == 0")
        self._init_miniblock()
        self.update_calls.insert(1, self.check_word_update)
        self.end_calls.append(self._inc_miniblock)
        if self.stim.reporting_pix:
            self.update_calls.append(self._set_pixreport)

    def check_word_update(self):
        if self.frame_num % self.wordframes == 0 and self.frame_num > 0:
            self._inc_wordidx()
            self.word1 = self.wordset.iloc[self.wordset_idx]["w1"]
            changed = self.stim.update_stim({})
            if changed is not None:
                changed = [(*v, self.frame_num) for v in changed]
                self._update_log.extend(changed)

    @property
    def word1(self):
        return self.stim.word1

    @word1.setter
    def word1(self, value):
        self.stim.word1 = value

    def _inc_wordidx(self):
        if self.wordset_idx == (len(self.wordset) - 1):
            pass
        else:
            self.wordset_idx += 1
        self.condition = self.wordset.iloc[self.wordset_idx]["condition"]

    def _inc_miniblock(self):
        self.wordset_idx = 0
        self.miniblock_idx += 1
        self.wordset = self.word_list.query(f"miniblock == {self.miniblock_idx}")
        self._init_miniblock()

    def _init_miniblock(self):
        initial = self.wordset.iloc[0]
        self.word1 = initial["w1"]
        self.frequencies["word1"] = initial["w1_freq"]
        if self.stim.reporting_pix:
            self.frequencies["reporting_pix"] = initial["w1_freq"]
        self.condition = initial["condition"]

    def _set_pixreport(self):
        pass


@dataclass
class FixationState(ps.StimulusState, StartStopTriggerLogMixin):
    stim: ims.FixationStim = field(kw_only=True)

    def __post_init__(self):
        super().attach_trigger()
        super().__post_init__()


@dataclass
class InterTrialState(ps.MarkovState, StartStopTriggerLogMixin):
    dur: None = field(kw_only=True, init=False, default=None)
    duration_bounds: tuple[float, float] = field(kw_only=True)
    rng: np.random.Generator = field(kw_only=True)

    def __post_init__(self):
        def duration_callable():
            return self.rng.uniform(*self.duration_bounds)

        self.dur = duration_callable
        super().attach_trigger()
        super().__post_init__()


@dataclass
class QueryState(ps.StimulusState, StartStopTriggerLogMixin):
    stim: ims.QueryStim = field(kw_only=True)
    query_tracker: Mapping = field(kw_only=True)
    update_fn: callable = field(kw_only=True)
    query_kwargs: Mapping = field(kw_only=True, default_factory=dict)

    def __post_init__(self):
        super().attach_trigger()
        super().__post_init__()

        self.test_word = None
        self.truth = None
        self.start_calls.insert(0, (self.update_fn, (self.query_tracker, self)))
        self.start_calls.insert(1, self._set_query)
        self.loggables.add(
            "start",
            pe.AttributeLogItem("test_word", True, self, "test_word"),
        )
        self.loggables.add(
            "start",
            pe.AttributeLogItem("truth", True, self, "truth"),
        )

    def _set_query(self):
        self.stim.stim_kwargs["query"]["text"] = f"{self.test_word}?"
