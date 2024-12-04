from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from intermodulation.core import ExperimentController
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
    TwoWordState,
)
from intermodulation.stimuli import FixationStim, OneWordStim, TwoWordStim

if TYPE_CHECKING:
    from typing import Hashable

    import psychopy.core
    import psychopy.visual
    from byte_triggers import MockTrigger, ParallelPortTrigger

    from intermodulation.core.events import ExperimentLog


@dataclass
class SyntaxConfig:
    rng: np.random.Generator
    clock: psychopy.core.Clock
    window: psychopy.visual.Window
    framerate: int
    worddf: pd.DataFrame
    wordsdf: pd.DataFrame | None = None


class SyntaxStateFactory:
    def __init__(self, config: SyntaxConfig):
        self.rng = config.rng
        self.clock = config.clock
        self.window = config.window
        self.framerate = config.framerate
        self.worddf = config.worddf
        if config.wordsdf is None:
            self.singleword = True
        else:
            self.singleword = False
            self.wordsdf = config.wordsdf

    def iti(self, next, duration_bounds):
        return InterTrialState(
            next=next,
            duration_bounds=duration_bounds,
            rng=self.rng,
        )

    def fixation(self, next, duration):
        return FixationState(
            next=next,
            dur=duration,
            stim=FixationStim(win=self.window, dot_config=DOT_CONFIG),
            window=self.window,
            clock=self.clock,
            framerate=self.framerate,
        )

    def pause(self, next):
        return OneWordState(
            next=next,
            dur=np.inf,
            window=self.window,
            stim=OneWordStim(
                win=self.window,
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
            framerate=self.framerate,
            flicker_handler="frame_count",
        )

    def twoword(self, nexts, word_dur, query_p, clock):
        return TwoWordState(
            next=nexts,
            transition=lambda: self.rng.choice([0, 1], p=[query_p, 1 - query_p]),
            dur=word_dur,
            window=self.window,
            stim=TwoWordStim(
                win=self.window,
                word1="experiment",
                word2="start",
                separation=WORD_SEP,
                fixation_dot=True,
                reporting_pix=REPORT_PIX,
                reporting_pix_size=REPORT_PIX_SIZE,
                text_config=TEXT_CONFIG,
                dot_config=DOT_CONFIG,
            ),
            word_list=self.wordsdf,
            frequencies={"words": {"word1": None, "word2": None}},
            clock=clock,
            framerate=self.framerate,
        )

    def oneword(self, next, word_dur, clock):
        return OneWordState(
            next=next,
            dur=word_dur,
            window=self.window,
            stim=OneWordStim(
                win=self.window,
                word1="experiment",
                text_config=TEXT_CONFIG,
                reporting_pix=REPORT_PIX,
                reporting_pix_size=REPORT_PIX_SIZE,
            ),
            word_list=self.worddf,
            frequencies={"words": {"word1": None}},
            clock=clock,
            framerate=self.framerate,
            flicker_handler="frame_count",
        )


class SyntaxExpFactory:
    def __init__(
        self,
        states: dict,
        logger: ExperimentLog,
        config: SyntaxConfig,
        freqs: np.ndarray,
        trigger: ParallelPortTrigger | MockTrigger | None = None,
        query: Hashable | None = None,
        twoword: Hashable | None = None,
        oneword: Hashable | None = None,
    ):
        if twoword is None and oneword is None:
            raise ValueError("At least one of twoword or oneword must be provided.")
        elif twoword is not None and oneword is not None:
            raise ValueError("Only one of twoword or oneword can be provided.")
        self.twoword = twoword
        self.oneword = oneword
        self.query = query
        self.states = states
        self.logger = logger
        self.config = config
        self.freqs = freqs
        if freqs.shape not in ((2,), (2, 1), (1, 2)):
            raise ValueError("Frequencies must be a 2-element array.")

        if trigger is not None:
            self._add_triggers = True
            self.trigger = trigger
        else:
            self.add_triggers = False

    def make_controller(
        self,
        start,
        N_blocks,
        current=None,
    ):
        controller = ExperimentController(
            states=self.states,
            window=self.config.window,
            start=start,
            logger=self.logger,
            clock=self.config.clock,
            trial_endstate="intertrial",
            N_blocks=N_blocks,
            K_blocktrials=len(self.words),
            current=current,
        )
        if self.twoword is not None:
            controller.add_loggable(
                self.twoword,
                "start",
                "word1",
                object=self.states[self.twoword].stim,
                attribute="word1",
            )
            controller.add_loggable(
                self.twoword,
                "start",
                "word2",
                object=self.states[self.twoword].stim,
                attribute="word2",
            )
            controller.add_loggable(
                self.twoword,
                "start",
                "word1_freq",
                object=self.states[self.twoword],
                attribute=("frequencies", "words", "word1"),
            )
            controller.add_loggable(
                self.twoword,
                "start",
                "word2_freq",
                object=self.states[self.twoword],
                attribute=("frequencies", "words", "word2"),
            )
            controller.add_loggable(
                self.twoword,
                "start",
                "condition",
                object=self.states[self.twoword],
                attribute="phrase_cond",
            )
            controller.add_loggable(
                self.query, "start", "word1", object=self.states[self.query], attribute="test_word"
            )
            controller.add_loggable(
                self.query, "start", "truth", object=self.states[self.query], attribute="truth"
            )

        else:
            controller.add_loggable(
                self.oneword,
                "start",
                "word1",
                object=self.states[self.oneword].stim,
                attribute="word1",
            )
            controller.add_loggable(
                self.oneword,
                "start",
                "word1_freq",
                object=self.states[self.oneword],
                attribute=("frequencies", "words", "word1"),
            )
            controller.add_loggable(
                self.oneword,
                "start",
                "condition",
                object=self.states[self.oneword],
                attribute="word_cond",
            )
        if self.add_triggers:
            self._add_triggers(controller)
        return controller

    def _add_triggers(self, controller):
        for state in self.states:
            if state not in controller.state_calls:
                controller.state_calls[state] = {}
            if "end" not in controller.state_calls[state]:
                controller.state_calls[state]["end"] = []
            if "start" not in controller.state_calls[state]:
                controller.state_calls[state]["start"] = []
            controller.state_calls[state]["end"].append(
                (self.trigger.signal, (TRIGGERS.STATEEND,))
            )
            if state == "pause":
                trig = TRIGGERS.BREAK
            elif state == "intertrial":
                trig = TRIGGERS.ITI
            elif state == "fixation":
                trig = TRIGGERS.FIXATION
            else:
                continue
            controller.state_calls[state]["start"].append((self.trigger.signal, (trig,)))
        controller.trial_calls.append(
            (
                self.trigger.signal,
                (TRIGGERS.TRIALEND,),
            )
        )
        controller.block_calls.append(
            (
                self.trigger.signal,
                (self.TRIGGERS.BLOCKEND,),
            )
        )
        controller.state_calls["pause"]["start"].append(
            (
                self.trigger.signal,
                (TRIGGERS.BREAK,),
            )
        )
        if self.twoword is not None:
            self._add_twoword_trigger(controller)
        if self.oneword is not None:
            self._add_oneword_trigger(controller)
        if self.query is not None:
            self._add_query_trigger(controller)

    def _add_twoword_trigger(self, controller):
        def choose_twoword_trigger(state, freqs, trigger):
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

        controller.state_calls[self.twoword]["start"].append(
            (
                choose_twoword_trigger,
                (self.states[self.twoword], self.freqs, self.trigger),
            )
        )

    def _add_oneword_trigger(self, controller):
        def choose_oneword_trigger(state, freqs, trigger):
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

        controller.state_calls[self.oneword]["start"].append(
            (
                choose_oneword_trigger,
                (self.states[self.oneword], self.freqs, self.trigger),
            )
        )

    def _add_query_trigger(self, controller):
        def choose_query_trigger(state, trigger):
            if state.truth:
                trigger.signal(TRIGGERS.QUERY.TRUE)
            else:
                trigger.signal(TRIGGERS.QUERY.FALSE)
            return

        controller.state_calls[self.query]["start"].append(
            (
                choose_query_trigger,
                (self.states[self.query], self.trigger),
            )
        )
