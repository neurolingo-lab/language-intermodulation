import operator
from ast import Not
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import reduce

import numpy as np
import psychopy.core
import psychopy.visual
from zmq import has

import intermodulation.events as imevents
import intermodulation.stimuli as imstim
from intermodulation.core.states import FlickerStimState, MarkovState

# TODO: Add logging of stimulus updates


@dataclass
class TwoWordState(MarkovState):
    frequencies: Sequence[float] = field(kw_only=True)
    window: psychopy.visual.Window = field(kw_only=True)
    framerate: float = 60.0
    clock: psychopy.core.Clock = field(default_factory=psychopy.core.Clock)
    text_config: dict = field(default_factory=imstim.TEXT_CONFIG.copy)
    dot_config: dict = field(default_factory=imstim.DOT_CONFIG.copy)
    words: Sequence[str] = ("test", "words")
    word_sep: int = imstim.WORD_SEP

    def __post_init__(self):
        super().__post_init__()
        self.start_calls.append(self._create_stim)
        self.update_calls.append(self._update_stim)
        self.end_calls.append(self._end_stim)
        self.frequencies = np.array(self.frequencies)

    @staticmethod
    def _create_stim(self, t):
        if not hasattr(self, "window"):
            raise AttributeError("Window must be set as state attribute before creating stimuli.")

        self.clear_logitems()
        self.stim = imstim.TwoWordStim(
            self.window, self.words, self.text_config, self.dot_config, self.word_sep
        )
        self.stim.update_stim({"words": {0: True, 1: True}, "shapes": {"fixdot": True}})
        self.stimon_t = t
        self.switches = {0: [], 1: []}
        self.wordstates = {0: [True], 1: [True]}
        self.target_switches = {
            word: np.arange(t, 100, 1 / (2 * self.frequencies[word])) for word in range(2)
        }

        # TODO: Add logging of stimuli creation
        self.window.callOnFlip(self._rec_flip_time, self.clock, 0)
        self.window.callOnFlip(self._rec_flip_time, self.clock, 1)

    def _rec_flip_time(self, clock, word=None):
        self.switches[word].append(clock.getTime())
        # print(f"Recorded {attrib} at {clock.getTime()}")

    @staticmethod
    def _update_stim(self, t):
        if not hasattr(self, "stim"):
            raise AttributeError("Stimuli must be created before updating.")
        self.clear_logitems()
        newstates = self.stim.states.copy()
        for word in range(2):
            swidx = len(self.switches[word])
            next_switcht = self.target_switches[word][swidx]
            if t - next_switcht > -2 / self.framerate:
                newstates["words"][word] = not self.stim.states["words"][word]
                self.window.callOnFlip(self._rec_flip_time, self.clock, word)
                self.wordstates[word].append(newstates["words"][word])
        changed = self.stim.update_stim(newstates)
        # TODO: Add logging of stimulus updates

    @staticmethod
    def _end_stim(self, t):
        if not hasattr(self, "stim"):
            raise AttributeError("Stimuli must be created before ending.")
        self.clear_logitems()
        self.stim.remove_stim()
        for word in range(2):
            self.wordstates[word].append(False)
            self.window.callOnFlip(self._rec_flip_time, self.clock, word)


@dataclass
class TwoWordFlicker(FlickerStimState):
    text_config: dict = field(default_factory=imstim.TEXT_CONFIG.copy)
    dot_config: dict = field(default_factory=imstim.DOT_CONFIG.copy)
    words: Sequence[str] = ("test", "words")
    word_sep: int = imstim.WORD_SEP

    def __post_init__(self):
        super().__post_init__()
        self.end_calls.append(self._end_stim)

    @staticmethod
    def _create_stim(self, t):
        if not hasattr(self, "window"):
            raise AttributeError("Window must be set as state attribute before creating stimuli.")

        self.clear_logitems()
        self.stim = imstim.TwoWordStim(
            self.window, self.words, self.text_config, self.dot_config, self.word_sep
        )
        self.stim.update_stim({"words": {0: True, 1: True}, "shapes": {"fixdot": True}})
        self.stimon_t = t
        self.switches = {0: [], 1: []}
        self.wordstates = {0: [True], 1: [True]}
        self.target_switches = {
            word: np.arange(t, 100, 1 / (2 * self.frequencies[word])) for word in range(2)
        }

        # TODO: Add logging of stimuli creation
        self.window.callOnFlip(self._rec_flip_time, self.clock, 0)
        self.window.callOnFlip(self._rec_flip_time, self.clock, 1)

    def _rec_flip_time(self, clock, word=None):
        self.switches[word].append(clock.getTime())
        # print(f"Recorded {attrib} at {clock.getTime()}")

    @staticmethod
    def _update_stim(self, t):
        if not hasattr(self, "stim"):
            raise AttributeError("Stimuli must be created before updating.")
        self.clear_logitems()
        newstates = self.stim.states.copy()
        for word in range(2):
            swidx = len(self.switches[word])
            next_switcht = self.target_switches[word][swidx]
            if t - next_switcht > -2 / self.framerate:
                newstates["words"][word] = not self.stim.states["words"][word]
                self.window.callOnFlip(self._rec_flip_time, self.clock, word)
                self.wordstates[word].append(newstates["words"][word])
        changed = self.stim.update_stim(newstates)
        # TODO: Add logging of stimulus updates

    @staticmethod
    def _end_stim(self, t):
        if not hasattr(self, "stim"):
            raise AttributeError("Stimuli must be created before ending.")
        self.clear_logitems()
        self.stim.remove_stim()
        for word in range(2):
            self.wordstates[word].append(False)
            self.window.callOnFlip(self._rec_flip_time, self.clock, word)


@dataclass
class ExperimentState:
    states: Mapping[Hashable, MarkovState]
    start: Hashable
    trial_end: Hashable
    exp_end: Hashable
    current: Hashable | None = None
    next: Hashable | None = None
    t_start: float = 0.0
    t_next: float = 0.0
    trial: int = 0
    block_trial: int = 0
    block: int = 0
    N_blocks: int = 1

    def __post_init__(self):
        if not isinstance(self.states, Mapping):
            raise TypeError("States must be a mapping of hashable identifiers to MarkovState.")
        if not all(isinstance(i, Hashable) for i in self.states):
            raise TypeError("All keys in `states` must be hashable.")
        if not isinstance(self.start, Hashable):
            raise TypeError("Start state must be a hashable.")
        if self.start not in self.states:
            raise ValueError("Start state must be in `states`.")
        if not isinstance(self.stimuli, Sequence):
            raise TypeError("Stimuli must be a sequence of stimulus values, objects, etc.")
        self.current = self.start
        self.next, dur = self.states[self.start].get_next()
        self.t_next = self.t_start + dur

    def _update_state(self, t):
        self.states[self.current].update_state(t)
        if t >= self.t_next:
            if self.current == self.trial_end:
                self._inc_trial()
            if self.current == self.exp_end:
                return
            self.states[self.current].end_state(t)
            self.current = self.next
            self.next, dur = self.states[self.current].get_next()
            self.t_start = t
            self.t_next = t + dur
            self.states[self.current].start_state(t)

    def _inc_trial(self):
        self.trial += 1
        self.block_trial += 1
        if self.block_trial >= len(self.stimuli):
            self._inc_block()

    def _inc_block(self):
        self.block += 1
        self.block_trial = 0
        if self.block >= self.N_blocks:
            self._end_exp()

    def _end_exp(self):
        self.current = self.exp_end
        self.next = None
        self.t_next = None


class WFTState(ExperimentState):
    clock: psychopy.core.Clock
    rng: np.random.Generator = np.random.default_rng()
    stim_indices: Mapping[Hashable, np.ndarray]

    def __post_init__(self):
        pass
