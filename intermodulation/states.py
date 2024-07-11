from ast import Not
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Union
from functools import reduce
import operator

import numpy as np
import psychopy.core

import intermodulation.events as imevents
import intermodulation.stimuli as imstim


def _nested_iteritems(d):
    for k, v in d.items():
        if isinstance(v, dict):
            for subk, v in _nested_iteritems(v):
                yield (k, *subk), v
        else:
            yield (k), v

def _nested_keys(d):
    for k, v in d.items():
        if isinstance(v, dict):
            for subk in _nested_keys(v):
                yield (k, *subk)
        else:
            yield (k)

def _nested_get(d, keys):
    for key in keys[:-1]:
        d = d[key]
    return d[keys[-1]]

def _nested_set(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value



@dataclass
class MarkovState:
    """
    Markov state class to allow for both deterministic and probabilistic state transitions,
    computing current state duration when determining the next state.

    Attributes
    ----------
    next : Hashable | Sequence[Hashable]
        Next state(s) to transition to. Can us any identifier that is hashable for the state.
        If a single state, the state is deterministic. If a sequence of multiple states,
        the next state is probabilistic and `transitions` must be provided.
    dur : float | Callable
        Duration of the current state. If a single float, the state has a fixed duration. If a
        callable, the state has a variable duration and the callable must return a duration.
    transition : None | Callable
        Probabilities of transitioning to the next state(s). If `next` is a single state, this
        attribute is not needed. If `next` is a sequence of states, this attribute must be a
        callable that returns an index in `next` based on the probabilities of transitioning.
    start_calls : list[Callable]
        List of functions to call when the state is started. Functions must take the state and
        the current time as arguments. Default is an empty list.
    end_calls : list[Callable]
        List of functions to call when the state is ended. Functions must take the state and
        the current time as arguments. Default is an empty list.
    update_calls : list[Callable]
        List of functions to call when the state is updated. Functions must take the state and
        the current time as arguments. Default is an empty list.
    log_onflip : list[str]
        List of attributes to log when the display is flipped. Default is an empty list.
    """

    next: Hashable | Sequence[Hashable]
    dur: float | Callable
    transition: None | Callable = None
    start_calls: list[Callable] = field(default_factory=list)
    end_calls: list[Callable] = field(default_factory=list)
    update_calls: list[Callable] = field(default_factory=list)
    log_onflip: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.next, (Hashable, Sequence)):
            raise TypeError("Next state must be a hashable or sequence of hashables.")
        if not isinstance(self.dur, (float, Callable)):
            raise TypeError(
                "Duration must be a float or callable function that gives an index" " into `next`."
            )
        if isinstance(self.next, Sequence) and not isinstance(self.next, str):
            if not isinstance(self.transition, Callable):
                raise TypeError(
                    "If `next` is a sequence, `transition` must be a callable function"
                    " that gives an index into `next`."
                )
            if not all(isinstance(i, Hashable) for i in self.next):
                raise TypeError("All elements of `next` must be hashable.")
        if isinstance(self.dur, int):
            self.dur = float(self.dur)

    def get_next(self, *args, **kwargs):
        """
        Get next state from this state. Arguments are passed to the transition function if it is
        callable.

        Returns
        -------
        Hashable
            The hashable identifier of the next state.
        float
            The duration of the current state.
        """
        match (self.next, self.transition):
            case [Sequence(), Callable()]:
                try:
                    next = self.next[self.transition(*args, **kwargs)]
                except IndexError:
                    raise ValueError("Transition function must return an index in `next`.")
            case [Hashable(), _]:
                next = self.next
        match self.dur:
            case float():
                dur = self.dur
            case Callable():
                dur = self.dur()
        return next, dur

    def start_state(self, t, *args, **kwargs):
        """
        Initiate state by calling all functions in `start_calls`.

        Parameters
        ----------
        t : float
            time of state initiation (usually next flip).
        """
        for f in self.start_calls:
            f(t, *args, **kwargs)

    def update_state(self, t, *args, **kwargs):
        """
        Update state by calling all functions in `update_calls`.

        Parameters
        ----------
        t : float
            time of state update (usually next flip).
        """
        for f in self.update_calls:
            f(t, *args, **kwargs)

    def end_state(self, t, *args, **kwargs):
        """
        End state by calling all functions in `end_calls`.

        Parameters
        ----------
        t : float
            time of state end (usually next flip).
        """
        for f in self.end_calls:
            f(t, *args, **kwargs)

    def clear_logitems(self):
        """
        Clear all items marked to be logged on flip.
        """
        self.log_onflip = []

@dataclass
class FlickerStimState(MarkovState):
    frequencies: dict[dict] = field(kw_only=True)
    window: psychopy.visual.Window = field(kw_only=True)
    framerate: float = 60.0
    precompute_flicker_t = 100.0
    clock: psychopy.core.Clock = field(default_factory=psychopy.core.Clock)

    def __post_init__(self):
        self.start_calls.append(self._create_stim)
        self.update_call.append(self._update_stim)
        self.frequencies = np.array(self.frequencies)
        self.stim = NotImplementedError("FlickerStimState is a primitive not intended for use on its own. "
                                        "Use a subclass with a specific stimulus type.")
        super().__post_init__()

    def _create_stim(self, t):
        """
        Create stimulus for flicker state. Must be implemented in subclass.

        Must create the internal attributes `stim`, `stimon_t`, `switches`, and `target_switches`.
        `stim` is the stimulus object from intermodulation.stimuli, `stimon_t` is the time the stimulus was started,
        `switches` is a dictionary of lists of times when the stimulus switches states, and `target_switches` is a
        a dictionary of arrays of times when the stimulus should switch states.

        Note that `switches` and `target_switches` can be nested dictionaries to allow for multiple stimuli  to be
        logically grouped. The structure of these nested dicts must match one another, and be consistent with the frequencies
        passed to the state.

        Parameters
        ----------
        t : float
            Time at next flip from the current clock (used at instiation of the state)
        """
        raise self.stim
    
    def _compute_flicker(self):
        if isinstance(self.stim, NotImplementedError):
            raise AttributeError("Stimulus must be created before computing flicker.")
        allkeys = _nested_keys(self.stim)
        ts = {}
        for keys in allkeys:
            try:
                if _nested_get(self.frequencies, keys) in (None, 0):
                    _nested_set(ts, keys, None)
                else:
                    _nested_set(ts, keys, np.arange(self.stimon_t, self.precompute_flicker_t, 1 / (2 * self.frequencies[keys])))
            except KeyError:
                _nested_set(ts, keys, None)
                
        self.target_switches = ts


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
