from collections.abc import Callable, Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Number
from typing import Tuple

import numpy as np
import psychopy.core
import psychopy.visual

import intermodulation.core.stimuli as stimuli
from intermodulation.core import _types
from intermodulation.utils import nested_deepkeys, nested_get, nested_set, parse_calls


@dataclass
class MarkovState(_types.MarkovState):
    """
    Markov state class to allow for both deterministic and probabilistic state transitions,
    computing current state duration when determining the next state. Useful base class, not for
    direct use.

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
    start_calls : list[Tuple[Callable, ...]]
        List of functions to call when the state is started. Each element must be a tuple with the
        callable as the first item. An additional sequence after the callable will be used as
        arguments, and a mapping will be used as kwargs. All functions will be called with the time
        as a first argument. Default is an empty list.
    end_calls : list[Tuple[Callable, ...]]
        List of functions to call when the state is ended. Same structrue as `start_calls`.
        Default is an empty list.
    update_calls : list[Tuple[Callable, ...]]
        List of functions to call when the state is updated. Same structrue as `start_calls`.
        Default is an empty list.
    log_onflip : list[str]
        List of attributes to log when the display is flipped. Default is an empty list.
    """

    next: Sequence[Hashable] | Hashable
    dur: float | Callable
    transition: None | Callable = None
    start_calls: list[Tuple[Callable, ...]] = field(default_factory=list)
    end_calls: list[Tuple[Callable, ...]] = field(default_factory=list)
    update_calls: list[Tuple[Callable, ...]] = field(default_factory=list)
    log_onflip: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.next, (Hashable, Sequence)):
            raise TypeError("Next state must be a hashable or sequence of hashables.")
        if not isinstance(self.dur, (Number, Callable)):
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
                    next = self.next[self.transition(*args, **kwargs)]  # type: ignore
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

    def start_state(self, t):
        """
        Initiate state by calling all functions in `start_calls`.

        Parameters
        ----------
        t : float
            time of state initiation (usually next flip).
        """
        for f, args, kwargs in parse_calls(self.start_calls):
            f(t, *args, **kwargs)

    def update_state(self, t):
        """
        Update state by calling all functions in `update_calls`.

        Parameters
        ----------
        t : float
            time of state update (usually next flip).
        """
        for f, args, kwargs in parse_calls(self.update_calls):
            f(t, *args, **kwargs)

    def end_state(self, t):
        """
        End state by calling all functions in `end_calls`.

        Parameters
        ----------
        t : float
            time of state end (usually next flip).
        """
        for f, args, kwargs in parse_calls(self.end_calls):
            f(t, *args, **kwargs)

    def clear_logitems(self):
        """
        Clear all items marked to be logged on flip.
        """
        self.log_onflip = []


@dataclass
class FlickerStimState(MarkovState):
    frequencies: Mapping[Hashable, Number | Mapping] = field(kw_only=True)
    window: psychopy.visual.Window = field(kw_only=True)
    stim: stimuli.StatefulStim = field(kw_only=True)
    stim_constructor_kwargs: Mapping = field(default_factory=dict)
    clock: psychopy.core.Clock = field(kw_only=True)
    framerate: float = 60.0
    precompute_flicker_t: float = 100.0

    def __post_init__(self):
        self.start_calls.append((self._create_stim, (self.stim_constructor_kwargs,)))
        self.start_calls.append((self._compute_flicker,))
        self.update_calls.append((self._update_stim,))
        self.end_calls.append((self._end_stim,))
        super().__post_init__()

    def _create_stim(self, t, constructor_kwargs):
        if not hasattr(self, "window"):
            raise AttributeError("Window must be set as state attribute before creating stimuli.")
        self.clear_logitems()
        self.stim.start_stim(constructor_kwargs)
        self.stimon_t = t
        self.log_onflip

    def _compute_flicker(self, t):
        if not hasattr(self, "stimon_t"):
            raise AttributeError("Stimulus must be created before computing flicker.")
        allkeys = list(nested_deepkeys(self.stim.construct))
        ts = {}
        for keys in allkeys:
            try:
                if nested_get(self.frequencies, keys) in (None, 0):
                    nested_set(ts, keys, None)
                else:
                    nested_set(
                        ts,
                        keys,
                        np.arange(
                            self.stimon_t,  # type: ignore
                            self.stimon_t + self.precompute_flicker_t,
                            1 / (2 * nested_get(self.frequencies, keys)),  # type: ignore
                        ),
                    )
            except KeyError:
                nested_set(ts, keys, None)

        self.target_switches = ts
        self.target_mask = {}
        for k in allkeys:
            mask = np.ones_like(nested_get(self.target_switches, k), dtype=bool)
            if len(mask.shape) > 0:
                mask[0] = False
            nested_set(self.target_mask, k, mask)

    def _update_stim(self, t):
        if not hasattr(self, "stim"):
            raise AttributeError("Stimuli must be created before updating.")
        self.clear_logitems()
        newstates = deepcopy(self.stim.states)
        for key in nested_deepkeys(self.target_switches):
            keytargets = nested_get(self.target_switches, key)
            keymask = nested_get(self.target_mask, key)
            if keytargets is None:
                continue
            close_enough = np.isclose(
                t, keytargets, rtol=0.0, atol=1 / (2 * self.framerate) - 1e-6
            )
            past_t = t > keytargets
            goodclose = (close_enough & keymask) | (past_t & keymask)
            # breakpoint()
            if np.any(goodclose):
                ts_idx = np.argwhere(goodclose).flatten()[-1]
                keymask[ts_idx] = False
                nested_set(newstates, key, not nested_get(self.stim.states, key))
        self.log_onflip.extend(self.stim.update_stim(newstates))

    def _compute_flicker_frame_count(self, t):
        pass

    def _update_stim_frame_count(self, t):
        pass

    def _end_stim(self, t):
        if not hasattr(self, "stim"):
            raise AttributeError("Stimuli must be created before ending.")
        self.stim.end_stim()
