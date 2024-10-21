from typing import Callable, Hashable, Literal, Mapping, Sequence, Tuple

import numpy as np
import psychopy.core
import psychopy.visual

from intermodulation.core.events import ExperimentLog
from intermodulation.core.states import MarkovState
from intermodulation.utils import lazy_time, parse_calls


class ExperimentController:
    def __init__(
        self,
        states: Mapping[Hashable, MarkovState],
        window: psychopy.visual.Window,
        start: Hashable,
        logger: ExperimentLog,
        clock: psychopy.core.Clock,
        trial_endstate: Hashable,
        N_blocks: int,
        K_blocktrials: int,
        current: Hashable | None = None,
        state_calls: Mapping[Hashable, Sequence[Callable | Tuple[Callable, Mapping]]] = {},
        trial_calls: Sequence[Callable | Tuple[Callable, Mapping]] = [],
        block_calls: Sequence[Callable | Tuple[Callable, Mapping]] = [],
    ) -> None:
        self.state_num = 0
        self.trial = 0
        self.block_trial = 0
        self.block = 0
        self.states = states
        self.win = window
        self.start = start
        self.next = start if current else None
        self.t_next = None
        self.current = current if current else start
        self.logger = logger
        self.clock = clock
        self.trial_endstate = trial_endstate
        self.N_blocks = N_blocks
        self.K_trials = K_blocktrials
        self.state_calls = state_calls
        self.trial_calls = trial_calls
        self.block_calls = block_calls
        self.state = None
        self._paused = False
        self._quitting = False
        if "pause" not in self.states:
            self._add_pause()

    def run_state(self, state: Hashable | None = None) -> None:
        # If the user manually ran a state, use that state otherwise default to current
        if state is None:
            if self.states[self.current] is not self.state:
                raise ValueError("Current state does not match stored state.")
            state = self.state
            state_key = self.current
        else:
            state_key = state
            state = self.states[state_key]
            self.state = state
            self.current = state_key
        self.next, dur = state.get_next()

        # Start the state once we have the next one and duration sorted
        flip_t = self.win.getFutureFlipTime(clock=self.clock)
        state.start_state(flip_t)
        for log in state.log_onflip:  # Queue onflip events for time logging
            self.logger.log(self.state_num, log, lazy_time(self.clock))
        flip_t = self.win.flip()  # Flip display
        self._log_flip(state)  # Process onflip queue and clear
        # Record accurate end time and log start, state, target end
        self.t_next = flip_t + dur
        state_logs = {
            "state_start": flip_t,
            "state": state_key,
            "next_state": self.next,
            "target_end": self.t_next,
            "trial_number": self.trial,
        }
        for key, value in state_logs.items():
            self.logger.log(self.state_num, key, value)

        self._event_calls(state_key, "start")  # Call any start events

        # Run updates while we wait on the next state
        while (t := self.win.getFutureFlipTime(clock=self.clock)) < self.t_next:
            state.update_state(t)
            # print(f"Current time is {t}, we want to end at {self.t_next}")
            for log in state.log_onflip:  # Queue onflip events for time logging
                self.logger.log(self.state_num, log, lazy_time(self.clock))
            self.win.flip()
            self._log_flip(state)  # Process onflip queue and clear
            self._event_calls(state_key, "update")  # Call any update events
            if self._check_pause():
                break
            if self._quitting:
                return

        # End the state
        state.end_state(self.win.getFutureFlipTime(clock=self.clock))
        for log in state.log_onflip:  # onflip events
            self.logger.log(self.state_num, log, lazy_time(self.clock))
        self.logger.log(self.state_num, "state_end", lazy_time(self.clock))
        self.win.flip()  # Flip display
        self._log_flip(state)
        self._event_calls(state_key, "end")  # Call any end events

    def inc_counters(self) -> None:
        old_state_num = self.state_num
        self.state_num += 1
        # import ipdb; ipdb.set_trace()  # noqa
        self.logger.log(old_state_num, "block_number", self.block)
        self.logger.log(old_state_num, "block_trial", self.block_trial)
        if self.current == self.trial_endstate:
            self.logger.log(old_state_num, "trial_end", True)
            self.trial += 1
            self.block_trial += 1
            for f, args, kwargs in parse_calls(self.trial_calls):
                f(*args, **kwargs)
            if self.block_trial == self.K_trials:
                self.logger.log(old_state_num, "block_end", True)
                self.block += 1
                self.block_trial = 0
                for f, args, kwargs in parse_calls(self.block_calls):
                    f(*args, **kwargs)
                if self.block == self.N_blocks:
                    self.next = None
        else:
            self.logger.log(old_state_num, "trial_end", False)
            self.logger.log(old_state_num, "block_end", False)
        return

    def run_experiment(self) -> None:
        if self.state is not None:
            raise ValueError("Experiment already running. How did we get here?")
        self.state = self.states[self.current]
        self.run_state(self.current)
        # breakpoint()
        if self.current == self.start:
            self.inc_counters()
        while self.next:
            self.current = self.next
            self.state = self.states[self.current]
            self.run_state(self.current)
            if self._quitting:
                break
            self.inc_counters()
        return

    def toggle_pause(self):
        self._paused = not self._paused
        return

    def quit(self):
        self._quitting = True
        return

    def add_loggable(
        self,
        state: Hashable,
        event: Literal["start", "update", "end"],
        key: str,
        value: str | float | int | bool | None = None,
        object=None,
        attribute: str | Sequence | None = None,
    ) -> None:
        if value is not None and object is not None:
            raise ValueError("Cannot specify both value and object.")

        if state not in self.state_calls:
            self.state_calls[state] = {}
        if event not in self.state_calls[state]:
            if event not in ["start", "update", "end"]:
                raise ValueError("Event must be one of 'start', 'update', or 'end'.")
            self.state_calls[state][event] = []

        if object is not None:
            if attribute is None:
                raise ValueError("If object is specified, attribute must also be specified.")

            self.state_calls[state][event].append((self._log_attrib, (key, object, attribute)))
        else:
            self.state_calls[state][event].append((self._log_value, (key, value)))
        return

    def _add_pause(self):
        self.states["pause"] = MarkovState(
            next="pause",
            dur=np.inf,
        )
        return

    def _check_pause(self):
        """Return True if we should break out of the current state, either to pause or resume."""
        if self.state is self.states["pause"]:
            if not self._paused:  # We are resuming
                self.next = self._resume
                del self._resume
            return not self._paused
        else:
            if self._paused:  # We are pausing, store the planned next state for resume
                self._resume = self.next
                self.next = "pause"
            return self._paused

    def _log_attrib(self, key, object, attribute):
        if isinstance(attribute, Sequence) and not isinstance(attribute, str):
            currlevel = getattr(object, attribute[0])
            for subkey in attribute[1:]:
                currlevel = getattr(currlevel, subkey)
            self.logger.log(self.state_num, key, currlevel)
        else:
            self.logger.log(self.state_num, key, getattr(object, attribute))
        return

    def _log_value(self, key, value):
        self.logger.log(self.state_num, key, value)
        return

    def _event_calls(self, state: Hashable, event: Literal["start", "update", "end"]):
        if state in self.state_calls and event in self.state_calls[state]:
            if isinstance(self.state_calls[state][event], Callable):
                self.state_calls[state][event]()
            else:
                for f, args, kwargs in parse_calls(self.state_calls[state][event]):
                    f(*args, **kwargs)

        if "all" in self.state_calls and event in self.state_calls["all"]:
            if isinstance(self.state_calls["all"][event], Callable):
                self.state_calls["all"][event]()
            else:
                for f, args, kwargs in parse_calls(self.state_calls["all"][event]):
                    f(*args, **kwargs)
        return

    def _log_flip(self, state_instance: MarkovState):
        self.logger.log_flip()
        state_instance.clear_logitems()
        return
