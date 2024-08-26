from typing import Callable, Hashable, Literal, Mapping, Sequence, Tuple

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
        self.logger.log(self.state_num, "state_start", flip_t)
        self.logger.log(self.state_num, key="state", value=state_key)
        self.logger.log(self.state_num, "next_state", self.next)
        self.logger.log(self.state_num, "target_end", self.t_next)
        self._event_calls(state_key, "start")  # Call any start events

        # Run updates while we wait on the next state
        while (t := self.win.getFutureFlipTime(clock=self.clock)) < self.t_next:
            state.update_state(t)
            for log in state.log_onflip:  # Queue onflip events for time logging
                self.logger.log(self.state_num, log, lazy_time(self.clock))
            self.win.flip()
            self._log_flip(state)  # Process onflip queue and clear
            self._event_calls(state_key, "update")  # Call any update events

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
        # breakpoint()
        if self.current == self.trial_endstate:
            self.logger.log(old_state_num, "trial_end", True)
            self.trial += 1
            self.block_trial += 1
            for call in self.trial_calls:
                if isinstance(call, tuple):
                    call[0](**call[1])
                else:
                    call()
            if self.block_trial == self.K_trials:
                self.logger.log(old_state_num, "block_end", True)
                self.block += 1
                self.block_trial = 0
                for call in self.block_calls:
                    if isinstance(call, tuple):
                        call[0](**call[1])
                    else:
                        call()
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
            self.inc_counters()
        return

    def _event_calls(self, state: Hashable, event: Literal["start", "update", "end"]):
        if state in self.state_calls and event in self.state_calls[state]:
            if isinstance(self.state_calls[state][event], Callable):
                self.state_calls[state][event]()
            else:
                for f, args, kwargs in parse_calls(self.state_calls[state][event]):
                    f(*args, **kwargs)
        return

    def _log_flip(self, state_instance: MarkovState):
        self.logger.log_flip()
        state_instance.clear_logitems()
        return
