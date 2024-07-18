from typing import Callable, Hashable, Mapping, Sequence, Tuple

import psychopy.core
import psychopy.visual

from intermodulation.core.events import ExperimentLog
from intermodulation.core.states import FlickerStimState, MarkovState
from intermodulation.core.stimuli import StatefulStim
from intermodulation.utils import lazy_time


class ExperimentController:
    def __init__(
        self,
        states: Mapping[Hashable, MarkovState],
        window: psychopy.visual.Window,
        start: Hashable,
        log_events: Mapping[Hashable, Mapping[Hashable, str]],
        logger: ExperimentLog,
        clock: psychopy.core.Clock,
        trial_endstate: Hashable,
        N_blocks: int,
        K_blocktrials: int,
        current: Hashable | None = None,
        trial_calls: Sequence[Callable | Tuple[Callable, Mapping]] = [],
        block_calls: Sequence[Callable | Tuple[Callable, Mapping]] = [],
    ) -> None:
        self.trial = 0
        self.block_trial = 0
        self.block = 0
        self.states = states
        self.win = window
        self.start = start
        self.next = start if current else None
        self.t_next = None
        self.current = current if current else start
        self.log_ev = log_events
        self.logger = logger
        self.clock = clock
        self.trial_endstate = trial_endstate
        self.N_blocks = N_blocks
        self.K_trials = K_blocktrials
        self.trial_calls = trial_calls
        self.block_calls = block_calls

    def _run_state(self, state: MarkovState) -> None:
        self.next, dur = state.get_next()
        self.t_next = self.clock.getTime() + dur
        state.start_state(self.clock.getTime())
        while t := self.clock.getTime() < self.t_next:
            state.update_state(t)
            self.win.flip()
        state.end_state(self.clock.getTime())

    def _update_trial(self) -> None:
        if self.state == self.trial_endstate:
            self.trial += 1
            self.block_trial += 1
            for call in self.trial_calls:
                if isinstance(call, tuple):
                    call[0](**call[1])
                else:
                    call()
            if self.block_trial == self.K_trials:
                self.block += 1
                self.block_trial = 0
                for call in self.block_calls:
                    if isinstance(call, tuple):
                        call[0](**call[1])
                    else:
                        call()
                if self.block == self.N_blocks:
                    self.next = None
        return
