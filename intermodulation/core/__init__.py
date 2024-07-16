from typing import Hashable, Mapping

from intermodulation.core.events import ExperimentLog
from intermodulation.core.states import FlickerStimState, MarkovState
from intermodulation.core.stimuli import StatefulStim


class ExperimentManager:
    def __init__(
        self,
        states: Mapping[Hashable, MarkovState],
        log_events: Mapping[Hashable, Mapping[Hashable, str]],
        start: Hashable,
        trial_end: Hashable,
        current: Hashable | None = None,
        N_blocks: int = 1,
    ) -> None:
        pass
