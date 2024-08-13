from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Number
from typing import Hashable

import numpy as np
import psychopy.core
import psychopy.visual

import intermodulation.core.states as imcs
import intermodulation.core.stimuli as imcstim
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


@dataclass
class TwoWordState(imcs.FlickerStimState):
    stim: ims.TwoWordStim = field(kw_only=True)
    stim_constructor_kwargs: Mapping = field(init=False, kw_only=True)

    def __post_init__(self):
        self.stim_constructor_kwargs = {}
        super().__post_init__()


@dataclass
class FixationState(imcs.FlickerStimState):
    stim: ims.FixationStim = field(kw_only=True, default=ims.FixationStim)
    dot_kwargs: Mapping = field(kw_only=True, default=DOT_DEFAULT.copy)
    frequencies: Mapping[Hashable, Number | Mapping] = field(
        init=False, kw_only=True, default_factory={"dot": None}.copy
    )

    def __post_init__(self):
        self.stim_constructor_kwargs = {"fixation": self.dot_kwargs}
        self.frequencies = {"fixation": None}
        self.stim = imcstim.StatefulStim(self.window, {"fixation": psychopy.visual.ShapeStim})
        super().__post_init__()


class InterTrialState(imcs.MarkovState):
    def __init__(self, next, duration_bounds=(1.0, 3.0), rng=np.random.default_rng()):
        def duration_callable():
            return rng.uniform(*duration_bounds)

        super().__init__(next=next, dur=duration_callable)
