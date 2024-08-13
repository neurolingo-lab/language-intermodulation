from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import psychopy.visual

import intermodulation.core.stimuli as ics

# constants
WINDOW_CONFIG = {
    "screen": 0,  # 0 is the primary monitor
    "fullscr": True,
    "winType": "pyglet",
    "allowStencil": False,
    "monitor": "testMonitor",
    "color": [0, 0, 0],
    "colorSpace": "rgb",
    "units": "deg",
    "checkTiming": False,
}
DEBUG_WINDOW_CONFIG = {
    "screen": 0,  # 0 is the primary monitor
    "fullscr": False,
    "winType": "pyglet",
    "allowStencil": False,
    "monitor": "testMonitor",
    "color": [0, 0, 0],
    "colorSpace": "rgb",
    "units": "deg",
    "checkTiming": False,
}
TEXT_CONFIG = {
    "font": "Arial",
    "height": 2.0,
    "wrapWidth": None,
    "ori": 0.0,
    "color": "white",
    "colorSpace": "rgb",
    "opacity": None,
    "languageStyle": "LTR",
    "depth": 0.0,
}
DOT_CONFIG = {
    "size": (0.05, 0.05),
    "vertices": "circle",
    "anchor": "center",
    "colorSpace": "rgb",
    "lineColor": "white",
    "fillColor": "white",
    "interpolate": True,
}
WORD_SEP: int = 4  # word separation in degrees
FLICKER_RATES = np.array([20.0, 2.0])  # Hz


@dataclass
class TwoWordStim(ics.StatefulStim):
    win: psychopy.visual.Window
    word1: str
    word2: str
    separation: float
    fixation_dot: bool = True
    text_config: Mapping = field(default_factory=TEXT_CONFIG.copy)
    dot_config: Mapping = field(default_factory=DOT_CONFIG.copy)

    def __post_init__(self):
        # Set up the stimulus constructors and arguments
        self.word_constructor_kwargs = {
            "words": {
                0: {
                    "text": self.word1,
                    "pos": (-self.separation / 2, 0),
                    "anchorHoriz": "right",
                    "alignText": "right",
                    **self.text_config,
                },
                1: {
                    "text": self.word2,
                    "pos": (self.separation / 2, 0),
                    "anchorHoriz": "left",
                    "alignText": "left",
                    **self.text_config,
                },
            },
            "fixation": self.dot_config,
        }
        constructors = {
            "words": {0: psychopy.visual.TextStim, 1: psychopy.visual.TextStim},
            "fixation": psychopy.visual.ShapeStim,
        }
        super().__init__(self.win, constructors)

    def start_stim(self, **kwargs):
        if (
            hasattr(kwargs, "stim_constructor_kwargs")
            and len(kwargs["stim_constructor_kwargs"].keys()) > 0
        ):
            raise ValueError(
                "Cannot pass stim_constructor_kwargs to TwoWordStim. If you want to "
                "modify the config after instantiation, modify the "
                "`.word_constructor_kwargs` attribute."
            )
        self.word_constructor_kwargs["words"][0]["text"] = self.word1
        self.word_constructor_kwargs["words"][1]["text"] = self.word2
        super().start_stim(self.word_constructor_kwargs)


@dataclass
class FixationStim(ics.StatefulStim):
    win: psychopy.visual.Window
    dot_kwargs: Mapping = field(default_factory=DOT_CONFIG.copy)

    def __post_init__(self):
        self.dot_constructor_kwargs = {"fixation": self.dot_kwargs}
        super().__init__(self.win, {"fixation": psychopy.visual.ShapeStim})

    def start_stim(self, **kwargs):
        if (
            hasattr(kwargs, "stim_constructor_kwargs")
            and len(kwargs["stim_constructor_kwargs"].keys()) > 0
        ):
            raise ValueError(
                "Cannot pass stim_constructor_kwargs to FixationStim. If you want to modify the "
                "config after instantiation, modify the `.dot_constructor_kwargs` attribute."
            )
        super().start_stim(self.dot_constructor_kwargs)
