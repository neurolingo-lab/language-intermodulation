from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psychopy.tools.monitorunittools as mut
import psychopy.visual
import psychopy.visual.rect

import intermodulation.core.stimuli as ics

# constants
WINDOW_CONFIG = {
    "screen": 0,  # 0 is the primary monitor
    "fullscr": True,
    "winType": "pyglet",
    "allowStencil": False,
    "monitor": "testMonitor",
    "color": [-1, -1, -1],
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
    "color": [-1, -1, -1],
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
    reporting_pix: bool = False
    reporting_pix_size: int = 4
    text_config: Mapping = field(default_factory=TEXT_CONFIG.copy)
    dot_config: Mapping = field(default_factory=DOT_CONFIG.copy)

    def __post_init__(self):
        # Set up the stimulus constructors and arguments
        self.stim_constructor_kwargs = {
            "words": {
                "word1": {
                    "text": self.word1,
                    "pos": (-self.separation / 2, 0),
                    "anchorHoriz": "right",
                    "alignText": "right",
                    **self.text_config,
                },
                "word2": {
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
            "words": {"word1": psychopy.visual.TextStim, "word2": psychopy.visual.TextStim},
            "fixation": psychopy.visual.ShapeStim,
        }
        if not self.fixation_dot:
            del self.stim_constructor_kwargs["fixation"]
            del constructors["fixation"]

        if self.reporting_pix:
            if self.reporting_pix_size % 2 != 0:
                raise ValueError("Reporting pixel size must be an even number.")
            upperR_corner = mut.convertToPix(
                pos=np.array((1.0, 1.0)), vertices=np.array((0.0, 0.0)), units="norm", win=self.win
            )
            pix_pos = (
                upperR_corner[0] - self.reporting_pix_size // 2,
                upperR_corner[1] - self.reporting_pix_size // 2,
            )
            self.stim_constructor_kwargs["reporting_pix"] = {
                "pos": pix_pos,
                "height": self.reporting_pix_size,
                "width": self.reporting_pix_size,
                "units": "pix",
                "fillColor": (1, 1, 1),
                "lineWidth": 0,
            }
            constructors["reporting_pix"] = psychopy.visual.rect.Rect

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
        self.stim_constructor_kwargs["words"]["word1"]["text"] = self.word1
        self.stim_constructor_kwargs["words"]["word2"]["text"] = self.word2
        super().start_stim(self.stim_constructor_kwargs)


@dataclass
class OneWordStim(ics.StatefulStim):
    win: psychopy.visual.Window
    word1: str
    text_config: Mapping = field(default_factory=TEXT_CONFIG.copy)

    def __post_init__(self):
        # Set up the stimulus constructors and arguments
        self.stim_constructor_kwargs = {
            "words": {
                "word1": {
                    "text": self.word1,
                    "anchorHoriz": "center",
                    "alignText": "center",
                    **self.text_config,
                },
            },
        }
        constructors = {
            "words": {
                "word1": psychopy.visual.TextStim,
            },
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
        self.stim_constructor_kwargs["words"]["word1"]["text"] = self.word1
        super().start_stim(self.stim_constructor_kwargs)


@dataclass
class FixationStim(ics.StatefulStim):
    win: psychopy.visual.Window
    dot_config: Mapping = field(default_factory=DOT_CONFIG.copy)

    def __post_init__(self):
        self.stim_constructor_kwargs = {"fixation": self.dot_config}
        super().__init__(self.win, {"fixation": psychopy.visual.ShapeStim})

    def start_stim(self, *args, **kwargs):
        if (
            hasattr(kwargs, "stim_constructor_kwargs")
            and len(kwargs["stim_constructor_kwargs"].keys()) > 0
        ):
            raise ValueError(
                "Cannot pass stim_constructor_kwargs to FixationStim. If you want to modify the "
                "config after instantiation, modify the `.stim_constructor_kwargs` attribute."
            )
        super().start_stim(self.stim_constructor_kwargs)


@dataclass
class QueryStim(ics.StatefulStim):
    win: psychopy.visual.Window
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    query_config: Mapping = field(default_factory=TEXT_CONFIG.copy)

    def __post_init__(self):
        self.stim_constructor_kwargs = {
            "query": {
                "text": "UNDEFINED?",
                "pos": (0, 0),
                "anchorHoriz": "center",
                "alignText": "center",
                **self.query_config,
            }
        }
        self.word_list = pd.DataFrame()
        self.stim_idx = None
        super().__init__(self.win, {"query": psychopy.visual.TextStim})

    def start_stim(self, *args, **kwargs):
        if (
            hasattr(kwargs, "stim_constructor_kwargs")
            and len(kwargs["stim_constructor_kwargs"].keys()) > 0
        ):
            raise ValueError(
                "Cannot pass stim_constructor_kwargs to QueryStim. If you want to modify the "
                "config after instantiation, modify the `.stim_constructor_kwargs` attribute."
            )

        super().start_stim(self.stim_constructor_kwargs)
