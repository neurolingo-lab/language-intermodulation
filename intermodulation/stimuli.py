from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import psychopy.tools.monitorunittools as mut
import psychopy.visual.rect
import psychopy.visual.shape
import psystate.stimuli as pst

from intermodulation.freqtag_spec import DOT_CONFIG, TEXT_CONFIG

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

@dataclass
class TwoWordStim(pst.StatefulStim):
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
        stim_kwargs = {
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
            "fixdot": self.dot_config,
        }
        constructors = {
            "word1": psychopy.visual.TextStim,
            "word2": psychopy.visual.TextStim,
            "fixdot": psychopy.visual.shape.ShapeStim,
        }
        if not self.fixation_dot:
            del stim_kwargs["fixdot"]
            del constructors["fixdot"]

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
            stim_kwargs["reporting_pix"] = {
                "pos": pix_pos,
                "height": self.reporting_pix_size,
                "width": self.reporting_pix_size,
                "units": "pix",
                "fillColor": (1, 1, 1),
                "lineWidth": 0,
            }
            constructors["reporting_pix"] = psychopy.visual.rect.Rect

        super().__init__(self.win, constructors, stim_kwargs)

    def start_stim(self):
        self.stim_kwargs["word1"]["text"] = self.word1
        self.stim_kwargs["word2"]["text"] = self.word2
        super().start_stim()

    def update_stim(self, kwargs):
        match len(self.stim), kwargs:
            case 0, _:
                raise ValueError("Stimulus not started.")
            case _, {"word1": {"text": _}, "word2": {"text": _}}:
                pass
            case _, {}:
                if "word1" not in kwargs:
                    kwargs["word1"] = {}
                if "word2" not in kwargs:
                    kwargs["word2"] = {}
                if self.stim["word1"].text != self.word1:
                    kwargs["word1"]["text"] = self.word1
                if self.stim["word2"].text != self.word2:
                    kwargs["word2"]["text"] = self.word2
        return super().update_stim(kwargs)


@dataclass
class OneWordStim(pst.StatefulStim):
    win: psychopy.visual.Window
    word1: str
    reporting_pix: bool = False
    reporting_pix_size: int = 4
    text_config: Mapping = field(default_factory=TEXT_CONFIG.copy)

    def __post_init__(self):
        # Set up the stimulus constructors and arguments
        stim_kwargs = {
            "word1": {
                "text": self.word1,
                "anchorHoriz": "center",
                "alignText": "center",
                **self.text_config,
            },
        }
        constructors = {
            "word1": psychopy.visual.TextStim,
        }
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
            stim_kwargs["reporting_pix"] = {
                "pos": pix_pos,
                "height": self.reporting_pix_size,
                "width": self.reporting_pix_size,
                "units": "pix",
                "fillColor": (1, 1, 1),
                "lineWidth": 0,
            }
            constructors["reporting_pix"] = psychopy.visual.rect.Rect

        super().__init__(self.win, constructors, stim_kwargs)

    def start_stim(self):
        self.stim_kwargs["word1"]["text"] = self.word1
        super().start_stim()

    def update_stim(self, kwargs):
        match len(self.stim), kwargs:
            case 0, _:
                raise ValueError("Stimulus not started.")
            case _, {"word1": {"text": _}}:
                pass
            case _, {}:
                if "word1" not in kwargs:
                    kwargs["word1"] = {}
                if self.stim["word1"].text != self.word1:
                    kwargs["word1"]["text"] = self.word1
        return super().update_stim(kwargs)


class FixationStim(pst.StatefulStim):
    def __init__(self, win: psychopy.visual.Window, dot_config: Mapping = DOT_CONFIG):
        constructors = {"fixation": psychopy.visual.shape.ShapeStim}
        stim_kwargs = {"fixation": dot_config}
        super().__init__(win, constructors, stim_kwargs)


class QueryStim(pst.StatefulStim):
    def __init__(
        self,
        win: psychopy.visual.Window,
        query_config: Mapping = TEXT_CONFIG,
    ):
        stim_kwargs = {
            "query": {
                "text": "UNDEFINED?",
                "pos": (0, 0),
                "anchorHoriz": "center",
                "alignText": "center",
                **query_config,
            }
        }
        super().__init__(win, {"query": psychopy.visual.TextStim}, stim_kwargs)
