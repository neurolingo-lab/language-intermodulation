import asyncio
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import psychtoolbox as ptb
from psychopy.visual import BaseVisualStim, TextStim, Window
from psychopy.visual.shape import ShapeStim
from scipy.interpolate import interp1d

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

# Special types to check
WordsDict = TypedDict("WordsDict", {0: bool, 1: bool})
ShapesDict = TypedDict("ShapesDict", {"dot": bool})
StimUpdate = TypedDict("StimUpdate", {"words": WordsDict, "shapes": ShapesDict})


class TwoWordStim:
    def __init__(
        self,
        window: Window,
        words: tuple[str, str] = ["test", "words"],
        text_config: dict[str, str | float | None] = TEXT_CONFIG,
        dot_config: dict[tuple[float, float], str, str, str, str, bool] = DOT_CONFIG,
        word_sep: int = WORD_SEP,
    ):
        self.win = window
        self.words = {
            0: TextStim(
                win=self.win,
                text=words[0],
                pos=(-(word_sep / 2), 0),
                anchorHoriz="right",
                alignText="right",
                **text_config,
            ),
            1: TextStim(
                win=self.win,
                text=words[1],
                pos=(word_sep / 2, 0),
                anchorHoriz="left",
                alignText="left",
                **text_config,
            ),
        }
        self.shapes = {
            "fixdot": ShapeStim(
                win=self.win,
                **dot_config,
            )
        }
        self.states = {
            "words": {0: False, 1: False},
            "shapes": {"fixdot": False},
        }
        self.shapes["fixdot"].setAutoDraw(False)
        for word in self.words:
            self.words[word].setAutoDraw(False)
        self.start_t = None

    def update_stim(self, states: StimUpdate):
        changed = []
        for key in states:
            attr = getattr(self, key)
            for subkey, state in states[key].items():
                attr[subkey].setAutoDraw(state)
                if self.states[key][subkey] != state:
                    changed.append((key, subkey))
        self.states = states
        return changed

    def remove_stim(self):
        for word in self.words:
            self.words[word].setAutoDraw(False)
        for shape in self.shapes:
            self.shapes[shape].setAutoDraw(False)
