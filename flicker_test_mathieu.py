from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import psychtoolbox as ptb
from psychopy.visual import ShapeStim, TextStim, Window

if TYPE_CHECKING:
    from numpy.typing import NDArray

# constants
WINDOW_CONFIG: dict[str, int | bool | list[int] | tuple[int, ...] | str] = {
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
TEXT_CONFIG: dict[str, str | float | None] = {
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
WORD_SEP: int = 4  # word separation in degrees
FLICKER_RATES: NDArray[np.float64] = np.array([20.0, 2.0])  # Hz


# initialize components
class Components:
    def __init__(self):
        self.win = Window(**WINDOW_CONFIG)
        self.framerate = self.win.getActualFrameRate()
        self.framerate = 60 if self.framerate is None else self.framerate
        self.word0 = TextStim(
            win=self.win,
            text="Red",
            pos=(-(WORD_SEP / 2), 0),
            **TEXT_CONFIG,
        )
        self.word1 = TextStim(
            win=self.win,
            text="Boat",
            pos=(WORD_SEP / 2, 0),
            **TEXT_CONFIG,
        )
        self.fixation_dot = ShapeStim(
            win=self.win,
            size=(0.05, 0.05),
            vertices="circle",
            anchor="center",
            colorSpace="rgb",
            lineColor="white",
            fillColor="white",
            interpolate=True,
        )
        self.fixation_dot.setAutoDraw(True)
        self.word0.setAutoDraw(True)
        self.word1.setAutoDraw(True)
        self.last_flip = None
        self.win.callOnFlip(self._get_PTB_flip_time)

    def _get_PTB_flip_time(self):
        self.last_flip = ptb.GetSecs()


components = Components()
win = components.win
# determine target timings
win.flip()
targets = dict(
    word0=(1 / FLICKER_RATES[0]) + components.last_flip,
    word1=1 / FLICKER_RATES[1] + components.last_flip,
)

timings = dict(word0=[], word1=[])
start = ptb.GetSecs()  # timer condition to end the loop
while ptb.GetSecs() <= start + 10:
    next_flip = win.getFutureFlipTime(clock="ptb")
    for key, value in targets.items():
        component = getattr(components, key)
        if next_flip >= value:
            component.setAutoDraw(not component.getAutoDraw())
            targets[key] += 1 / FLICKER_RATES[int(key[-1])]
            timings[key].append(components.last_flip)
    win.flip()
    win.callOnFlip(components._get_PTB_flip_time)  # reset on every-flip

df_timings = {key: pd.Series(value).diff().describe() for key, value in timings.items()}
print(pd.concat(df_timings, axis=1).rename(columns={0: "word0", 1: "word1"}))
