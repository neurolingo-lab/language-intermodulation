import numpy as np
import psychopy.visual

import intermodulation.core as imc

# constants
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
    "checkTiming": True,
}

FLICKER_RATES = np.array([20.0, 2.0])  # Hz
