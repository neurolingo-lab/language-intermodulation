import numpy as np
import psychopy.core
import psychopy.monitors
import psychopy.visual
from psyquartz import Clock

# constants
RANDOM_SEED = 42  # CHANGE IF NOT DEBUGGING!! SET TO NONE FOR RANDOM SEED
rng = np.random.default_rng(RANDOM_SEED)

REPORT_PIX_SIZES = np.arange(2, 50, 2)
REPORT_PIX_STEPS = 10
TIME_PER_PIX = 1.0  # seconds

# Detailed display parameters
DISPLAY_RES = (1280, 720)
DISPLAY_DISTANCE = 120  # cm
DISPLAY_WIDTH = 55 * (1280 / 1920)  # cm
DISPLAY_HEIGHT = 30.5 * (720 / 1080)  # cm
FOVEAL_ANGLE = 5.0  # degrees

WINDOW_CONFIG = {
    "screen": 0,  # 0 is the primary monitor
    "fullscr": True,
    "winType": "pyglet",
    "allowStencil": False,
    "monitor": "testMonitor",
    "color": [-1, -1, -1],
    "colorSpace": "rgb",
    "units": "pix",
    "checkTiming": False,
}

propixx = psychopy.monitors.Monitor(name="propixx", width=DISPLAY_WIDTH, distance=DISPLAY_DISTANCE)
propixx.setSizePix((1280, 720))
propixx.save()
WINDOW_CONFIG["monitor"] = "propixx"

window = psychopy.visual.Window(**WINDOW_CONFIG)
clock = Clock()

colorsteps = np.linspace(-1, 1, REPORT_PIX_STEPS)

rect = psychopy.visual.rect.Rect(
    window, height=REPORT_PIX_SIZES[-1], width=REPORT_PIX_SIZES[-1], fillColor=(1, 1, 1)
)
rect.draw()
window.flip()

start_t = clock.getTime()
while clock.getTime() < start_t + 30:
    pass

for size in REPORT_PIX_SIZES:
    for color in colorsteps:
        target_t = clock.getTime() + TIME_PER_PIX
        while clock.getTime() < target_t:
            rect.height = size
            rect.width = size
            rect.fillColor = (color, color, color)
            rect.draw()
            window.flip()
