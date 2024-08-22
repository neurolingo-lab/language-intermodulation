import psychopy.visual
from byte_triggers import ParallelPortTrigger
from psychopy.visual.rect import Rect
from psychopy.core import Clock

# constants
WINDOW_CONFIG = {
    "screen": 0,  # 0 is the primary monitor
    "fullscr": True,
    "winType": "pyglet",
    "allowStencil": False,
    "monitor": "testMonitor",
    "color": [-1, -1, -1],
    "colorSpace": "rgb",
    "units": "pix",
    "checkTiming": True,
    "waitBlanking": False,
}

trigger = ParallelPortTrigger("/dev/parport0")
window = psychopy.visual.Window(**WINDOW_CONFIG)
framerate = window.getActualFrameRate()
print(framerate)
clock = Clock()

# Run and save the frame rate test with the addition of a flickering whole-screen white box
# which will produce a framerate / 2 Hz flicker
screenbox = Rect(
    window, units="pix", size=window.size, fillColor=[1, 1, 1], opacity=1.0
)
window.flip()
i = 0  # Frame counter
clock.reset()
while i < 2400:
    if i % 2 == 0:
        trigger.signal(1)
    if i % 2 == 0:
        screenbox.draw()
    window.flip()
    i += 1
print(f"{i} frames in {clock.getTime()} s")
