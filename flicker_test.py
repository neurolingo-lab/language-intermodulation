from psychopy import prefs
from psychopy import visual, core, data, logging, hardware, constants
import psychopy.iohub as io
import numpy as np
import pandas as pd
from intermodulation import utils

WORDSTART = 1.0  # seconds
WORDLEN = 6.0  # seconds
ITILEN = 1.0  # seconds
FLICKER_RATES = np.array([20, 2])  # Hz
WINDOW_CONFIG = {
    "screen": 0,  # 0 is the primary monitor
    "fullscr": True,
    # "size": [600, 400],
    "winType": "pyglet",  # Backend for window handling
    "allowStencil": False,  # Allow the use of the stencil buffer (for masks)
    "monitor": "testMonitor",  # Monitor profile to use
    "color": [0, 0, 0],  # Background color (in RGB [-1, 1])
    "colorSpace": "rgb",  # Color space to use
    # "useFBO": True,
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
WORD_SEP = 8  # Degrees

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# store info about the experiment session
expName = "flicker_test"  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    "participant": str(np.random.randint(0, 999999)),
    "session": "001",
    "date|hid": data.getDateStr(),
    "expName|hid": expName,
    "psychopyVersion|hid": "2024.1.3",
}

# --- Setup input devices ---
ioConfig = {}
ioConfig["Keyboard"] = dict(use_keymap="psychopy")
win = visual.Window(**WINDOW_CONFIG)
ioServer = io.launchHubServer(window=win, **ioConfig)
deviceManager.ioServer = ioServer

# create a default keyboard to catch quit commands
if deviceManager.getDevice("defaultKeyboard") is None:
    deviceManager.addDevice(
        deviceClass="psychopy.hardware.keyboard.KeyboardDevice",
        deviceName="defaultKeyboard",
        backend="iohub",
    )
rate = win.getActualFrameRate()
if rate is None:
    rate = 60  # couldn't get a reliable measure so guess
print(f"Frame rate: {rate:0.5f} Hz")
expInfo["frameRate"] = rate
win.hideMessage()

frameTolerance = 0.001  # how close to onset before 'same' frame
# get frame duration from frame rate in expInfo
if "frameRate" in expInfo and expInfo["frameRate"] is not None:
    frameDur = 1.0 / expInfo["frameRate"]
else:
    raise ValueError("Frame rate not measured correctly. Check hardware.")

# Start Code - component code to be run after the window creation

# --- Initialize components for Routine "trial" ---
word0 = visual.TextStim(
    win=win,
    name="word0",
    text="Red",
    pos=(-(WORD_SEP / 2), 0),
    **TEXT_CONFIG,
)
word1 = visual.TextStim(
    win=win,
    name="word1",
    text="Boat",
    pos=(WORD_SEP / 2, 0),
    **TEXT_CONFIG,
)
fixation_dot = visual.ShapeStim(
    win=win,
    name="fixation_dot",
    size=(0.05, 0.05),
    vertices="circle",
    anchor="center",
    colorSpace="rgb",
    lineColor="white",
    fillColor="white",
    interpolate=True,
)
# Create two clocks, the global clock synced to system time and the routine timer to keep track of
# time in the task
globalClock = core.Clock()
ioServer.syncClock(globalClock)
routineTimer = core.Clock()
win.flip()  # Draw the window now that the task timer has started

trialComponents = [word0, word1, fixation_dot]
for thisComponent in trialComponents:
    thisComponent.status = constants.NOT_STARTED

# ------Prepare to start Routine "trial" and print diagnostics about rates and timing------
t = 0
frameN = -1
flicker_frames = (1 / np.array(FLICKER_RATES)) / frameDur
flicker_n = np.round(flicker_frames).astype(int)
start_n = np.ceil(WORDSTART / frameDur).astype(int)
end_n = np.ceil((WORDSTART + WORDLEN) / frameDur).astype(int)
trial_end_n = np.ceil((WORDSTART + WORDLEN + ITILEN) / frameDur).astype(int)
timing_info = {
    "Flicker rates (Hz)": FLICKER_RATES,
    "Flicker duration (s)": 1 / FLICKER_RATES,
    "Start frame": start_n,
    "End frame": end_n,
    "Frames per flicker": flicker_n,
    "Frame duration (s)": frameDur,
    "Actual flicker duration (s)": flicker_n * frameDur,
    "Actual flicker rate (Hz)": 1 / (flicker_n * frameDur),
    "Flicker rate difference (Hz)": FLICKER_RATES - 1 / (flicker_n * frameDur),
    "Flicker rate difference (ms per flick)": (1 / FLICKER_RATES - (flicker_n * frameDur)) * 1000,
}
for key, val in timing_info.items():
    print(f"{key}: {val}")

continueRoutine = True
routineForceEnd = False
actual_framerate = {0: [], 1: []}
globalClock = core.Clock()
ioServer.syncClock(globalClock)
routineTimer = core.Clock()
win.flip()
actual_t = {0: [], 1: []}
while continueRoutine and routineTimer.getTime() < (WORDSTART + WORDLEN):
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)

    # Check for quit (the Esc key)
    if utils.quit_experiment(deviceManager, win):
        break

    # update/draw components on each frame
    # update words
    for i, word in enumerate([word0, word1]):
        word_flicker_n = flicker_n[i]
        match (word.status, frameN):  # Check the status of the word and act accordingly
            case [constants.NOT_STARTED, start_n]:
                utils.start_stim(word, frameN)
            case [constants.STARTED, _] | [constants.FINISHED, _] if frameN > start_n:
                flipped = utils.flicker_stim(word, word_flicker_n, frameN)
                if flipped:
                    actual_t[i].append(word.t_last_switch)
            case [_, end_n]:
                utils.stop_stim(word, frameN)

    match (fixation_dot.status, frameN):  # Start fixation dot or end the trial at trial_end_n
        case [constants.NOT_STARTED, _]:
            utils.start_stim(fixation_dot, frameN)
        case [_, trial_end_n]:
            utils.quit_experiment(deviceManager, win)

    win.flip()

for i in range(2):
    actual_t[i] = pd.Series(actual_t[i]).diff().describe()

print(pd.concat(actual_t, axis=1).rename(columns={0: "word0", 1: "word1"}))
