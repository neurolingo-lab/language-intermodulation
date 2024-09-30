import pandas as pd
from psychopy import prefs
from psychopy import plugins

plugins.activatePlugins()
prefs.hardware["audioLib"] = "ptb"
prefs.hardware["audioLatencyMode"] = "3"
from psychopy import (
    visual,
    core,
    data,
    logging,
    hardware,
)
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED,
    STARTED,
    PLAYING,
    PAUSED,
    STOPPED,
    FINISHED,
    PRESSED,
    RELEASED,
    FOREVER,
    priority,
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy.random import randint

import psychopy.iohub as io
from psychopy.hardware import keyboard

WORDSTART = 1.0  # seconds
WORDLEN = 6.0  # seconds
FLICKER_RATES = [20, 2]  # Hz

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# store info about the experiment session
psychopyVersion = "2024.1.3"
expName = "test"  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    "participant": f"{randint(0, 999999):06.0f}",
    "session": "001",
    "date|hid": data.getDateStr(),
    "expName|hid": expName,
    "psychopyVersion|hid": psychopyVersion,
}

# start off with values from experiment settings
# if in pilot mode, apply overrides according to preferences
# force windowed mode
if prefs.piloting["forceWindowed"]:
    _fullScr = False
    # set window size
    _winSize = prefs.piloting["forcedWindowSize"]
# _winSize = [3440, 1440]
# _fullScr = True
# override logging level
_loggingLevel = logging.getLevel(prefs.piloting["pilotLoggingLevel"])

# --- Setup input devices ---
ioConfig = {}

# Setup iohub keyboard
ioConfig["Keyboard"] = dict(use_keymap="psychopy")

win = visual.Window(
    size=_winSize,
    fullscr=_fullScr,
    screen=0,
    winType="pyglet",
    allowStencil=False,
    monitor="testMonitor",
    color=[0, 0, 0],
    colorSpace="rgb",
    blendMode="avg",
    useFBO=True,
    units="deg",
    checkTiming=True,
)

ioSession = "1"
if "session" in expInfo:
    ioSession = str(expInfo["session"])
ioServer = io.launchHubServer(window=win, **ioConfig)
# store ioServer object in the device manager
deviceManager.ioServer = ioServer

# create a default keyboard (e.g. to check for escape)
if deviceManager.getDevice("defaultKeyboard") is None:
    deviceManager.addDevice(
        deviceClass="keyboard", deviceName="defaultKeyboard", backend="iohub"
    )
expInfo["frameRate"] = win._monitorFrameRate
print(expInfo["frameRate"])
win.hideMessage()

frameTolerance = 0.001  # how close to onset before 'same' frame
endExpNow = False  # flag for 'escape' or other condition => quit the exp
# get frame duration from frame rate in expInfo
if "frameRate" in expInfo and expInfo["frameRate"] is not None:
    frameDur = 1.0 / round(expInfo["frameRate"])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Start Code - component code to be run after the window creation

# --- Initialize components for Routine "trial" ---
word0 = visual.TextStim(
    win=win,
    name="word0",
    text="Red",
    font="Arial",
    pos=(-2, 0),
    height=2.0,
    wrapWidth=None,
    ori=0.0,
    color="white",
    colorSpace="rgb",
    opacity=None,
    languageStyle="LTR",
    depth=0.0,
)
word1 = visual.TextStim(
    win=win,
    name="word1",
    text="Boat",
    font="Arial",
    units="deg",
    pos=(2, 0),
    height=2.0,
    wrapWidth=None,
    ori=0.0,
    color="white",
    colorSpace="rgb",
    opacity=None,
    languageStyle="LTR",
    depth=-1.0,
)
fixation_dot = visual.ShapeStim(
    win=win,
    name="fixation_dot",
    size=(0.05, 0.05),
    vertices="circle",
    ori=0.0,
    pos=(0, 0),
    anchor="center",
    lineWidth=1.0,
    colorSpace="rgb",
    lineColor="white",
    fillColor="white",
    opacity=None,
    depth=-2.0,
    interpolate=True,
)

globalClock = core.Clock()
ioServer.syncClock(globalClock)
routineTimer = core.Clock()
win.flip()

trialComponents = [word0, word1, fixation_dot]
for thisComponent in trialComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, "status"):
        thisComponent.status = NOT_STARTED

t = 0
frameN = -1
flicker_frames = (1 / np.array(FLICKER_RATES)) / frameDur
flickerNarr = np.trunc(flicker_frames).astype(int)
print(f"For a flicker rate of {FLICKER_RATES} Hz, the number of frames is {flickerNarr} at this "
      f"refresh rate. The actual flicker rate will be {1 / (flickerNarr * frameDur)} Hz. The "
      f"resulting difference is {FLICKER_RATES - 1 / (flickerNarr * frameDur)} Hz.")

continueRoutine = True
routineForceEnd = False
actual_framerate = {0: [], 1: []}
globalClock = core.Clock()
ioServer.syncClock(globalClock)
routineTimer = core.Clock()
win.flip()
while continueRoutine and routineTimer.getTime() < (WORDSTART + WORDLEN):
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame

    # *word0* updates
    for i, word in enumerate([word0, word1]):
        flickerN = flickerNarr[i]
        # if word0 is starting this frame...
        if word.status == NOT_STARTED and tThisFlip >= WORDSTART - frameTolerance:
            # keep track of start time/frame for later
            word.frameNStart = frameN  # exact frame index
            word.tStart = t  # local t and not account for scr refresh
            word.tStartRefresh = tThisFlipGlobal  # on global time
            word.tLastRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(word, "tStartRefresh")  # time at next scr refresh
            win.timeOnFlip(word, "tLastRefresh")  # time of last refresh
            # update status
            word.status = STARTED
            word.state = True
            word.setAutoDraw(True)

        # if word0 is active this frame...
        if frameN % flickerN == 0 and tThisFlip > WORDSTART + frameDur:
            word.setAutoDraw(not word.state)
            word.state = not word.state
            if frameN - word.frameNStart >= flickerN:
                actual_framerate[i].append(1 / (tThisFlip - word.tLastRefresh))
            win.timeOnFlip(word, "tLastRefresh")  # time of last refresh

        # if word is stopping this frame...
        if word.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > word.tStartRefresh + WORDLEN - frameTolerance:
                # keep track of stop time/frame for later
                word.tStop = t  # not accounting for scr refresh
                word.tStopRefresh = tThisFlipGlobal  # on global time
                word.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                # update status
                word.status = FINISHED
                word.setAutoDraw(False)

    # *fixation_dot* updates

    # if fixation_dot is starting this frame...
    if fixation_dot.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        fixation_dot.frameNStart = frameN  # exact frame index
        fixation_dot.tStart = t  # local t and not account for scr refresh
        fixation_dot.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(fixation_dot, "tStartRefresh")  # time at next scr refresh
        # add timestamp to datafile
        # update status
        fixation_dot.status = STARTED
        fixation_dot.setAutoDraw(True)

    # if fixation_dot is stopping this frame...
    if fixation_dot.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > fixation_dot.tStartRefresh + (WORDLEN + WORDSTART) - frameTolerance:
            # keep track of stop time/frame for later
            fixation_dot.tStop = t  # not accounting for scr refresh
            fixation_dot.tStopRefresh = tThisFlipGlobal  # on global time
            fixation_dot.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            # update status
            fixation_dot.status = FINISHED
            fixation_dot.setAutoDraw(False)

    # check if all components have finished
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    win.flip()

actualfr_word0 = pd.Series(actual_framerate[0][:-1], name="word0")
actualfr_word1 = pd.Series(actual_framerate[1][:-1], name="word1")
print(actualfr_word0.describe())
print(actualfr_word1.describe())

