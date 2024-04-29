from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins

plugins.activatePlugins()
prefs.hardware["audioLib"] = "sounddevice"
prefs.hardware["audioLatencyMode"] = "3"
from psychopy import (
    sound,
    gui,
    visual,
    core,
    data,
    event,
    logging,
    clock,
    colors,
    layout,
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
from numpy import (
    sin,
    cos,
    tan,
    log,
    log10,
    pi,
    average,
    sqrt,
    std,
    deg2rad,
    rad2deg,
    linspace,
    asarray,
)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

WORDSTART = 1.0  # seconds
WORDLEN = 6.0  # seconds
FLICKER_RATE = 10  # Hz

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
win.getActualFrameRate(infoMsg="Getting actual frame rate")
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
word1 = visual.TextStim(
    win=win,
    name="word1",
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
word2 = visual.TextStim(
    win=win,
    name="word2",
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

trialComponents = [word1, word2, fixation_dot]
for thisComponent in trialComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, "status"):
        thisComponent.status = NOT_STARTED

t = 0
frameN = -1
flicker_frames = (1 / FLICKER_RATE) / frameDur
flickerN = np.trunc(flicker_frames).astype(int)
print(f"For a flicker rate of {FLICKER_RATE} Hz, the number of frames is {flickerN} at this "
      f"refresh rate. The actual flicker rate will be {1 / (flickerN * frameDur)} Hz. The "
      f"resulting difference is {FLICKER_RATE - 1 / (flickerN * frameDur)} Hz.")

continueRoutine = True
routineForceEnd = False
while continueRoutine and routineTimer.getTime() < (WORDSTART + WORDLEN):
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame

    # *word1* updates

    # if word1 is starting this frame...
    if word1.status == NOT_STARTED and tThisFlip >= WORDSTART - frameTolerance:
        # keep track of start time/frame for later
        word1.frameNStart = frameN  # exact frame index
        word1.tStart = t  # local t and not account for scr refresh
        word1.tStartRefresh = tThisFlipGlobal  # on global time
        word1.tLastRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(word1, "tStartRefresh")  # time at next scr refresh
        win.timeOnFlip(word1, "tLastRefresh")  # time of last refresh
        # update status
        word1.status = STARTED
        word1.state = True
        word1.setAutoDraw(True)

    # if word1 is active this frame...
    if frameN % flickerN == 0 and tThisFlip > WORDSTART + frameDur:
        word1.setAutoDraw(not word1.state)
        word1.state = not word1.state
        print("flipping word1")
        win.timeOnFlip(word1, "tLastRefresh")  # time of last refresh

    # if word1 is stopping this frame...
    if word1.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > word1.tStartRefresh + WORDLEN - frameTolerance:
            # keep track of stop time/frame for later
            word1.tStop = t  # not accounting for scr refresh
            word1.tStopRefresh = tThisFlipGlobal  # on global time
            word1.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            # update status
            word1.status = FINISHED
            word1.setAutoDraw(False)

    # *word2* updates

    # if word2 is starting this frame...
    if word2.status == NOT_STARTED and tThisFlip >= WORDSTART - frameTolerance:
        # keep track of start time/frame for later
        word2.frameNStart = frameN  # exact frame index
        word2.tStart = t  # local t and not account for scr refresh
        word2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(word2, "tStartRefresh")  # time at next scr refresh
        # add timestamp to datafile
        # update status
        word2.status = STARTED
        word2.setAutoDraw(True)

    # if word2 is active this frame...
    if word2.status == STARTED:
        # update params
        pass

    # if word2 is stopping this frame...
    if word2.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > word2.tStartRefresh + WORDLEN - frameTolerance:
            # keep track of stop time/frame for later
            word2.tStop = t  # not accounting for scr refresh
            word2.tStopRefresh = tThisFlipGlobal  # on global time
            word2.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            # update status
            word2.status = FINISHED
            word2.setAutoDraw(False)

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
