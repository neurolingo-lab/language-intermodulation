"""
General task-related utility functions.
"""

import psychopy.core
import psychopy.visual
import psychopy.constants


def start_stim(
    stim: psychopy.visual.BaseVisualStim,
    frame: int,
):
    """
    Function to start a stimulus and set the necessary attributes for flicker_stim.

    The stimulus object will gain the non-standard
    attributes:
     -`t_start` : global time at which stimulus was started (float)
     -`t_last_flip` : global time at which the stimulus state last changed (float)
     -`frame_start` : frame number at which the stimulus was started (int)
     -`frame_last_flip` : frame number at which the stimulus state last changed (int)
     -`state` : boolean indicating whether the stimulus is currently being displayed (bool)

    Parameters
    ----------
    stim : psychopy.visual.BaseVisualStim
        The stimulus object to start. Note above for non-standard attribute requirements.
    clock : psychopy.core.Clock
        The clock being used as timing. **Must be the same clock used to set t_start and
        t_last_flip attributes.**
    """
    stim.setAutoDraw(True) # Set the stimulus to draw every frame (i.e., to be visible)
    # Set the time at which the stimulus was started (next flip time)
    stim.t_start = None
    stim.win.timeOnFlip(stim, "t_start")
    # Set the time at which the stimulus state last changed (next flip time)
    stim.t_last_switch = None
    stim.win.timeOnFlip(stim, "t_last_switch")
    # Set the frame number at which the stimulus was started
    stim.frame_start = frame
    # Set the frame number at which the stimulus state last changed
    stim.frame_last_flip = frame
    # Set the visible state of the stimulus to true for information (doesn't affect drawing)
    stim.state = True
    # Set the status of the stimulus to started
    stim.status = psychopy.constants.STARTED
    # Create an empty attribute for the stop time so win.timeOnFlip doesn't throw an error
    stim.t_stop = None
    return


def flicker_stim(
    stim: psychopy.visual.BaseVisualStim,
    flicker_frames: float,
    frame: int,
):
    """
    Function to switch stimulus state at a given flicker rate.

    The stimulus object should have the non-standard
    attributes:
     -`t_start` : global time at which stimulus was started (float)
     -`t_last_switch` : global time at which the stimulus state last changed (float)
     -`frame_start` : frame number at which the stimulus was started (int)
     -`frame_last_switch` : frame number at which the stimulus state last changed (int)
     -`state` : boolean indicating whether the stimulus is currently being displayed (bool)


    Parameters
    ----------
    stim : psychopy.visual.BaseVisualStim
        The stimulus object to flicker. Note above for non-standard attribute requirements.
    flicker_frames : int
        Number of frames between each flip of the stimulus state.
    frame: int
        Current frame number.
    win : psychopy.visual.Window
        The window that the stimulus is being drawn to.
    """
    # Calculate how many frames it's been since the stimulus started
    frame_diff = frame - stim.frame_start
    # If the difference is 0 the stimulus just started, so don't flicker. Otherwise if it's
    # been an even multiple of flicker_frames since start, toggle the state.
    if frame_diff > 0 and frame_diff % flicker_frames == 0:
        stim.setAutoDraw(not stim.state)
        stim.state = not stim.state
        stim.win.timeOnFlip(stim, "t_last_switch")  # Make sure to update the last switch time
        stim.frame_last_switch = frame
        return True
    return False


def stop_stim(
    stim: psychopy.visual.BaseVisualStim,
    frame: int,        
):
    """
    Stop a stimulus while storing stop times and frames to the new attributes `t_stop` and `frame_stop`.

    Parameters
    ----------
    stim : psychopy.visual.BaseVisualStim
        Stimulus object to stop drawing
    frame : int
        Current frame number
    """
    stim.setAutoDraw(False)  # Stop drawing the stimulus
    stim.win.timeOnFlip(stim, "t_stop")  # Set the time at which the stimulus was stopped
    stim.frame_stop = frame  # Set the frame number at which the stimulus was stopped
    stim.status = psychopy.constants.FINISHED  # Set the status of the stimulus to finished
    return


def quit_experiment(
        device_manager,
        window,
):
    """
    Function to quit the experiment on keypress of 'q' or 'escape'. Only closes the window,
    but doesn't kill the python process (for debugging and testing).

    Parameters
    ----------
    device_manager : psychopy.hardware.DeviceManager
        The device manager object that contains the ioServer object.
    window : psychopy.visual.Window
        The window object to close.
    """
    keyboard = device_manager.getDevice("defaultKeyboard")
    if keyboard.getKeys(keyList=["q", "escape"]):
        window.close()
        return True
    return False