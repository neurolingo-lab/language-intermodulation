"""
General task-related utility functions.
"""
import psychopy.core
import psychopy.visual


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
