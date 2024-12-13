from pathlib import Path

import pytest

TESTING_SEED = 42

f1_fpc = 20  # Frames per cycle of F1
f2_fpc = 24  # Frames per cycle of F2
stim_nframes = 120  # Number of frames to display the words for. Must be divisible by above.

if stim_nframes % f1_fpc != 0 or stim_nframes % f2_fpc != 0:
    raise ValueError("stim_nframes must be divisible by f1_fpc and f2_fpc.")


@pytest.fixture
def window():
    from psychopy.visual import Window

    return Window(
        fullscr=False, color=[-1, -1, -1], colorSpace="rgb", units="deg", checkTiming=False
    )


@pytest.fixture
def clock():
    from psychopy.core import Clock

    return Clock()


@pytest.fixture
def trigger():
    from byte_triggers import MockTrigger

    return MockTrigger()


@pytest.fixture
def worddf():
    import pandas as pd

    df = pd.read_csv(Path(__file__).parents[3] / "two_word_stimuli_nina.csv")
    return df


@pytest.fixture
def oneworddf():
    import pandas as pd

    df = pd.read_csv(Path(__file__).parents[3] / "one_word_stimuli.csv")
    return df


@pytest.fixture
def framerate(window):
    rate = window.getActualFrameRate()
    options = [60, 120, 165, 240]
    nearest = min(options, key=lambda x: abs(x - rate))
    return nearest


@pytest.fixture
def freqs(framerate):
    # not the simplest way to write it, but clearly expresses that we want N cycles per frame
    # times the base frame interval as the period of the frequency.
    return [1 / (f1_fpc / framerate), 1 / (f2_fpc / framerate)]


@pytest.fixture
def word_dur(framerate):
    return stim_nframes / framerate


@pytest.fixture
def rng():
    import numpy as np

    return np.random.default_rng(TESTING_SEED)
