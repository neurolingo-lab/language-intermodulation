import psychopy.visual as pv
import pytest
from psychopy.visual.shape import ShapeStim

from intermodulation.core.stimuli import StatefulStim
from intermodulation.core.utils import nested_deepkeys, nested_get


@pytest.fixture
def window():
    win = pv.Window(
        screen=0,
        size=(800, 600),
        fullscr=False,
        winType="pyglet",
        allowStencil=False,
        monitor="testMonitor",
        color=[0, 0, 0],
        colorSpace="rgb",
        units="deg",
        checkTiming=False,
    )
    win.flip()
    return win


@pytest.fixture
def constructors():
    return {
        "words": {"w1": pv.TextStim, "w2": pv.TextStim},
        "shapes": {"fixdot": ShapeStim},
    }


@pytest.fixture
def constructor_kwargs():
    return {
        "words": {"w1": {"text": "hello", "height": 2.0}, "w2": {"text": "world", "height": 2.0}},
        "shapes": {"fixdot": {"vertices": "circle", "anchor": "center", "size": (0.05, 0.05)}},
    }


class TestStatefulStim:
    def test_init(self, window, constructors):
        stim = StatefulStim(window, constructors)
        assert stim.win is window
        assert stim.construct == constructors
        assert stim.states == {
            "words": {"w1": False, "w2": False},
            "shapes": {"fixdot": False},
        }
        assert stim.stim == {}
        window.close()

    def test_start_stim(self, window, constructors, constructor_kwargs):
        stim = StatefulStim(window, constructors)
        with pytest.raises(ValueError):
            stim.start_stim({"words": {"w4": {"text": "hello"}}})
        with pytest.raises(ValueError):
            stim.start_stim({"words": {"w1": {"text": "hello", "win": window}}})
        stim.start_stim(constructor_kwargs)
        assert list(nested_deepkeys(stim.stim)) == list(nested_deepkeys(stim.construct))
        assert all(
            [
                isinstance(nested_get(stim.stim, ("words", k)), pv.TextStim)
                for k in constructors["words"]
            ]
        )
        assert all(
            [
                isinstance(nested_get(stim.stim, ("shapes", k)), ShapeStim)
                for k in constructors["shapes"]
            ]
        )
        assert all([nested_get(stim.states, ("words", k)) for k in constructors["words"]])
        window.close()

    def test_update_stim(self, window, constructors, constructor_kwargs):
        stim = StatefulStim(window, constructors)
        stim.start_stim(constructor_kwargs)
        newstates = {  # All states start as True, so this will only change w2
            "words": {"w1": True, "w2": False},
            "shapes": {"fixdot": True},
        }
        changed = stim.update_stim(newstates)
        assert changed == [
            ("words", "w2"),
        ]
        assert all(
            [
                nested_get(stim.states, k) == nested_get(newstates, k)
                for k in nested_deepkeys(newstates)
            ]
        )
        window.close()

    def test_end_stim(self, window, constructors, constructor_kwargs):
        stim = StatefulStim(window, constructors)
        stim.start_stim(constructor_kwargs)
        stim.end_stim()
        assert all([not nested_get(stim.states, k) for k in nested_deepkeys(stim.states)])
        assert len(list(nested_deepkeys(stim.stim))) == 0
        window.close()
