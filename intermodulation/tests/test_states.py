import numpy as np
import psychopy.core
import psychopy.visual
import pytest

import intermodulation.core.states as cstates
from intermodulation import states


@pytest.fixture
def window():
    return psychopy.visual.Window(
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


@pytest.fixture
def lowfreqs():
    return [4.0, 10.0]


@pytest.fixture
def clock():
    return psychopy.core.Clock()


class TestMarkovState:
    def test_deterministic(self):
        # Str label, float duration
        state = cstates.MarkovState(next="next", dur=1.0)
        assert state.next == "next"
        assert state.dur == 1
        nexstate, dur = state.get_next()
        assert nexstate == "next"
        assert dur == 1
        # Int label, float duration
        state = cstates.MarkovState(next=120, dur=1.0)
        assert state.next == 120
        assert state.dur == 1
        nexstate, dur = state.get_next()
        assert nexstate == 120
        assert dur == 1
        # Str label, callable duration
        durfunc = lambda: 1.0
        state = cstates.MarkovState(next="next", dur=durfunc)
        nexstate, dur = state.get_next()
        assert dur == 1
        assert nexstate == "next"
        # Int label, callable duration
        state = cstates.MarkovState(next=120, dur=durfunc)
        nexstate, dur = state.get_next()
        assert dur == 1
        assert nexstate == 120

    def test_transition(self):
        # Str labels, float duration
        state = cstates.MarkovState(next=["next1", "next2"], dur=1.0, transition=lambda: 0)
        nexstate, dur = state.get_next()
        assert nexstate == "next1"
        assert dur == 1
        # Int labels, float duration
        state = cstates.MarkovState(next=[120, 121], dur=1.0, transition=lambda: 0)
        nexstate, dur = state.get_next()
        assert nexstate == 120
        assert dur == 1
        # Str labels, callable duration
        state = cstates.MarkovState(next=["next1", "next2"], dur=lambda: 1.0, transition=lambda: 0)
        nexstate, dur = state.get_next()
        assert dur == 1
        assert nexstate == "next1"
        # Int labels, callable duration
        state = cstates.MarkovState(next=[120, 121], dur=lambda: 1.0, transition=lambda: 0)
        nexstate, dur = state.get_next()
        assert dur == 1
        assert nexstate == 120

    def test_calls(self):
        call_log = []

        def updatelog(st, t):
            call_log.append((st, t))

        state = cstates.MarkovState(
            next="next",
            dur=1.0,
            start_calls=[updatelog],
            update_calls=[updatelog],
            end_calls=[updatelog],
        )
        state.start_state(0.0)
        assert call_log[0] == (state, 0.0)
        state.update_state(1.0)
        assert len(call_log) == 2
        assert call_log[1] == (state, 1.0)
        state.end_state(2.0)
        assert len(call_log) == 3
        assert call_log[2] == (state, 2.0)

    def test_logitems(self):
        state = cstates.MarkovState(next="next", dur=1.0)
        state.log_onflip = ["test"]
        state.clear_logitems()
        assert state.log_onflip == []
        # see if adding the clear call to the end state func works
        state.log_onflip = ["test"]
        state.end_calls.append(lambda _, __: state.clear_logitems())
        state.end_state(0.0)
        assert state.log_onflip == []


class TestFlickerState:
    def test_flicker_init(self):
        freqs = {"word": 1.0, "shape": 2.0}
        state = cstates.FlickerStimState(
            next="next", dur=1.0, frequencies=freqs, window=None, clock=None
        )
        assert state.frequencies == freqs
        assert state.window is None
        assert state.clock is None
        assert hasattr(state, "precompute_flicker_t")
        assert hasattr(state, "framerate")
        for f in state.start_calls:
            assert callable(f)
        for f in state.update_calls:
            assert callable(f)
        assert isinstance(state.stim, NotImplementedError)

    def test_flicker_create(self):
        freqs = {"word": 1.0, "shape": 2.0}
        state = cstates.FlickerStimState(
            next="next", dur=1.0, frequencies=freqs, window=None, clock=None
        )
        with pytest.raises(NotImplementedError):
            state.start_state(0.0)

    def test_flicker_compute(self):
        freqs = {"words": {"w1": 1.0, "w2": 2.0}, "shape": 0}
        state = cstates.FlickerStimState(
            next="next", dur=1.0, frequencies=freqs, window=None, clock=None
        )
        state.stim = {"words": {"w1": None, "w2": None}, "shape": None}
        state.stimon_t = 0.0
        state._compute_flicker()
        w1_target = np.arange(0, state.precompute_flicker_t, 1 / (2 * freqs["words"]["w1"]))
        w2_target = np.arange(0, state.precompute_flicker_t, 1 / (2 * freqs["words"]["w2"]))
        assert hasattr(state, "target_switches")
        assert np.all(state.target_switches["words"]["w1"] == w1_target)
        assert np.all(state.target_switches["words"]["w2"] == w2_target)
        assert state.target_switches["shape"] is None


class TestTwoWordState:

    def test_twoword_init(self, lowfreqs, window):
        state = states.TwoWordState(next="next", dur=1.0, frequencies=lowfreqs, window=window)
        assert state.window == window
        assert state.words[0] == "test"
        assert state.words[1] == "words"

    def test_twoword_create(self, lowfreqs, window):
        state = states.TwoWordState(next="next", dur=1.0, frequencies=lowfreqs, window=window)
        state.start_state(0.0)
        assert state.stim.words[0].autoDraw
        assert state.stim.words[1].autoDraw
        assert state.stim.shapes["fixdot"].autoDraw
        assert state.stim.words[0].text == "test"
        assert state.stim.words[1].text == "words"
        assert state.stimon_t == 0.0
        assert state.lastflip_t == 0.0
        window.flip()
        window.close()

    def test_twoword_flicker(self):
        lowfreqs = [4.0, 10.0]
        clock = psychopy.core.Clock()
        window = psychopy.visual.Window(
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

        rate = window.getActualFrameRate()
        state = states.TwoWordState(
            next="next",
            dur=1.0,
            frequencies=lowfreqs,
            window=window,
            framerate=rate,
            clock=clock,
        )
        while clock.getTime() < 1.0:
            window.flip()
        clock.reset()
        state.start_state(window.getFutureFlipTime(clock=clock))
        window.flip()
        while clock.getTime() < 1.0 + 1 / rate:
            state.update_state(window.getFutureFlipTime(clock=clock))
            window.flip()
        window.close()
        assert len(state.switches[0]) in (7, 8, 9)
        assert len(state.switches[1]) in (18, 19, 20, 21)
