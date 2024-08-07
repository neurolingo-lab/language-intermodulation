import itertools as it

import numpy as np
import psychopy.core
import psychopy.visual
import pytest
from scipy.interpolate import interp1d

import intermodulation.core.states as cstates
import intermodulation.core.stimuli as cstim
from intermodulation.utils import (nested_deepkeys, nested_get,
                                   nested_iteritems, nested_set)


def get_flicker_set():
    basedict = {"words": {"w1": 4.0, "w2": 5.0}, "shapes": {"fixdot": None}}
    basekeys = list(nested_deepkeys(basedict))
    valueset = [4.0, 5.0, 0.0, None]
    fullset = []

    for vals in it.product(valueset, repeat=len(basekeys)):
        subdict = {}
        for key, val in zip(basekeys, vals):
            nested_set(subdict, key, val)
        fullset.append(subdict)
    return fullset


@pytest.fixture
def lowfreqs():
    return {"words": {"w1": 4.0, "w2": 5.0}, "shapes": {"fixdot": None}}


@pytest.fixture
def clock():
    return psychopy.core.Clock()


@pytest.fixture
def stim(window, constructors):
    return cstim.StatefulStim(window=window, constructors=constructors)


@pytest.fixture
def flickerstate(lowfreqs, window, clock, stim, constructor_kwargs):
    return cstates.FlickerStimState(
        next="next",
        dur=1.0,
        frequencies=lowfreqs,
        window=window,
        stim=stim,
        stim_constructor_kwargs=constructor_kwargs,
        clock=clock,
    )


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

        def updatelog(t):
            call_log.append(t)

        state = cstates.MarkovState(
            next="next",
            dur=1.0,
            start_calls=[(updatelog,)],
            update_calls=[(updatelog,)],
            end_calls=[(updatelog,)],
        )
        state.start_state(0.0)
        assert call_log[0] == 0.0
        state.update_state(1.0)
        assert len(call_log) == 2
        assert call_log[1] == 1.0
        state.end_state(2.0)
        assert len(call_log) == 3
        assert call_log[2] == 2.0

    def test_logitems(self):
        state = cstates.MarkovState(next="next", dur=1.0)
        state.log_onflip = ["test"]
        state.clear_logitems()
        assert state.log_onflip == []
        # see if adding the clear call to the end state func works
        state.log_onflip = ["test"]
        state.end_calls.append((lambda t: state.clear_logitems(),))
        state.end_state(0.0)
        assert state.log_onflip == []


class TestFlickerState:
    def test_flicker_init(self, lowfreqs, stim):
        state = cstates.FlickerStimState(
            next="next",
            dur=1.0,
            frequencies=lowfreqs,
            window=None,
            stim=stim,
            clock=None,
        )
        assert state.frequencies == lowfreqs
        assert state.window is None
        assert state.clock is None
        assert hasattr(state, "precompute_flicker_t")
        assert hasattr(state, "framerate")
        for f in state.start_calls:
            assert callable(f[0])
        for f in state.update_calls:
            assert callable(f[0])

    def test_flicker_create(self, flickerstate, constructor_kwargs):
        flickerstate._create_stim(0.0, constructor_kwargs)
        assert flickerstate.stimon_t == 0.0
        assert all(
            [
                nested_get(flickerstate.stim.states, k)
                for k in nested_deepkeys(flickerstate.stim.states)
            ]
        )
        flickerstate.window.close()

    def test_flicker_compute(self, lowfreqs, flickerstate):
        flickerstate.stimon_t = 0.0
        flickerstate._compute_flicker(0.0)
        w1_target = np.arange(
            0, flickerstate.precompute_flicker_t, 1 / (2 * lowfreqs["words"]["w1"])
        )
        w2_target = np.arange(
            0, flickerstate.precompute_flicker_t, 1 / (2 * lowfreqs["words"]["w2"])
        )
        assert hasattr(flickerstate, "target_switches")
        assert np.all(flickerstate.target_switches["words"]["w1"] == w1_target)
        assert np.all(flickerstate.target_switches["words"]["w2"] == w2_target)
        assert flickerstate.target_switches["shapes"]["fixdot"] is None

    def test_flicker_start(self, flickerstate, constructor_kwargs):
        flickerstate.start_state(0.0)
        assert flickerstate.stimon_t == 0.0
        assert all(
            [
                nested_get(flickerstate.stim.states, k)
                for k in nested_deepkeys(flickerstate.stim.states)
            ]
        )
        flickerstate.window.close()

    @pytest.mark.parametrize("lowfreqs", get_flicker_set())
    def test_flicker_update_stim_twowords(
        self, lowfreqs, window, stim, clock, constructor_kwargs
    ):
        flickers = [v for k, v in nested_iteritems(lowfreqs) if v not in (None, 0.0)]
        flickerkeys = [k for k, v in nested_iteritems(lowfreqs) if v not in (None, 0.0)]
        bestframerate = np.prod(flickers)
        test_t = np.arange(0, 1, 1 / bestframerate)
        # Test standard update with some flickering stimuli, but not all
        flickerstate = cstates.FlickerStimState(
            next="next",
            dur=1.0,
            frequencies=lowfreqs,
            window=window,
            stim=stim,
            stim_constructor_kwargs=constructor_kwargs,
            clock=clock,
            framerate=bestframerate,
        )
        flickerstate.start_state(0.0)
        predstates = {}
        for k in flickerkeys:
            tgt = nested_get(flickerstate.target_switches, k)
            draws = np.zeros_like(tgt, dtype=bool)
            draws[::2] = True
            predstate = interp1d(tgt, draws, kind="previous")
            predstates[k] = predstate
        for t in test_t[1:]:
            flickerstate.update_state(t)
            for k, predstate in predstates.items():
                kstate = nested_get(flickerstate.stim.states, k)
                assert any(
                    [  # Ugly but heads off any floating point errors
                        kstate == predstate(t),
                        kstate == predstate(t - 1e-8),
                        kstate == predstate(t + 1e-8),
                    ]
                )

    def test_flicker_end(self, flickerstate, constructor_kwargs):
        flickerstate.start_state(0.0)
        flickerstate.end_state(1.0)
        assert all(
            [
                not nested_get(flickerstate.stim.states, k)
                for k in nested_deepkeys(flickerstate.stim.states)
            ]
        )
        assert len(list(nested_deepkeys(flickerstate.stim.stim))) == 0
        flickerstate.window.close()
                for k in nested_deepkeys(flickerstate.stim.states)
            ]
        )
        assert len(list(nested_deepkeys(flickerstate.stim.stim))) == 0
        flickerstate.window.close()
