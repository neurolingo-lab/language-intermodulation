from dataclasses import dataclass, field

import numpy as np
import psychopy.core
import psychopy.visual
import pytest

from intermodulation.core.events import ExperimentLog
from intermodulation.core.ExperimentController import ExperimentController
from intermodulation.core.states import MarkovState

FLIPDUR = 0.5


@pytest.fixture
def dummy_clock():
    class DummyClock:
        def __init__(self):
            self.t = 0

        def getTime(self):
            return self.t

        def incTime(self, dt):
            self.t += dt

        def setTime(self, t):
            self.t = t

        def reset(self):
            self.t = 0

    return DummyClock()


@pytest.fixture
def dummy_window(dummy_clock):
    class DummyWindow:
        def __init__(self):
            self.flip_count = 0
            self.clock = dummy_clock

        def flip(self):
            self.flip_count += 1
            self.clock.incTime(FLIPDUR)

        def getFutureFlipTime(self, clock):
            return clock.getTime() + FLIPDUR

    return DummyWindow()


@pytest.fixture
def logger():
    loggables = {
        "per_state": [
            "state_number",
            "trial_number",
            "state",
            "next_state",
            "target_end",
            "trial_end",
            "block_end",
            "state_start",
            "state_end",
        ],
        "continuous_per_state": [
            "update_flip",
        ],
    }
    return ExperimentLog(loggables)


@pytest.fixture
def transfunc():
    rng = np.random.default_rng(0)

    def prob():
        return rng.choice([0, 1], p=[0.8, 0.2])

    return prob


@pytest.fixture
def determin_states():
    statemap = {
        "state1": MarkovState("state2", 2),
        "state2": MarkovState("state3", 1),
        "state3": MarkovState("state4", 2),
        "state4": MarkovState("state1", 1),
    }
    return statemap


@pytest.fixture
def prob_states(transfunc):
    statemap = {
        "state1": MarkovState("state2", 2),
        "state2": MarkovState(["state2", "state3"], 1, transition=transfunc),
        "state3": MarkovState("state4", 2),
        "state4": MarkovState("state1", 1),
    }
    return statemap


@pytest.fixture
def loggingstate():
    @dataclass
    class LoggingState(MarkovState):
        name: str = field(kw_only=True)

        def start_state(self, t):
            self.log_onflip.append("state_start")

        def update_state(self, t):
            self.log_onflip.append("update_flip")

        def end_state(self, t):
            self.log_onflip.append("state_end")

    return LoggingState


class TestExpController:
    def test_init(
        self,
        determin_states,
        dummy_window,
        dummy_clock,
    ):
        controller = ExperimentController(
            determin_states,
            dummy_window,
            "state1",
            logger,
            dummy_clock,
            "state4",
            2,
            2,
        )
        assert controller.trial == 0
        assert controller.block_trial == 0
        assert controller.block == 0
        assert controller.states == determin_states
        assert controller.win == dummy_window
        assert controller.start == "state1"
        assert controller.next == None
        assert controller.t_next is None
        assert controller.current == "state1"
        assert controller.logger == logger
        assert controller.clock == dummy_clock
        assert controller.trial_endstate == "state4"
        assert controller.N_blocks == 2
        assert controller.K_trials == 2
        assert controller.trial_calls == []
        assert controller.block_calls == []
        assert controller.state is None

        controller = ExperimentController(
            determin_states,
            dummy_window,
            "state1",
            logger,
            dummy_clock,
            "state4",
            2,
            2,
            current="state2",
        )
        assert controller.state_num == 0
        assert controller.start == "state1"
        assert controller.next == "state1"
        assert controller.current == "state2"

    def test_run_callbacks(self, determin_states, dummy_window, logger):
        real_clock = psychopy.core.Clock()

        call_log = []

        def updatelog(event):
            call_log.append(event)

        controller = ExperimentController(
            determin_states,
            dummy_window,
            "state1",
            logger,
            real_clock,
            "state4",
            2,
            2,
            state_calls={
                "state1": {
                    ev: [
                        (updatelog, (ev,)),
                    ]
                    for ev in ["start", "update", "end"]
                }
            },
        )
        controller.run_state("state1")
        assert len(call_log) >= 3
        assert call_log.index("start") == 0
        assert call_log.index("update") == 1
        assert call_log.index("end") == len(call_log) - 1

    def test_state_run_logger(self, loggingstate, dummy_window, dummy_clock, logger):
        logging_states = {"state1": loggingstate("state1", 2.0, name="state1")}
        controller = ExperimentController(
            logging_states,
            dummy_window,
            "state1",
            logger,
            dummy_clock,
            "state4",
            2,
            2,
        )
        controller.run_state("state1")
        assert logger.states[0]["state_start"] == 0.5
        assert logger.states[0]["state"] == "state1"
        assert len(logger.continuous[0]["update_flip"]) == 3
        assert 1.5 in logger.continuous[0]["update_flip"]
        assert logger.states[0]["state_end"] == 2.5

    def test_exp_run_logger(self, loggingstate, dummy_window, dummy_clock, logger):
        logging_states = {"state1": loggingstate("state1", 2.0, name="state1")}
        controller = ExperimentController(
            logging_states,
            dummy_window,
            "state1",
            logger,
            dummy_clock,
            "state1",
            2,
            2,
        )
        controller.run_experiment()
        assert len(logger.states) == 4
        assert len(logger.continuous[0]["update_flip"]) == 3
        assert logger.states[0]["state_end"] == 2.5
        assert logger.states[1]["state_start"] == 3.0
        assert all([st["state"] == "state1" for _, st in logger.states.items()])
        assert all([st["state_end"] - st["state_start"] == 2.0 for _, st in logger.states.items()])

    def test_run_det_state(self, determin_states, dummy_window, dummy_clock, logger):
        controller = ExperimentController(
            determin_states,
            dummy_window,
            "state1",
            logger,
            dummy_clock,
            "state4",
            2,
            2,
        )
        controller.run_experiment()
        assert controller.next == None
        assert controller.current == "state4"
        assert controller.logger.states[0]["state_start"] == 0.5
        assert controller.logger.states[0]["state_end"] == 2.5
        statenums = [int(tr["state"][-1]) for _, tr in logger.states.items()]
        assert statenums == [1, 2, 3, 4] * 4
        durs = [tr["state_end"] - tr["state_start"] for _, tr in logger.states.items()]
        assert durs == [2.0, 1.0] * 8

        # Clock runs in 0.5s increments, so there should be 4 flips per state, 4 states per trial,
        # and 4 trials in the experiment. Therefore 64 flips.
        assert controller.win.flip_count == 4 * 4 * 4
