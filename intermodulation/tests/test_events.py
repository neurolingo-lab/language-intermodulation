from time import sleep

import psychopy.core
import pytest
import numpy as np

from intermodulation.events import ExperimentLog
from intermodulation.utils import lazy_time


@pytest.fixture
def clock():
    return psychopy.core.Clock()


class TestExpLog:
    def test_explog_init(self):
        log = ExperimentLog()
        assert log.trials == {}
        assert log.continuous == {}
        assert log.log_on_flip == []

    def test_explog_log_noasync(self):
        log = ExperimentLog()
        log.log(1, "trial_number", 1)
        assert 1 in log.trials
        assert log.trials[1]["trial_number"] == 1
        assert len(log.continuous[1]) == 4
        assert all(len(log.continuous[1][k]) == 0 for k in log.continuous[1])
        log.log(1, "word_1", "hello")
        assert log.trials[1]["word_1"] == "hello"
        assert len(log.continuous[1]) == 4
        assert all(len(log.continuous[1][k]) == 0 for k in log.continuous[1])

        log = ExperimentLog()
        with pytest.raises(ValueError):
            log.log(1, "not_a_key", "value")
        log.log(1, "word_1_switches", 1)
        log.log(1, "word_1_switches", 2)
        log.log(1, "word_1_switches", 3)
        assert log.continuous[1]["word_1_switches"] == [1, 2, 3]

    def test_explog_log_async(self, clock):
        log = ExperimentLog()
        baset1 = clock.getTime()
        log.log(1, "word_1_switches", lazy_time(clock))
        sleep(0.5)
        assert len(log.log_on_flip) == 1
        baset2 = clock.getTime()
        log.log(1, "word_2_switches", lazy_time(clock))
        assert len(log.log_on_flip) == 2
        assert 1 in log.continuous
        assert log.continuous[1]["word_1_switches"] == []
        assert log.continuous[1]["word_2_switches"] == []
        sleep(0.5)
        log.log_flip()
        assert len(log.log_on_flip) == 0
        assert len(log.continuous[1]["word_1_switches"]) == 1
        assert len(log.continuous[1]["word_2_switches"]) == 1
        assert np.isclose(log.continuous[1]["word_1_switches"], baset1 + 1, atol=0.005)
        assert np.isclose(log.continuous[1]["word_2_switches"], baset2 + 0.5, atol=0.005)

    def test_explog_multitrial(self, clock):
        log = ExperimentLog()
        baset1 = clock.getTime()
        log.log(1, "word_1_switches", lazy_time(clock))
        log.log(1, "word_2_switches", lazy_time(clock))
        log.log(1, "word_1_states", True)
        log.log(1, "word_2_states", False)
        log.log(1, "word_1", "hello")
        log.log(1, "word_1", "world")
        log.log_flip()
        log.log(2, "word_1_switches", lazy_time(clock))
        log.log(2, "word_2_switches", lazy_time(clock))
        log.log(2, "word_1_states", True)
        log.log(2, "word_2_states", False)
        log.log(2, "word_1", "hello")
        log.log(2, "word_1", "world")
        log.log_flip()
        assert len(log.trials) == 2
        assert len(log.continuous) == 2
        all_tr_keys = log.loggables["per_trial"]
        for k in ["word_1", "word_2", "trial_number"]:
            assert log.trials[1][k] != log.trials[3][k]
            assert log.trials[2][k] != log.trials[3][k]
            all_tr_keys.remove(k)
        for k in all_tr_keys:
            assert np.isnan(log.trials[1][k])
            assert np.isnan(log.trials[2][k])
            assert np.isnan(log.trials[3][k])
