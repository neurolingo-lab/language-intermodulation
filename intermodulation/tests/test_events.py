from time import sleep

import psychopy.core
import pytest

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
        assert log.continuous == {}
        log.log(1, "word_1", "hello")
        assert log.trials[1]["word_1"] == "hello"
        assert log.continuous == {}

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
        assert log.continuous == {}
        baset2 = clock.getTime()
        log.log(1, "word_2_switches", lazy_time(clock))
        assert len(log.log_on_flip) == 2
        assert log.continuous == {}
        sleep(0.5)
        log.log_flip()
        assert len(log.log_on_flip) == 0
        assert len(log.continuous[1]["word_1_switches"]) == 1
        assert len(log.continuous[1]["word_2_switches"]) == 1
        assert np.isclose(log.continous["word_1_switches"][0], baset1 + 1, atol=0.001)
        assert np.isclose(log.continous["word_2_switches"][0], baset2 + 0.5, atol=0.001)
