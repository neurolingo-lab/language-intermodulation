import numpy as np
import psystate.events as pe
import pytest

import intermodulation.freqtag_spec as spec
import intermodulation.states as ims
import intermodulation.stimuli as imst
import intermodulation.utils as imu
from intermodulation.tests.fixtures import (  # noqa: F401
    clock,
    framerate,
    freqs,
    oneworddf,
    rng,
    trigger,
    window,
    word_dur,
    worddf,
)

miniblock_len = 10
transitions = {"query": "iti", "iti": "fixation", "pause": "fixation"}
trans_2w = {"fixation": "twoword", "twoword": "query"}
trans_2w.update(transitions)
trans_1w = {"fixation": "oneword", "oneword": "query"}
trans_1w.update(transitions)


@pytest.fixture
def wordlist(worddf, rng, freqs):  # noqa: F811
    outdf = imu.prep_miniblocks(
        task="twoword",
        rng=np.random.default_rng,
        df=worddf,
        miniblock_len=miniblock_len,
        freqs=freqs,
    )
    return outdf


@pytest.fixture
def wordlist_1w(worddf, rng, freqs):  # noqa: F811
    outdf = imu.prep_miniblocks(
        task="oneword",
        rng=np.random.default_rng,
        df=worddf,
        miniblock_len=miniblock_len,
        freqs=freqs,
    )
    return outdf


@pytest.fixture
def twoword(
    window,  # noqa: F811
    word_dur,  # noqa: F811
    framerate,  # noqa: F811
    clock,  # noqa: F811
    freqs,  # noqa: F811
    wordlist,  # noqa: F811
):
    stim = imst.TwoWordStim(
        win=window,
        word1="word1",
        word2="word2",
        reporting_pix=True,
        reporting_pix_size=8,
        separation=3.0,
        text_config=spec.TEXT_CONFIG.copy(),
    )
    state = ims.TwoWordMiniblockState(
        next=trans_2w["twoword"],
        dur=word_dur * miniblock_len,
        window=window,
        framerate=framerate,
        stim=stim,
        stim_dur=word_dur,
        clock=clock,
        frequencies={"w1": freqs[0], "w2": freqs[1], "fixdot": None},
        word_list=wordlist,
        loggables=pe.Loggables(
            start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
            end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
        ),
        log_updates=True,
        strict_freqs=True,
    )
    return state


@pytest.fixture
def oneword(window, word_dur, framerate, clock, freqs, wordlist_1w):  # noqa: F811
    stim = imst.OneWordStim(
        win=window,
        word1="word",
        reporting_pix=True,
        reporting_pix_size=8,
        text_config=spec.TEXT_CONFIG.copy(),
    )
    state = ims.OneWordMiniblockState(
        next=trans_1w["oneword"],
        dur=word_dur * miniblock_len,
        window=window,
        framerate=framerate,
        stim=stim,
        stim_dur=word_dur,
        clock=clock,
        frequencies={"w1": freqs[0], "fixdot": None},
        word_list=wordlist_1w,
        loggables=pe.Loggables(
            start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
            end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
        ),
        log_updates=True,
        strict_freqs=True,
    )
    return state


@pytest.fixture
def fixstim(window, clock, trigger):  # noqa: F811
    stim = imst.FixationStim(window)
    state = ims.FixationState(
        next=trans_2w["fixation"],
        dur=1.0,
        stim=stim,
        window=window,
        clock=clock,
        loggables=pe.Loggables(
            start=[pe.FunctionLogItem("state_start", True, clock.getTime, timely=True)],
            end=[pe.FunctionLogItem("state_end", True, clock.getTime, timely=True)],
        ),
        trigger=trigger,
        trigger_val=spec.TRIGGERS.FIXATION,
    )
    return state


@pytest.fixture
def fixstim_1w(fixstim):
    fixstim.next = trans_1w["fixation"]
    return fixstim
