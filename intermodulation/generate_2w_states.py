import numpy as np
import pandas as pd

from intermodulation.freqtag_spec import (
    DOT_CONFIG,
    REPORT_PIX,
    REPORT_PIX_SIZE,
    TEXT_CONFIG,
    WORD_SEP,
)
from intermodulation.states import (
    FixationState,
    InterTrialState,
    OneWordState,
    QueryState,
    TwoWordState,
)
from intermodulation.stimuli import FixationStim, OneWordStim, QueryStim, TwoWordStim


def generate_2w_states(
    rng,
    FIXATION_DURATION,
    WORD_DURATION,
    QUERY_DURATION,
    ITI_BOUNDS,
    QUERY_P,
    clock,
    window,
    framerate,
    wordsdf,
):
    states_2word = {
        "pause": OneWordState(
            next="fixation",
            dur=np.inf,
            window=window,
            stim=OneWordStim(
                win=window,
                word1="Time for a break!",
                text_config=TEXT_CONFIG,
                reporting_pix=REPORT_PIX,
                reporting_pix_size=REPORT_PIX_SIZE,
            ),
            word_list=pd.DataFrame(
                {
                    "w1": ["Time for a break!"],
                    "w2": [
                        None,
                    ],
                    "w1_freq": [0],
                    "condition": ["pause"],
                }
            ),
            frequencies={"words": {"word1": None}},
            clock=clock,
            framerate=framerate,
            flicker_handler="frame_count",
        ),
        "intertrial": InterTrialState(
            next="fixation",
            duration_bounds=ITI_BOUNDS,
            rng=rng,
        ),
        "fixation": FixationState(
            next="words",
            dur=FIXATION_DURATION,
            stim=FixationStim(win=window, dot_config=DOT_CONFIG),
            window=window,
            clock=clock,
            framerate=framerate,
        ),
        "query": QueryState(
            next="intertrial",
            dur=QUERY_DURATION,
            stim=QueryStim(win=window, rng=rng, query_config=TEXT_CONFIG),
            window=window,
            clock=clock,
            framerate=framerate,
            rng=rng,
        ),
        "words": TwoWordState(
            next=["query", "intertrial"],
            transition=lambda: rng.choice([0, 1], p=[QUERY_P, 1 - QUERY_P]),
            dur=WORD_DURATION,
            window=window,
            stim=TwoWordStim(
                win=window,
                word1="experiment",
                word2="start",
                separation=WORD_SEP,
                fixation_dot=True,
                reporting_pix=REPORT_PIX,
                reporting_pix_size=REPORT_PIX_SIZE,
                text_config=TEXT_CONFIG,
                dot_config=DOT_CONFIG,
            ),
            word_list=wordsdf,
            frequencies={"words": {"word1": None, "word2": None}},
            clock=clock,
            framerate=framerate,
            flicker_handler="frame_count",
        ),
    }

    return states_2word
