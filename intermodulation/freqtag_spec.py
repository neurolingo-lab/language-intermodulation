from attridict import AttriDict

from intermodulation.core.utils import nested_iteritems

# Detailed display parameters for experiment
WORD_SEP: float = 0.3  # word separation in degrees

DISPLAY_RES = (1280, 720)
DISPLAY_DISTANCE = 120  # cm
DISPLAY_HEIGHT = 30.5 * (720 / 1080)  # cm
DISPLAY_WIDTH = 55 * (1280 / 1920)  # cm
FOVEAL_ANGLE = 5.0  # degrees
REPORT_PIX = True
REPORT_PIX_SIZE = 10
WINDOW_CONFIG = {
    "screen": 0,  # 0 is the primary monitor
    "fullscr": True,
    "winType": "pyglet",
    "allowStencil": False,
    "monitor": "testMonitor",
    "color": [-1, -1, -1],
    "colorSpace": "rgb",
    "units": "deg",
    "checkTiming": False,
}
TEXT_CONFIG = {
    "font": "Cousine Nerd Font Mono",
    "height": 0.75,
    "wrapWidth": None,
    "ori": 0.0,
    "color": "white",
    "colorSpace": "rgb",
    "opacity": None,
    "languageStyle": "LTR",
    "depth": 0.0,
    "bold": True,
}
DOT_CONFIG = {
    "size": (0.05, 0.05),
    "vertices": "circle",
    "anchor": "center",
    "colorSpace": "rgb",
    "lineColor": "white",
    "fillColor": "white",
    "interpolate": True,
}

LOGGABLES = {
    "per_state": [
        "state_number",
        "state",
        "next_state",
        "state_start",
        "target_end",
        "state_end",
        "trial_number",
        "block_number",
        "block_trial",
        "trial_end",
        "block_end",
        "word1",
        "word2",
        "word1_freq",
        "word2_freq",
        "condition",
        "truth",
    ],
    "continuous_per_state": [
        ("words", "word1"),
        ("words", "word2"),
        "reporting_pix",
    ],
}

# Trigger codes for the experiment to send via the parallel port
# Nested logic is used to group triggers by condition and sub-condition
# Organization is as follows:
# - 10-20: General triggers
# - 20-30: Query condition triggers (nexted namespace for query word being in
#      previous trial (TRUE) or not (FALSE))
# - 30-40: Two-word stimulus condition triggers (nested namespace for phrase, non-phrase, non-word)
#      And within each condition (P, NP, NW) another nested namespace for frequency tag 1 being on
#      the left or right
# - 40-50: One-word stimulus condition triggers (nested namespace for word, non-word)
#      And within each condition (W, NW) another nested namespace for frequency tag 1 being on L/R
TRIGGERS = AttriDict(
    STATEEND=10,
    TRIALEND=11,
    BLOCKEND=12,
    ITI=13,
    FIXATION=14,
    BREAK=15,
    INTERBLOCK=16,
    ABORT=17,
    ERROR=18,
    EXPEND=255,
    # 20-30 are reserved for the query condition
    QUERY=dict(
        TRUE=20,
        FALSE=21,
    ),
    # 30-40 are reserved for the two-word stimulus condition
    TWOWORD=dict(
        PHRASE=dict(
            F1LEFT=30,
            F1RIGHT=31,
        ),
        NONPHRASE=dict(
            F1LEFT=32,
            F1RIGHT=33,
        ),
        NONWORD=dict(
            F1LEFT=34,
            F1RIGHT=35,
        ),
    ),
    # 40-50 are reserved for the one-word stimulus condition
    ONEWORD=dict(
        WORD=dict(
            F1=40,
            F2=41,
        ),
        NONWORD=dict(
            F1=42,
            F2=43,
        ),
    ),
    MASK=50,
)
LUT_TRIGGERS = {v: k for k, v in nested_iteritems(TRIGGERS)}
