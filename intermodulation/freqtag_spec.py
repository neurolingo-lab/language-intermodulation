from pathlib import Path
from types import SimpleNamespace

from attridict import AttriDict

# Parameters for the experiment itself
WORDSPATH = Path(__file__).parents[1]
MINIBLOCK_LEN = 10
N_BLOCKS = 3
N_1W_BLOCKS = 2
FREQUENCIES = [6, 7.05882353]
WORD_DUR = 2.833333333333333333333333
ITI_BOUNDS = [1.0, 3.0]
FIXATION_DUR = 0.5
QUERY_PAUSE_DUR = 1.0
QUERY_DUR = 3.0
LOCALIZER_MINIBLOCK_LEN = 12
LOCALIZER_WORD_DUR = 0.2
LOCALIZER_ITI_BOUNDS = [0.4, 0.6]

PAUSE_KEY = "4"
TASK1_EXPL = dict(
    text="This experiment will begin with a dot on the screen.\n\n Stare at the dot when you see it, "
    "and continue to stare at the dot when words appear.\nThese words will be "
    "followed by a pause, then questions.\n\nIf the question word was present in the last sequence of words you just saw, "
    "press yes (index finger), otherwise press no (middle finger).",
    anchorHoriz="center",
    alignText="center",
    pos=(0, 0),
    color="white",
    height=1.0,
)
INTERTASK_TEXT = (
    "Part 1 Done! Time for a break!\n" "Press the pause button to continue once you're ready."
)
INTERTASK_TEXT2 = "Please let the experimenter know you're ready and the task will start."
LOCALIZER_EXPL = dict(
    text="This experiment will begin with a dot on the screen.\n\n Stare at the dot when you see it, "
    "and continue to stare at the dot when words appear.",
    anchorHoriz="center",
    alignText="center",
    pos=(0, 0),
    color="white",
    height=1.0,
)
# Debug parameters
debug = SimpleNamespace(
    N_BLOCKS=2,
    N_1W_BLOCKS=1,
    FREQUENCIES=[16.666666, 25.0],
    WORD_DUR=2.1,
    ITI_BOUNDS=[0.5, 1.0],
    FIXATION_DUR=2.0,
    FRAMERATE=240,
    FULLSCR=False,
    REPORT_PIX_SIZE=36,
    WINDOW_CONFIG={
        "screen": 0,  # 0 is the primary monitor
        "fullscr": False,
        "winType": "pyglet",
        "allowStencil": False,
        "monitor": "testMonitor",
        "color": [-1, -1, -1],
        "colorSpace": "rgb",
        "units": "deg",
        "checkTiming": False,
    },
)

# Detailed display parameters for experiment
TRIGGER = "/dev/parport0"
FULLSCR = True
FRAMERATE = 240
WORD_SEP: float = 0.3  # word separation in degrees

DISPLAY_RES = (1280, 720)
DISPLAY_DISTANCE = 120  # cm
DISPLAY_HEIGHT = 30.5 * (720 / 1080)  # cm
DISPLAY_WIDTH = 55 * (1280 / 1920)  # cm
FOVEAL_ANGLE = 5.0  # degrees
REPORT_PIX = True
REPORT_PIX_SIZE = 36
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
    MINIBLOCK=dict(
        TWOWORD=dict(
            PHRASE=dict(
                F1LEFT=130,
                F1RIGHT=131,
            ),
            NONPHRASE=dict(
                F1LEFT=132,
                F1RIGHT=133,
            ),
            NONWORD=dict(
                F1LEFT=134,
                F1RIGHT=135,
            ),
        ),
        ONEWORD=dict(
            WORD=dict(
                F1=140,
                F2=141,
            ),
            NONWORD=dict(
                F1=142,
                F2=143,
            ),
        ),
    ),
    LOCALIZER=dict(
        FIXATION=50,
        ITI=51,
        SENTENCE=52,
        NONWORD=53,
    ),
)


def nested_iteritems(d):
    for k, v in d.items():
        if isinstance(v, dict):
            for subk, v in nested_iteritems(v):
                yield (k, *subk), v
        else:
            yield (k,), v


def nested_deepkeys(d):
    for k, v in d.items():
        if isinstance(v, dict):
            for subk in nested_deepkeys(v):
                yield (k, *subk)
        else:
            yield (k,)


LUT_TRIGGERS = {v: k for k, v in nested_iteritems(TRIGGERS)}
for i in range(1, 256):
    if i not in LUT_TRIGGERS:
        LUT_TRIGGERS[i] = ("INVALID",)

VALID_TRANS = (
    *(
        ("/".join(k), "STATEEND")
        for k in nested_deepkeys(TRIGGERS)
        if k[0] in ("QUERY", "TWOWORD", "ONEWORD", "ITI", "FIXATION", "BREAK", "MASK")
    ),
    ("STATEEND", "TRIALEND"),
    *(
        ("STATEEND", "/".join(k))
        for k in nested_deepkeys(TRIGGERS)
        if k[0] in ("QUERY", "TWOWORD", "ONEWORD", "ITI", "BREAK", "MASK", "FIXATION")
    ),
    ("TRIALEND", "BLOCKEND"),
    ("TRIALEND", "FIXATION"),
    ("TRIALEND", "BREAK"),
    ("BLOCKEND", "FIXATION"),
    ("BLOCKEND", "BREAK"),
    *(
        (t, "BREAK")
        for t in (
            "BLOCKEND",
            "TRIALEND",
        )
    ),
)
