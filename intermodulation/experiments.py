import asyncio
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
import psychopy.constants
import psychopy.core
import psychopy.logging
import psychopy.visual

import intermodulation.stimuli as stimuli

states = SimpleNamespace(
    EXPSTART=-1,
    FIXATION=0,
    WORDPAIR=1,
    QUERY=2,
    ITI=3,
    TRIALEND=4,
    BREAK=5,
    NEWBLOCK=6,
    FINISHED=-2,
)


@dataclass
class MarkovState:
    """
    Markov state class to allow for both deterministic and probabilistic state transitions,
    computing current state duration when determining the next state.

    Can deal with:
     - Deterministic state transitions
       - `next` is a single integer for the state
     - Probabilistic state transitions
       - `next` is a tuple of two integers for the possible next states
       - `probs` is a tuple of two floats for the probabilities of those states

     - Fixed duration states
       - `dur` is a single float for the duration of the state
     - Variable duration states
       - `dur` is a tuple of two floats
       - `durfunc` is a callable that takes the two floats (bounds, characteristics of
         a distribution, etc.) and returns a duration
    """

    next: int | tuple[int, int]
    dur: float | tuple[float, float]
    probs: None | tuple[float, float] = None
    durfunc: None | Callable = None
    rng: np.random.Generator = np.random.default_rng()

    def __post_init__(self):
        if hasattr(self.next, "__len__"):
            if self.probs is None:
                raise TypeError(
                    "Probabilities must be provided for " "probabilistic state transitions."
                )
            if len(self.next) != 2 or len(self.probs) != 2:
                raise ValueError("Probabilistic state transitions must have two possible states.")
        if hasattr(self.dur, "__len__"):
            if self.durfunc is None:
                raise TypeError("Duration function must be provided for variable duration states.")
            if len(self.dur) != 2:
                raise ValueError("Variable duration states must have two duration bounds.")

    def get_next(self):
        if self.probs is None:
            next = self.next
        else:
            next = self.next[self.rng.choice([0, 1], p=self.probs)]
        if hasattr(self.dur, "__len__"):
            dur = self.durfunc(*self.dur)
        else:
            dur = self.dur
        return next, dur


@dataclass
class ExperimentSpec:
    flicker_rates: tuple[float, float]
    flicker_map: tuple[int, int]
    words: pd.DataFrame
    iti_bounds: tuple[float, float]
    fixation_t: float
    word_t: float
    n_blocks: int
    word_sep: int
    newblock_t: float = 3.0
    query_t: float = 2.0
    query_p: float = 0.0
    break_t: float = 3.0

    def __post_init__(self):
        self.flicker_rates = np.array(self.flicker_rates)
        self.flickertimer = stimuli.FlickerTimes(self.flicker_rates)

    def flicker(self, start_time: float, frame_time: float):
        return np.array(self.flickertimer.get_states(start_time, frame_time), dtype=bool)


@dataclass
class ExperimentState:
    transitions: dict[int, MarkovState]
    expspec: ExperimentSpec
    blockwords: np.ndarray
    clock: psychopy.core.Clock
    current: int = states.EXPSTART
    next: int | None = None
    t_next: float | None = None
    trial: int = 0
    t_start: float = 0.0
    block: int = 0
    block_trial: int = 0
    flicker_map: tuple[int, int] = (0, 1)
    rng: np.random.Generator = np.random.default_rng()
    queue: list[tuple[int, float]] = field(default_factory=list)

    def __post_init__(self):
        self.blockwords = self.rng.permutation(len(self.blockwords))
        if len(self.queue) == 0:
            self.build_queue()

    def next_state(self):
        if len(self.queue) == 0:
            raise AttributeError("Queue is empty. Cannot get next state.")

        self.current = self.queue.pop(0)[0]
        if len(self.queue) == 0:
            self.next = None
            self.t_next = None
        else:
            self.next = self.queue[0][0]
            self.t_next = self.queue[0][1]

    def build_queue(self):
        if len(self.queue) != 0:
            raise AttributeError("Queue is not empty. Cannot build a new queue.")
        if self.current != states.TRIALEND and self.current != states.EXPSTART:
            psychopy.logging.warn(
                "Current state is not TRIALEND or EXPSTART." " Building a new queue anyways."
            )

        self.queue.append(self.transitions[self.current].get_next())
        self.next, self.t_next = self.queue[0]
        while self.queue[-1][0] not in (states.TRIALEND, states.FINISHED):
            new_next, new_t = self.transitions[self.queue[-1][0]].get_next()
            self.queue.append((new_next, new_t + self.queue[-1][1]))

    def next_trial(self):
        if len(self.queue) != 0:
            psychopy.logging.warn("Queue is not empty. Starting new trial anyways.")
            self.queue = []
        self.trial += 1
        self.block_trial += 1
        if self.block_trial >= len(self.blockwords):
            self.new_block()
        self.build_queue()
        self.flicker_map = self.rng.permutation(self.flicker_map)
        self.next_state()

    def new_block(self):
        self.blockwords = self.rng.permutation(len(self.blockwords))
        self.block += 1
        self.block_trial = 0
        if self.block >= self.expspec.n_blocks:
            self.current = states.FINISHED
            self.next = None
            self.t_next = None
            psychopy.logging.warn("Experiment finished.")
        else:
            self.current = states.NEWBLOCK


class WordFreqTagging:
    def __init__(
        self,
        window_config: dict,
        expspec: ExperimentSpec,
        text_config: dict[str, str | float | None] = stimuli.TEXT_CONFIG,
        dot_config: dict[tuple[float, float], str, str, str, str, bool] = stimuli.DOT_CONFIG,
        random_seed: int = None,
    ):
        # Store experiment variables
        self.expspec = expspec
        self.words = expspec.words
        self.text_config = text_config
        self.dot_config = dot_config

        # Create the window
        self.win = psychopy.visual.Window(**window_config)
        self.framerate = self.win.getActualFrameRate(infoMsg="Preparing to do science...")
        if self.framerate is None:
            self.framerate = 60
            psychopy.logging.warn("Could not get framerate, using 60 Hz.")

        # Set up the internal clock and the timing log
        self.clock = psychopy.core.Clock()
        self.t_log = ExperimentLog()
        self.flip_times = []
        self.log_update = None

        # Initialize the internal state
        self.rng = np.random.default_rng(random_seed)
        self.transitions = {
            states.FIXATION: MarkovState(next=states.WORDPAIR, dur=expspec.fixation_t),
            states.WORDPAIR: MarkovState(
                next=(states.QUERY, states.ITI),
                dur=expspec.word_t,
                probs=(expspec.query_p, 1 - expspec.query_p),
            ),
            states.QUERY: MarkovState(next=states.ITI, dur=expspec.query_t),
            states.ITI: MarkovState(
                next=states.TRIALEND, dur=expspec.iti_bounds, durfunc=self.rng.uniform
            ),
            states.TRIALEND: MarkovState(next=states.FIXATION, dur=1 / self.framerate),
            states.EXPSTART: MarkovState(next=states.FIXATION, dur=0.1),
            states.NEWBLOCK: MarkovState(next=states.FIXATION, dur=expspec.newblock_t),
            states.BREAK: MarkovState(next=states.FIXATION, dur=expspec.break_t),
            states.FINISHED: MarkovState(next=states.FINISHED, dur=0.0),
        }
        self.state = ExperimentState(
            transitions=self.transitions,
            expspec=expspec,
            blockwords=self.words.index.to_numpy(),
            clock=psychopy.core.Clock(),
            rng=self.rng,
        )

    def start_trial(self):
        # Create the trial components
        wordpair = self.words.loc[self.state.blockwords[self.state.block_trial]]
        flicker_map = self.state.flicker_map
        self.trial = stimuli.Trial(
            window=self.win,
            words=wordpair.iloc[:2].values,
            text_config=self.text_config,
            dot_config=self.dot_config,
            word_sep=self.expspec.word_sep,
        )
        self.state.clock.reset()

        async def record_start(self):
            self.trial.start_t = self.state.clock.getTime()

        log_items = {
            "fixation_start": "fliptime",
            "trial_number": "state.trial",
            "block_number": "state.block",
            "block_trial": "state.block_trial",
            "word_1": wordpair.iloc[0],
            "word_1_freq": self.expspec.flicker_rates[flicker_map[0]],
            "word_2": wordpair.iloc[1],
            "word_2_freq": self.expspec.flicker_rates[flicker_map[1]],
            "word_cond": wordpair["cond"],
        }

        futures = []
        futures.append(record_start(self))
        for key, value in log_items.items():
            futures.append(self.t_log.update("trials", key, value, self))
        return futures

    def start_wordpair(self):
        states = {"words": {0: True, 1: True}, "shapes": {"fixdot": True}}
        self.trial.update_stim(states)
        futures = []
        futures.append(self.t_log.update("trials", "wordpair_start", "fliptime", self))
        for word in states["words"]:
            futures.append(
                self.t_log.update(
                    "trial_states", word, ("fliptime", True), self, key2=self.state.trial
                )
            )
        return futures

    def flicker_wordpair(self, next_flip):
        newstates = self.expspec.flicker(self.trial.start_t, next_flip)
        flicker_map = self.state.flicker_map
        states = {
            "words": {0: newstates[flicker_map[0]], 1: newstates[flicker_map[1]]},
            "shapes": {"fixdot": True},
        }
        changed = self.trial.update_stim(states)
        futures = []
        for key, word in changed:
            if key == "words":
                futures.append(
                    self.t_log.update(
                        "trial_states",
                        word,
                        ("fliptime", states[key][word]),
                        self,
                        key2=self.state.trial,
                    )
                )
        return futures

    def end_stimuli(self):
        self.trial.remove_stim()
        futures = []
        futures.append(self.t_log.update("trials", "wordpair_end", "fliptime", self))
        for word in range(2):
            futures.append(
                self.t_log.update(
                    "trial_states", word, ("fliptime", False), self, key2=self.state.trial
                )
            )
        return futures

    def end_trial(self):
        futures = []
        futures.append(self.t_log.update("trials", "iti_end", "fliptime", self))
        del self.trial
        return futures

    def update(self):
        next_flip = self.win.getFutureFlipTime(clock=self.state.clock)
        self.win.callOnFlip(self._get_exp_flip_time)
        futures = []
        match self.state:
            case ExperimentState(
                current=states.EXPSTART | states.ITI,
                next=states.FIXATION | states.TRIALEND,
                t_next=start_t,
            ):
                if start_t <= next_flip:
                    updates = []
                    self.state.next_state()
                    if self.state.current == states.TRIALEND:
                        updates.extend(self.end_trial())
                        self.state.next_trial()
                    updates.extend(self.start_trial())
                    futures.extend(updates)
            case ExperimentState(current=states.FIXATION, next=states.WORDPAIR, t_next=start_t):
                if start_t <= next_flip:
                    self.state.next_state()
                    updates = self.start_wordpair()
                    futures.extend(updates)
            case ExperimentState(
                current=states.WORDPAIR, next=states.ITI | states.QUERY, t_next=start_t
            ):
                if start_t <= next_flip:
                    self.state.next_state()
                    if self.state.current == states.ITI:
                        updates = self.end_stimuli()
                    elif self.state.current == states.QUERY:
                        raise NotImplementedError("Query state not implemented.")
                    futures.extend(updates)
                else:
                    updates = self.flicker_wordpair(next_flip)
                    futures.extend(updates)
            case (states.FINISHED, _):
                return

        self.win.flip()
        asyncio.run(self._update_log(futures))

    def end(self):
        self.win.close()

    def run(self):
        while self.state.current != states.FINISHED:
            try:
                self.update()
            except Exception as e:
                self.end()
                psychopy.logging.error(f"Error in experiment: {e}")
                self.t_log.save("error_log.pkl")
                raise e

        self.end()
        return self.t_log

    def _get_exp_flip_time(self):
        flipt = self.clock.getTime()
        self.flip_times.append(flipt)
        self.last_flip = flipt

    def _run_tasks(self):
        pass

    async def _update_log(self, futures):
        await asyncio.gather(*futures)
        return


@dataclass
class ExperimentLog:
    trials: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    trial_states: dict[int, dict[int, list[tuple[float, bool]]]] = field(
        default_factory=lambda: {0: defaultdict(list), 1: defaultdict(list)}
    )

    async def update(self, attrib, key, value, exp, key2=None):
        value = self.parse_value(value, exp)

        if key2 is None:
            getattr(self, attrib)[key].append(value)
        else:
            getattr(self, attrib)[key][key2].append(value)

    def parse_value(self, value, exp):
        twoval = False
        if isinstance(value, tuple):
            twoval = True
            secondval = value[1]
            value = value[0]
            if not isinstance(value, str):
                raise ValueError(f"Invalid value string. Must have string in first position.")
        if isinstance(value, str):
            match value.split("."):
                case ["fliptime"]:
                    return1 = exp.last_flip
                case ["state", subkey]:
                    return1 = getattr(exp.state, subkey)
                case [other]:
                    return1 = other
        else:
            return1 = value
        if twoval:
            return return1, secondval
        else:
            return return1

    def save(self, fn: str):
        with open(fn, "wb") as fw:
            pickle.dump({"trial_states": self.trial_states, "trials": self.trials}, fw)
