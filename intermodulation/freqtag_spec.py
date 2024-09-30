import numpy as np

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
    ],
    "continuous_per_state": [
        ("words", "word1"),
        ("words", "word2"),
    ],
}


def assign_frequencies_to_words(wordsdf, freq1: float, freq2: float, rng: np.random.Generator):
    """
    Counterbalanced assignment of one of two frequencies to each word in a DataFrame. Returns
    a new DataFrame with the columns "w1_freq" and "w2_freq" added. Each pair will have one of
    each frequency.

    Parameters
    ----------
    wordsdf : pd.DataFrame
        Dataframe of word pairs to assign frequencies to.
    freq1 : float
        Value to assign to half of the words.
    freq2 : float
        Value to assign to the other half of the words.
    rng : np.random.Generator
        RNG object to use
    """
    outdf = wordsdf.copy()
    mask = np.zeros(len(wordsdf), dtype=bool)
    f_choices = rng.choice(np.arange(len(wordsdf)), size=len(wordsdf) // 2, replace=False)
    mask[f_choices] = True
    w1_f = np.where(mask, freq1, freq2)
    w2_f = np.where(mask, freq2, freq1)

    outdf["w1_freq"] = w1_f
    outdf["w2_freq"] = w2_f
    return outdf


def add_logging_to_controller(controller, states, twoword, oneword, query):
    controller.add_loggable(
        twoword, "start", "word1", object=states[twoword].stim, attribute="word1"
    )
    controller.add_loggable(
        twoword, "start", "word2", object=states[twoword].stim, attribute="word2"
    )
    controller.add_loggable(
        twoword,
        "start",
        "word1_freq",
        object=states[twoword],
        attribute=("frequencies", "words", "word1"),
    )
    controller.add_loggable(
        twoword,
        "start",
        "word2_freq",
        object=states[twoword],
        attribute=("frequencies", "words", "word2"),
    )
    controller.add_loggable(
        twoword, "start", "condition", object=states[twoword], attribute="phrase_cond"
    )
    controller.add_loggable(query, "start", "word1", object=states[query], attribute="test_word")
