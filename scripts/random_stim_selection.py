from pathlib import Path

import numpy as np
import pandas as pd

stimpath = Path("../word_ngrams/").resolve()

phrases = pd.read_csv(stimpath / "phrases_final_candidates.csv")
cons_clust = pd.read_csv(stimpath / "cons_clust_final_candidates.csv")
SEED = 42
rng = np.random.default_rng(SEED)

# Select 60 phrases ensuring that no words are repeated
# There are 169 unique Adj/Det first words, sample one pair from each of those groups
w1_unique = phrases.groupby("w1").sample(1, random_state=rng)
# Now ensure we only have one pair for each unique noun, w2, which may be repeated with multipl w1
w2_unique = w1_unique.groupby("w2").sample(1, random_state=rng)
# Select the 60 least common from that final sample of ~110
phrase = w2_unique.sort_values("ngram_rank", ascending=False).iloc[:60]
phrase = phrase[["w1", "w2"]].copy().reset_index(drop=True)

# Select 60 consonant clusters ensuring even sampling of the initial letter bigrams
cons_subsamp = cons_clust.groupby("clus1").sample(3, random_state=rng).sample(60, random_state=rng)
# Use the remainder of the clusters for the single-word stimuli
oneword_nw = (
    cons_clust.loc[~cons_clust.index.isin(cons_subsamp.index)]
    .sample(30, random_state=rng)[["non_word"]]
    .rename(columns={"non_word": "w1"})
)

# Generate a copy of the phrases list with one random word replaced by a consonant cluster
# Start by selecting 60 words from the phrases list, balanced for noun or adjective/determiner
phrasemelt = phrase.melt()
word_list = phrasemelt.groupby("variable").sample(30)
# Take the remaining words for use in the single-word portion of the task
oneword_w = (
    phrasemelt.loc[~phrasemelt.index.isin(word_list.index)]
    .sample(30)
    .copy()[["value"]]
    .rename(columns={"value": "w1"})
)
# Next choose the position these words will occupy in the non-word pair by making a boolean mask
# of whether the word is in the first (True) or second (False) position
posmask = np.zeros(60, dtype=bool)
posmask[rng.choice(np.arange(60), 30, replace=False)] = True
# Now create a DataFrame with the words and their positions
nw_w1 = np.where(~posmask, word_list["value"], cons_subsamp["non_word"])
nw_w2 = np.where(posmask, word_list["value"], cons_subsamp["non_word"])
nonword = pd.DataFrame({"w1": nw_w1, "w2": nw_w2, "condition": "non-word"})

# Generate a "non-phrase" list which is simply the reverse of phrase
nonphrase = phrase.copy()
nonphrase = nonphrase.rename(columns={"w1": "w2", "w2": "w1"})
nonphrase["condition"] = "non-phrase"

# Combine the three lists into one
phrase["condition"] = "phrase"
stimuli_2w = pd.concat([phrase, nonword, nonphrase], ignore_index=True)

# Combine the leftover words and the consonant clusters into a single list of single-word stimuli
oneword_w["condition"] = "word"
oneword_nw["condition"] = "non-word"
stimuli_1w = pd.concat([oneword_w, oneword_nw], ignore_index=True)

# Save the stimuli lists
stimuli_2w.to_csv(stimpath.parent / "two_word_stimuli.csv")
stimuli_1w.to_csv(stimpath.parent / "one_word_stimuli.csv")
