from pathlib import Path

import numpy as np
import pandas as pd

# %%  Load in data of group stimuli and consonant clusters
stimpath = Path(__file__).parents[2] / "word_ngrams"

groupstims = {
    "odd": pd.read_csv(stimpath / "final_P_NP_oddsubs.csv").iloc[:, 1:],
    "even": pd.read_csv(stimpath / "final_P_NP_evensubs.csv").iloc[:, 1:],
}
cons_clust = pd.read_csv(stimpath / "cons_clust_final_candidates.csv")
SEED = 42
rng = np.random.default_rng(SEED)


# %%  Generate candidate lists for non-word stimuli
phrase = {
    "odd": groupstims["odd"].query("condition == 'phrase'"),
    "even": groupstims["even"].query("condition == 'phrase'"),
}
nonphrase = {
    "odd": groupstims["odd"].query("condition == 'non-phrase'"),
    "even": groupstims["even"].query("condition == 'non-phrase'"),
}
repwords = {
    "odd": pd.concat(
        (phrase["odd"].sample(30, random_state=rng), nonphrase["odd"].sample(30, random_state=rng))
    ),
    "even": pd.concat(
        (
            phrase["even"].sample(30, random_state=rng),
            nonphrase["even"].sample(30, random_state=rng),
        )
    ),
}
# Select 90 consonant clusters ensuring even sampling of the initial letter bigrams
cons_subsamp = cons_clust.groupby("clus1").sample(4, random_state=rng).sample(90, random_state=rng)

# %% Generate the stimuli lists
# Select 60 consonant clusters for the non-word stimuli pairs
twoword_clust = {}
oneword_clust = {}
for group in ["odd", "even"]:
    tw_idx = rng.choice(np.arange(90), 60, replace=False)
    ow_idx = np.array([x for x in np.arange(90) if x not in tw_idx])
    ow_idx = rng.permutation(ow_idx)
    twoword_clust[group] = cons_subsamp["non_word"].values[tw_idx]
    oneword_clust[group] = cons_subsamp["non_word"].values[ow_idx]

# Generate a copy of the phrases list with one random word replaced by a consonant cluster
# Start by selecting 60 words from the phrases list, balanced for noun or adjective/determiner
twoword_nw = {}
remaining_seen = {}
remaining_unseen = {}
for group in ["odd", "even"]:
    seenmelt = phrase[group].melt(value_vars=["w1", "w2"])
    other = "even" if group == "odd" else "odd"
    unseenmelt = phrase[other].melt(value_vars=["w1", "w2"])
    seen_list = seenmelt.groupby("variable").sample(15, random_state=rng)
    remaining_seen[group] = seenmelt.loc[~seenmelt.index.isin(seen_list.index)]
    unseen_list = unseenmelt.groupby("variable").sample(15, random_state=rng)
    remaining_unseen[group] = unseenmelt.loc[~unseenmelt.index.isin(unseen_list.index)]
    seenposmask = np.zeros(30, dtype=bool)
    seenposmask[rng.choice(np.arange(30), 15, replace=False)] = True
    unseenposmask = np.zeros(30, dtype=bool)
    unseenposmask[rng.choice(np.arange(30), 15, replace=False)] = True
    group_nw = pd.concat(
        [
            pd.DataFrame(
                {
                    "w1": seen_list["value"].where(seenposmask, twoword_clust[group]),
                    "w2": seen_list["value"].where(~seenposmask, twoword_clust[group]),
                    "condition": "non-word",
                }
            ),
            pd.DataFrame(
                {
                    "w1": unseen_list["value"].where(unseenposmask, twoword_clust[group]),
                    "w2": unseen_list["value"].where(~unseenposmask, twoword_clust[group]),
                    "condition": "non-word",
                }
            ),
        ],
        ignore_index=True,
    )
    twoword_nw[group] = group_nw.sample(frac=1, random_state=rng)

for group in ["odd", "even"]:
    allstim = pd.concat([phrase[group], nonphrase[group], twoword_nw[group]], ignore_index=True)
    allstim.to_csv(stimpath.parents[0] / f"{group}_two_word_stimuli.csv")

# %%
# Take the remaining words for use in the single-word portion of the task
oneword_words = {}
oneword_stim = {}
for group in ["odd", "even"]:
    seen = remaining_seen[group].groupby("variable").sample(8, random_state=rng)
    unseen = remaining_unseen[group].groupby("variable").sample(8, random_state=rng)
    oneword_words[group] = (
        pd.concat([seen, unseen], ignore_index=True).iloc[:15].rename(columns={"value": "w1"})
    )
    oneword_words[group]["condition"] = "word"
    nw = pd.DataFrame(oneword_clust[group], columns=["w1"])
    nw["condition"] = "non-word"
    oneword_stim[group] = pd.concat([oneword_words[group], nw], ignore_index=True)

for group in ["odd", "even"]:
    oneword_stim[group].to_csv(stimpath.parents[0] / f"{group}_one_word_stimuli.csv")
