from intermodulation.utils import maxdepth_keys, nested_get, nested_keys, nested_pop, nested_set


class StatefulStim:
    def __init__(self, window, constructors):
        self.win = window
        self.construct = constructors
        allkeys = nested_keys(constructors)
        self.states = {}
        for k in allkeys:
            nested_set(self.states, k, False)
        self.stim = {}

    def start_stim(self, constructor_kwargs):
        """
        Create the stimulus objects using the constructors and kwargs passed. The structure of the
        dict of kwargs must match the structure of the constructors dict. Some constructors may
        be skipped if they are not passed in the kwargs."""
        const_keys = maxdepth_keys(self.construct, depth=-1)
        kw_keys = maxdepth_keys(constructor_kwargs, depth=-1)
        if not set(kw_keys).issubset(set(const_keys)):
            raise ValueError("Mismatched keys between constructor and kwargs.")
        if any(["win" in nested_get(constructor_kwargs, k) for k in kw_keys]):
            raise ValueError("Cannot pass window to StatefulStim as it already has a window.")
        for k in const_keys:
            nested_set(
                self.stim,
                k,
                nested_get(self.construct, k)(win=self.win, **nested_get(constructor_kwargs, k)),
            )
            nested_set(self.states, k, True)

    def update_stim(self, newstates):
        updatekeys = nested_keys(newstates)
        if not set(updatekeys).issubset(set(nested_keys(self.states))):
            raise ValueError("Mismatched keys between new states and current states.")
        changed = []
        for k in updatekeys:
            currstate = nested_get(self.states, k)
            if currstate != nested_get(newstates, k):
                changed.append(k)
            nested_get(self.stim, k).setAutoDraw(nested_get(newstates, k))
            nested_set(self.states, k, nested_get(newstates, k))
        return changed

    def end_stim(self):
        allkeys = nested_keys(self.stim)
        for k in allkeys:
            nested_pop(self.stim, k).setAutoDraw(False)
            nested_set(self.states, k, False)
