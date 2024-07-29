from collections.abc import Mapping, Sequence


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


def nested_keys(d, keys=[]):
    for k, v in d.items():
        if type(v) is dict:
            yield (*keys, k)
            yield from nested_keys(v, keys=[*keys, k])
        else:
            yield (*keys, k)


def maxdepth_keys(d, depth=10, deepest=False):
    """Return all keys in a nested dictionary up to a maximum depth. Not only those keys not pointing to dicts.
    If depth is negative, return keys up to -N length relative to the maximum depth of the dict."""
    allkeys = list(nested_deepkeys(d)) if deepest else list(nested_keys(d))
    if depth < 0:
        maxd = max(map(len, allkeys))
        return [k for k in allkeys if len(k) <= maxd + depth]
    depth += 1
    return [k for k in allkeys if len(k) <= depth]


def nested_get(d, keys):
    for key in keys[:-1]:
        d = d[key]
    return d[keys[-1]]


def nested_set(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def nested_pop(d, keys):
    for key in keys[:-1]:
        d = d[key]
    return d.pop(keys[-1])


async def lazy_time(clock):
    return clock.getTime()

def parse_calls(call_list):
    """
    Parse a list of calls to be made during a state.

    Parameters
    ----------
    call_list : list[Tuple[Callable, ...]]
        List of functions to call during the state.
    index : int
        Index of the call list to parse.

    Returns
    -------
    Callable
        The callable function to be called.
    Tuple
        The arguments to be passed to the callable.
    Mapping
        The keyword arguments to be passed to the callable.
    """
    for call in call_list:
        if not isinstance(call, Sequence):
            raise TypeError("Call list must be a list-like.")
        sequences = tuple(filter(lambda x: isinstance(x, Sequence), call))
        mappings = tuple(filter(lambda x: isinstance(x, Mapping), call))
        f = call[0]
        if len(sequences) > 0:
            args = sequences[0]
            if len(sequences) > 1:
                raise ValueError("Only one sequence of arguments is allowed.")
        else:
            args = ()
        
        if len(mappings) > 0:
            kwargs = mappings[0]
            if len(mappings) > 1:
                raise ValueError("Only one mapping of keyword arguments is allowed.")
        else:
            kwargs = {}

        yield f, args, kwargs