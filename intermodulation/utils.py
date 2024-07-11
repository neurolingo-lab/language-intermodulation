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
