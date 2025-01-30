import asyncio

import psychopy.core
import pytest

from intermodulation.core.utils import (
    lazy_time,
    maxdepth_keys,
    nested_deepkeys,
    nested_get,
    nested_iteritems,
    nested_keys,
    nested_set,
)


@pytest.fixture
def nested_dict():
    return {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3,
                "f": 4,
            },
        },
    }


@pytest.fixture
def deepkeys():
    return [
        ("a",),
        ("b", "c"),
        ("b", "d", "e"),
        ("b", "d", "f"),
    ]


@pytest.fixture
def allkeys():
    return [
        ("a",),
        ("b",),
        ("b", "c"),
        ("b", "d"),
        ("b", "d", "e"),
        ("b", "d", "f"),
    ]


class TestNested:
    def test_iteritems(self, nested_dict, deepkeys):
        for k, v in nested_iteritems(nested_dict):
            assert nested_get(nested_dict, k) == v
            deepkeys.remove(k)
        assert len(deepkeys) == 0

    def test_deepkeys(self, nested_dict, deepkeys):
        assert list(nested_deepkeys(nested_dict)) == deepkeys

    def test_keys(self, nested_dict, allkeys):
        assert list(nested_keys(nested_dict)) == allkeys

    def test_get(self, nested_dict):
        assert nested_get(nested_dict, ("a",)) == 1
        assert nested_get(nested_dict, ("b", "c")) == 2
        assert nested_get(nested_dict, ("b", "d", "e")) == 3
        assert nested_get(nested_dict, ("b", "d", "f")) == 4

    def test_set(self, nested_dict):
        nested_set(nested_dict, ("a",), 5)
        assert nested_get(nested_dict, ("a",)) == 5
        nested_set(nested_dict, ("b", "c"), 6)
        assert nested_get(nested_dict, ("b", "c")) == 6
        nested_set(nested_dict, ("b", "d", "e"), 7)
        assert nested_get(nested_dict, ("b", "d", "e")) == 7
        nested_set(nested_dict, ("b", "d", "f"), 8)
        assert nested_get(nested_dict, ("b", "d", "f")) == 8

    def test_maxdepth(self, nested_dict, deepkeys, allkeys):
        assert maxdepth_keys(nested_dict, depth=0, deepest=True) == [
            ("a",),
        ]
        assert maxdepth_keys(nested_dict, depth=1, deepest=True) == [("a",), ("b", "c")]
        assert maxdepth_keys(nested_dict, depth=2, deepest=True) == deepkeys
        assert maxdepth_keys(nested_dict, depth=-1, deepest=True) == [("a",), ("b", "c")]
        assert maxdepth_keys(nested_dict, depth=0) == [("a",), ("b",)]
        assert maxdepth_keys(nested_dict, depth=1) == [("a",), ("b",), ("b", "c"), ("b", "d")]
        assert maxdepth_keys(nested_dict, depth=2) == allkeys
        assert maxdepth_keys(nested_dict, depth=-1) == [("a",), ("b",), ("b", "c"), ("b", "d")]


def test_lazy_clock():
    clock = psychopy.core.Clock()
    asyncio.run(asyncio.sleep(0.5))
    assert asyncio.run(lazy_time(clock)) > 0.5
