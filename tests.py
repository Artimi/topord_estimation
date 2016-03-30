# -*- coding: utf-8 -*-
import numpy as np
import numpy.testing
from topord_estimator.estimate import distance, TopordTPA, precedes, swap
import igraph
import pytest


@pytest.fixture
def g():
    return igraph.Graph(n=12, directed=True, edges=[(0, 1), (0, 2), (0, 3), (0, 4), (1, 10), (2, 5),
                                                 (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (9, 10),
                                                 (7, 11), (8, 11), (10, 11)])


def is_precedence_feasible(graph, array):
    controlled = set()
    for i in array:
        controlled.add(i)
        predecessors = set(graph.subcomponent(i, mode=igraph.IN))
        if not predecessors.issubset(controlled):
            return False
    return True


def test_distance_zero():
    linext = np.array([1, 2, 3, 4])
    vector = np.array([0.9, 1.2, 2.7, 3.1])
    assert distance(vector, linext) == 0


def test_distance():
    linext = np.array([2, 1, 3, 4])
    vector = np.array([0.9, 1.2, 2.7, 3.1])
    assert distance(vector, linext) == 1


def test_generate(g):
    tpa = TopordTPA(g)
    sigma = tpa.generate(10, tpa.n)
    assert sorted(sigma) == list(range(tpa.n))
    assert is_precedence_feasible(g, sigma)


def test_precedes(g):
    assert not precedes(g, -1, -1)
    assert not precedes(g, 5, -1)
    assert precedes(g, 0, 1)
    assert precedes(g, 2, 9)
    assert not precedes(g, 9, 2)
    assert not precedes(g, 5, 6)


def test_swap():
    a = np.arange(4)
    swap(a, 0, 1)
    numpy.testing.assert_array_equal(a, np.array([1, 0, 2, 3]))
    swap(a, 1, 3)
    numpy.testing.assert_array_equal(a, np.array([1, 3, 2, 0]))


def test_sigma_home(g):
    tpa = TopordTPA(g)
    assert is_precedence_feasible(g, tpa.sigma_home)

def test_get_uniform(g):
    tpa = TopordTPA(g)
    S = tpa.generate(tpa.n, tpa.n)
    X = tpa.sample_continuous(S, tpa.n)
    assert distance(X, tpa.sigma_home) < tpa.n


