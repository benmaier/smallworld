"""
Generate small-world networks
"""

import networkx as nx
from numpy import random

from smallworld.theory import get_connection_probabilities
from smallworld.tools import assert_parameters
from smallworld.tools import get_largest_component as _get_largest_component

def get_fast_smallworld_graph(N, k_over_2, beta):
    """
    Loop over all possibler short-range edges and add
    each with probability :math:`p_S`.

    Sample :math:`m_L` from a binomial distribution
    :math:`\mathcal B(N(N-1-k)/2, p_L`. For each edge
    m with :math:`0\leq m\leq m_L` sample a node :math:`u`
    and a possible long-range neighbor :math:`v` until
    egde :math:`(u,v)` has not yet been sampled.
    Then add :math:`(u,v)` to the network.
    """

    assert_parameters(N,k_over_2,beta)
    pS, pL = get_connection_probabilities(N,k_over_2,beta)

    G = nx.Graph()
    G.add_nodes_from(range(N))

    N = int(N)
    k_over_2 = int(k_over_2)

    
    # add short range edges in order (Nk/2)
    for u in range(N):
        for v in range(u+1, u+k_over_2+1):
            if random.rand() < pS:
                G.add_edge(sorted([u,v % N]))

    # sample number of long-range edges
    mL_max = N*(N-1-2*k_over_2) // 2
    mL = random.binomial(mL_max, pL, 1)

    for m in range(mL):
        while True:
            u = random.randint(0,N-1)
            v = u + k_over_2 + random.randint(1, N - 1 - 2*k_over_2)
            v %= N
            if not G.has_edge(u,v):
                G.add_edge(u,v)
                break

    return G
    

def get_smallworld_graph(N,k_over_2,beta,use_slow_algorithm=False,get_largest_component=False):
    """
    Get a modified small-world network with number of nodes `N`,
    mean degree `k=2*k_over_2` and long-range impact `0 <= beta <= 1`.
    At beta = 0,
    """

    if use_slow_algorithm:
        G = nx.Graph()
        G.add_nodes_from(range(N))

        G.add_edges_from(get_edgelist_slow(N,k_over_2,beta))
    else:
        G = get_fast_graph(N, k_over_2, beta)

    if get_largest_component:
        G = _get_largest_component(G)

    return G


def get_edgelist_slow(N,k_over_2,beta):
    """
    Loop over all pair of nodes, calculate their lattice
    distance and add an edge according to short-range
    or long-range connection probability, respectively
    """

    assert_parameters(N,k_over_2,beta)
    pS, pL = get_connection_probabilities(N,k_over_2,beta)

    N = int(N)
    k_over_2 = int(k_over_2)

    E = []

    for i in range(N-1):
        for j in range(i+1,N):

            distance = j - i
            probability = pL

            if (distance <= k_over_2) or ((N - distance) <= max_neighbor):
                p = pS
            else:
                p = pL

            if random.rand() < pL:
                E.append((u,v))

    return E

