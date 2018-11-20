"""
Various handy things.
"""

import numpy as np
import networkx as nx


def assert_parameters(N,k_over_2,beta):
    """Assert that `N` is integer, `k_over_2` is integer and `0 <= beta <= 1`"""

    assert(k_over_2 == int(k_over_2))
    assert(N == int(N))
    assert(beta >= 0.0)
    assert(beta <= 1.0)


def get_largest_component(G):
    """Return the largest connected component of graph `G`."""

    subgraphs = nx.connected_component_subgraphs(G,copy=False)
    new_G = max(subgraphs, key=len)
    G = nx.convert_node_labels_to_integers(new_G)

    return G

def get_number_of_unique_two_stars_for_each_node(G):
    k = np.array([ d[1] for d in G.degree() ], dtype=float)
    num = k*(k-1.0)/2.0
    return num

def get_number_of_unique_two_stars_per_node(G):

    return np.mean(get_number_of_unique_two_stars_for_each_node(G))

def get_number_of_unique_triangles_for_each_node(G):

    A = nx.adjacency_matrix(G)

    A3 = A.dot(A).dot(A)
    T = np.array(A3.diagonal(),dtype=float) / 2.0
    #T = np.array(list(nx.triangles(G).values()), dtype=float)

    return T

def get_number_of_unique_triangles_per_node(G):

    return np.mean(get_number_of_unique_triangles_for_each_node(G))

if __name__ == "__main__":
    from time import time

    G = nx.fast_gnp_random_graph(10000,5.0/10000)

    start = time()
    A = nx.adjacency_matrix(G)
    A3 = A.dot(A).dot(A)
    T = np.array(A3.diagonal(),dtype=float) / 2.0
    end = time()
    print(T)
    print(end-start)
    start = time()
    T = np.array(list(nx.triangles(G).values()), dtype=float)
    end = time()
    print(end-start)

