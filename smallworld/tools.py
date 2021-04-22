"""
Various handy things.
"""

import numpy as np
import networkx as nx

import scipy.sparse as sprs


def assert_parameters(N,k_over_2,beta):
    """Assert that `N` is integer, `k_over_2` is integer and `0 <= beta <= 1`"""

    assert(k_over_2 == int(k_over_2))
    assert(N == int(N))
    assert(beta >= 0.0)
    assert(beta <= 1.0)


def get_largest_component(G):
    """Return the largest connected component of graph `G`."""

    new_G = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
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

def get_sparse_matrix_from_rows_and_cols(N, rows, cols):

    A = sprs.csc_matrix((np.ones_like(rows),(rows,cols)), shape=(N,N),dtype=float)

    return A

def get_random_walk_eigenvalue_gap(A,maxiter=10000):

    W = A.copy()
    W = W.astype(float)
    degree = np.array(W.sum(axis=1),dtype=float).flatten()

    for c in range(W.shape[1]):
        W.data[W.indptr[c]:W.indptr[c+1]] /= degree[c]

    lambda_max,_ = sprs.linalg.eigs(W,k=3,which='LR',maxiter=maxiter)
    lambda_max = np.abs(lambda_max)
    ind_zero = np.argmax(lambda_max)
    lambda_1 = lambda_max[ind_zero]
    lambda_max2 = np.delete(lambda_max,ind_zero)
    lambda_2 = max(lambda_max2)

    return 1 - lambda_2.real

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

