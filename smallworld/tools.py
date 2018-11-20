"""
Various handy things.
"""

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
