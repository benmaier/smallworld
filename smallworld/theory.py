"""
Compute various things analytically.
"""

import numpy as np
from scipy.stats import binom

from smallworld.tools import assert_parameters

def binomial_mean(n, p):
    return n*p

def binomial_variance(n, p):
    return n*p*(1-p)

def binomial_second_moment(n, p):
    return binomial_variance(n, p) + binomial_mean(n, p)**2


def get_connection_probabilities(N,k_over_2,beta):
    """
    Return the connection probabilities :math:`p_S` and :math:`p_L`.
    """

    assert_parameters(N,k_over_2,beta)

    k = float(int(k_over_2 * 2))

    pS = k / (k + beta*(N-1.0-k))
    pL = k * beta / (k + beta*(N-1.0-k))

    return pS, pL

def get_degree_distribution(N,k_over_2,beta,kmax=None):
    """
    Return degrees `k` and corresponding probabilities math:`P_k`
    up to maximum degree `kmax` (= `N-1` if not provided).
    """


    if kmax is None:
        kmax = N-1

    assert_parameters(N,k_over_2,beta)

    k = int(2*k_over_2)
    N = int(N)

    pS, pL = get_connection_probabilities(N,k_over_2,beta)

    B_short = binom(k, pS)
    B_long = binom(N-1-k, pL)

    ks = np.arange(kmax+1)
    Pk = np.zeros_like(ks)

    for _k in ks:
        _P_k = 0.0
        for kS in range(min(k,_k)):
            _P_k += B_short(kS) * B_long(_k-kS)
        Pk[_k] = _P_k

    return ks, Pk
            

def get_degree_second_moment(N,k_over_2,beta):

    assert_parameters(N,k_over_2,beta)

    pS, pL = get_connection_probabilities(N,k_over_2,beta)

    k = int(2*k_over_2)

    return   binomial_second_moment(k, pS)\
           + binomial_second_moment(N-1-k, pL) \
           + 2 * binomial_mean(k, pS) * binomial_mean(N-1-k, pL)

def get_degree_variance(N,k_over_2,beta):

    assert_parameters(N,k_over_2,beta)

    pS, pL = get_connection_probabilities(N,k_over_2,beta)
    k = int(2*k_over_2)

    return binomial_variance(k, pS) + binomial_variance(N-1-k, pL)


def number_of_triangles(N,k_over_2,beta):

    assert_parameters(N,k_over_2,beta)

    # TODO
