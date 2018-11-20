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

def get_connection_probability_arrays(N, k_over_2, beta):
    """
    Return the connection probabilities :math:`p_S` and :math:`p_L`
    but for beta being a `numpy.ndarray`.
    """
    assert_parameters(N,k_over_2,0.0)

    assert(np.all(beta>=0.0))
    assert(np.all(beta<=1.0))

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

    B_short = binom(k, pS).pmf
    B_long = binom(N-1-k, pL).pmf

    ks = np.arange(kmax+1)
    Pk = np.array(np.zeros_like(ks), dtype=float)

    for _k in ks:
        _P_k = 0.0
        for kS in range(min(k,_k)+1):
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


def expected_number_of_unique_triangles_per_node(N,k_over_2,beta):

    if type(beta) == np.ndarray:
        pS, pL = get_connection_probability_arrays(N, k_over_2, beta)
    else:
        pS, pL = get_connection_probabilities(N,k_over_2,beta)

    N = int(N)
    k = int(2*k_over_2)

    if N % 2 == 0:
        raise ValueError("This currently only works for odd number of nodes N")
 
    R = k_over_2
    L = (N-1) // 2
 
    big_triangle = (R**2 - R)/2 + R
    small_triangle = (R**2 - R)/2
    S3 = small_triangle * 3
    S2L = 3 * big_triangle
    SL2 = 2 * (small_triangle +(L-2*R) * R) +\
          big_triangle +\
          2*(L-R)*R +\
          (2*R+1)*(L-R) - 2*big_triangle - (L-R) 
    L3 = (L-R)**2 - ((2*R+1)*(L-R) - 2*big_triangle) +\
         (L-R)**2 - big_triangle
 
    return S3 * pS**3 + S2L * pS**2*pL + SL2 * pS*pL**2 + L3 * pL**3

def expected_number_of_unique_two_stars_per_node(N,k_over_2,beta):

    if type(beta) == np.ndarray:
        pS, pL = get_connection_probability_arrays(N, k_over_2, beta)
    else:
        pS, pL = get_connection_probabilities(N,k_over_2,beta)

    N = int(N)
    k = int(2*k_over_2)

    if N % 2 == 0:
        raise ValueError("This currently only works for odd number of nodes N")
 
    R = k_over_2
    L = (N-1) // 2
 
    S2 = (R**2 - R) + R**2
    SL = 4 * (L-R) * R
    L2 = ((L-R)**2 - (L-R)) + (L-R)**2
 
    return S2 * pS**2 + SL * pS*pL + L2 * pL**2
