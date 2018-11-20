import matplotlib.pyplot as pl
import numpy as np
from collections import Counter

from smallworld import get_smallworld_graph
from smallworld.theory import get_degree_distribution

from time import time

def get_counter_hist(ks):
    c = Counter(ks)
    kmax = np.max(ks)
    k = np.arange(kmax+1,dtype=int)
    s = 0
    P = np.array(np.zeros_like(k),dtype=float)
    for _k in k:
        P[_k] = c[_k]
        s += c[_k]
    P /= s

    return k, P

N = 300
k_over_2 = 3
beta = 0.5

N_meas = 100

ks_fast = []
ks_slow = []

t_fast = []
t_slow = []

for meas in range(N_meas):

    print(meas)

    start = time()
    G_fast = get_smallworld_graph(N, k_over_2, beta)
    end = time()
    t_fast.append(end-start)

    start = time()
    G_slow = get_smallworld_graph(N, k_over_2, beta, use_slow_algorithm = True)
    end = time()
    t_slow.append(end-start)

    ks_fast.extend([ d[1] for d in G_fast.degree()])
    ks_slow.extend([ d[1] for d in G_slow.degree()])

print("needed t =", np.mean(t_fast), "s per run for the fast algorithm")
print("needed t =", np.mean(t_slow), "s per run for the slow algorithm")

k_f, P_f = get_counter_hist(ks_fast)
k_s, P_s = get_counter_hist(ks_slow)
k_t, P_t = get_degree_distribution(N, k_over_2, beta, kmax=max(k_f.max(), k_s.max()))

pl.plot(k_f, P_f, 'o', label='fast algorithm',mfc='w')
pl.plot(k_s, P_s, 's', label='slow algorithm',mfc='w')
pl.plot(k_t, P_t, '-', label='theory',lw=1)
pl.legend()

pl.xlabel('node degree $k$')
pl.ylabel('probability $P_k$')

pl.show()
