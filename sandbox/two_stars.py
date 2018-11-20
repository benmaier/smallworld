import matplotlib.pyplot as pl
import numpy as np
from collections import Counter

from smallworld import get_smallworld_graph
from smallworld.theory import expected_number_of_unique_two_stars_per_node
from smallworld.theory import expected_number_of_unique_triangles_per_node
from smallworld.tools import get_number_of_unique_two_stars 
from smallworld.tools import get_number_of_unique_triangles_for_each_node 

from time import time

N = 15
k_over_2 = 2

betas = np.logspace(-3,0,10)

N_meas = 10000

mean_two_stars = []
mean_triangles = []

for ib, beta in enumerate(betas):
    print(ib+1,"/",len(betas))
    two_stars = []
    triangles = []
    for meas in range(N_meas):

        G = get_smallworld_graph(N, k_over_2, beta, use_slow_algorithm=False)
        two_stars.extend(list(get_number_of_unique_two_stars(G)))
        triangles.extend(list(get_number_of_unique_triangles_for_each_node(G)))

    mean_two_stars.append(np.mean(two_stars))
    mean_triangles.append(np.mean(triangles))


mean_two_stars = np.array(mean_two_stars)
mean_triangles = np.array(mean_triangles)

pl.figure()
pl.plot(betas, mean_two_stars,'s',mfc='w')
S = expected_number_of_unique_two_stars_per_node(N, k_over_2, betas)
pl.plot(betas, S,lw=1,c='k')

pl.xscale('log')
#pl.yscale('log')
pl.xlabel(r'$\beta$')
pl.ylabel('expected number of two stars per node')

pl.figure()
pl.plot(betas, mean_triangles,'s',mfc='w')
T = expected_number_of_unique_triangles_per_node(N, k_over_2, betas)
pl.plot(betas, T,lw=1,c='k')

pl.xscale('log')
pl.yscale('log')
pl.xlabel(r'$\beta$')
pl.ylabel('expected number of triangles per node')

pl.figure()
pl.plot(betas, mean_triangles/mean_two_stars,'s',mfc='w')
pl.plot(betas, T/S,lw=1,c='k')
pl.xscale('log')
pl.yscale('log')
pl.xlabel(r'$\beta$')
pl.ylabel('clustering coefficient')

pl.show()
