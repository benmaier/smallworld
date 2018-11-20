import matplotlib.pyplot as pl
import numpy as np

from smallworld.theory import get_effective_medium_eigenvalue_gap, get_effective_medium_eigenvalue_gap_from_matrix


N = 100
k_over_2 = 2
betas = np.logspace(-3,0,10)

from_matrix = np.zeros_like(betas)

for ib, beta in enumerate(betas):
    this_val = get_effective_medium_eigenvalue_gap_from_matrix(N, k_over_2, beta)
    from_matrix[ib] = this_val

theory = get_effective_medium_eigenvalue_gap(N,k_over_2,betas)

pl.plot(betas, 1./from_matrix,'s',c='k',mfc='w',label='from matrix')
pl.plot(betas, 1./theory,'-',c='k',label='theory')

pl.xscale('log')

pl.show()
