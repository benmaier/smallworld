from smallworld.draw import draw_network
from smallworld import get_smallworld_graph

import matplotlib.pyplot as pl

# define network parameters
N = 21
k_over_2 = 2
betas = [0, 0.025, 1.0]
labels = [ r'$\beta=0$', r'$\beta=0.025$', r'$\beta=1$']

focal_node = 0

fig, ax = pl.subplots(1,3,figsize=(9,3))


# scan beta values
for ib, beta in enumerate(betas):

    # generate small-world graphs and draw
    G = get_smallworld_graph(N, k_over_2, beta)
    draw_network(G,k_over_2,focal_node=focal_node,ax=ax[ib])

    ax[ib].set_title(labels[ib],fontsize=11)

# show
pl.subplots_adjust(wspace=0.3)
pl.show()
