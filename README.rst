smallworld
==========

| Generate and analyze small-world networks according to the revised
Watts-Strogatz model where the randomization
| at *β* = 1 is truly equal to the Erdős-Rényi network model.

The Watts-Strogatz model each node is asked to rewire its *k*/2
rightmost edges with probality *β*. This means thaeach node has halways
minimum degree *k*/2. Also, at *β* = 1, each edge has been rewired.
Hence the probability of it existing is 0, contrary to the ER model.

In the adjusted model, each pair of nodes is connected with a certain
connection probability. If the lattice distance between the potentially
connected nodes is d(i,j) <= *k*/2 then they are connected with
short-range probability ``p_S = k / (k + β (N-1-k))``, otherwise they're
connected with long-range probability ``p_L = β * p_S``.

Install
-------

::

    pip install smallworld

Beware: ``smallworld`` only works with Python 3!

Example
-------

In the following example you can see how to generate and draw according
to the model described above.

.. code:: python

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

|visualization example|

.. |visualization example| image:: https://github.com/benmaier/smallworld/raw/master/sandbox/small_worlds.png
