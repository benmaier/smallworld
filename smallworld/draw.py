"""
Methods to draw those networks.
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as pl

#def plot_edge(ax,N,u,v,phis,color=None):
colors =  [
            '#666666',
            '#1b9e77',
            '#e7298a'
            ]

mpl.rcParams['font.size'] = 9
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['axes.titlesize'] = 'medium'
#mpl.rcParams['xtick.labelsize'] = 'small'
#mpl.rcParams['ytick.labelsize'] = 'small'
mpl.rcParams['xtick.labelsize'] = 'medium'
mpl.rcParams['ytick.labelsize'] = 'medium'
mpl.rcParams['lines.markersize'] = 4
mpl.rcParams['lines.linewidth'] = 1.0


def bezier_curve(P0,P1,P2,n=20):

    t = np.linspace(0,1,20)
    B = np.zeros((n,2))
    for part in range(n):
        t_ = t[part]

        B[part,:] = (1-t_)**2 * P0 + 2*(1-t_)*t_*P1+t_**2*P2

    return B

def is_shortrange(i,j,N,k_over_2):
    distance = np.abs(i-j)

    return distance <= k_over_2 or N-distance <= k_over_2

def draw_network(G, k_over_2, R=10,focal_node=None, ax=None,markersize=None,linewidth=1.0,linkcolor=None):
    """
    Draw a small world network.

    Parameters
    ==========
    G : network.Graph
        The network to be drawn
    R : float, default : 10.0
        Radius of the circle
    focal_node : int, default : None
        If this is given, highlight edges
        connected to this node.
    ax : matplotlib.Axes, default : None
        Axes to draw on. If `None`, will generate
        a new one.

    Returns
    =======
    ax : matplotlib.Axes
    """

    G_ = G.copy()


    if ax is None:
        fig, ax = pl.subplots(1,1,figsize=(3,3))

    focal_alpha = 1

    if focal_node is None:
        non_focal_alpha = 1
        focal_lw = linewidth*1.0
        non_focal_lw = linewidth*1.0
    else:
        non_focal_alpha = 0.6
        focal_lw = linewidth*1.5
        non_focal_lw = linewidth*1.0


    N = G_.number_of_nodes()

    phis = 2*np.pi*np.arange(N)/N + np.pi/2

    x = R * np.cos(phis)
    y = R * np.sin(phis)

    points = np.zeros((N,2))
    points[:,0] = x
    points[:,1] = y
    origin = np.zeros((2,))

    col = list(colors)
    if linkcolor is not None:
        col[0] = linkcolor


    ax.axis('equal')
    ax.axis('off')

    if focal_node is None:
        edges = list(G_.edges(data=False))
    else:
        focal_edges = [ e for e in G_.edges(data=False) if focal_node in e]
        G_.remove_edges_from(focal_edges)
        edges = list(G_.edges) + focal_edges

    for i, j in edges:

        phi0 = phis[i]
        phi1 = phis[j]
        dphi = phi1 - phi0

        if dphi > np.pi:
            dphi = 2*np.pi - dphi
            phi0, phi1 = phi1, phi0
            phi1 += 2*np.pi

        distance = np.abs(i-j)

        if i == focal_node or j == focal_node:
            if distance <= k_over_2 or N-distance <= k_over_2:
                this_color = col[2]
            else:
                this_color = col[1]
            this_alpha = focal_alpha
            this_lw = focal_lw
        else:
            this_color = col[0]
            this_alpha = non_focal_alpha
            this_lw = non_focal_lw

        if distance == 1 or N-distance == 1:

            these_phis = np.linspace(phi0, phi1,20)
            these_x = R * np.cos(these_phis)
            these_y = R * np.sin(these_phis)

        else:
            if is_shortrange(i,j,N,k_over_2):
                ophi = phi0 + dphi/2
                o = np.array([
                            0.6*R*np.cos(ophi),
                            0.6*R*np.sin(ophi),
                    ])
            else:
                o = origin
            B = bezier_curve(points[i],o,points[j],n=20)
            these_x = B[:,0]
            these_y = B[:,1]


        ax.plot(these_x, these_y,c=this_color,alpha=this_alpha,lw=this_lw)

    ax.plot(x,y,'o',c='k',mec='#ffffff',ms=markersize)

    return ax

if __name__ == "__main__":
    from smallworld import get_smallworld_graph

    N = 50
    k_over_2 = 2
    beta = 0.01

    focal_node = 0

    G = get_smallworld_graph(N, k_over_2, beta)
    draw_network(G,k_over_2,focal_node=0)

    pl.show()
