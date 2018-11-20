# -*- coding: utf-8 -*-
"""
`smallworld` offers routines to generate modified small-world-networks
which interpolate between a k-nearest-neighbor-lattice and an Erdos-Renyi
network (contrary to the traditional Watts-Strogatz model).
"""

from .metadata import __version__
from .generate import get_smallworld_graph
