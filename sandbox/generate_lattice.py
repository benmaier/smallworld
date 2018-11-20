import matplotlib.pyplot as pl
import numpy as np

N = 31

x = np.arange(N)

for y in range(N):
    pl.plot(x, np.ones_like(x)*y,'o',mfc='w',c='k',ms=3,mew=0.5)

pl.axis('square')
pl.axis('off')

pl.savefig('grid.pdf')
pl.show()


