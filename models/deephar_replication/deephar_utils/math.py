import numpy as np
from scipy.stats import multivariate_normal

def linspace_2d(nb_rols, nb_cols, dim=0):

    def _lin_sp_aux(size, nb_repeat, start, end):
        linsp = np.linspace(start, end, num=size)
        x = np.empty((nb_repeat, size), dtype=np.float32)

        for d in range(nb_repeat):
            x[d] = linsp

        return x

    if dim == 1:
        return (_lin_sp_aux(nb_rols, nb_cols, 0.0, 1.0)).T
    return _lin_sp_aux(nb_cols, nb_rols, 0.0, 1.0)

def normalpdf2d(numbins, xmean, ymean, var):
    lin = np.linspace(0, numbins-1, numbins)

    # Produce a gaussian in X and in Y
    x = multivariate_normal.pdf(lin, mean=xmean, cov=var)
    x = x.reshape((1, numbins)).repeat(numbins, axis=0)
    y = multivariate_normal.pdf(lin, mean=ymean, cov=var)
    y = y.reshape((numbins, 1)).repeat(numbins, axis=1)
    g = x * y

    if g.sum() > 1e-7: #previously was K.epsilon(), which seems to default at 1e-7
        return g / g.sum()

    return np.zeros(g.shape)

