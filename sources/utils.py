import numpy as np


def make_binning_operator(x, x_min, x_max, n_bin, weights=None,
                          binning="linear", squared=False, namaster=False):
    '''
    This function defines the binning operator for the data x.
    '''

    if binning == "linear":
        bin_edges = np.linspace(x_min, x_max, n_bin+1, endpoint=True)
    elif binning == "log":
        bin_edges = np.geomspace(x_min, x_max, n_bin+1, endpoint=True)
    else:
        raise ValueError(f"Binning type {binning} not supported.")

    w = np.ones_like(x, dtype=np.float64) if weights is None else weights

    if not namaster:
        B = np.zeros((n_bin, len(x)))
        for i in range(n_bin):
            M = np.logical_and(bin_edges[i] <= x, x < bin_edges[i+1])

            if squared:
                B[i, M] = w[M]**2/np.sum(w[M])**2
            else:
                B[i, M] = w[M]/np.sum(w[M])
        return B
    else:
        B = np.full(len(x), fill_value=-1, dtype=int)

        for i in range(n_bin):
            M = np.logical_and(bin_edges[i] <= x, x < bin_edges[i+1])
            B[M] = i
        return B
