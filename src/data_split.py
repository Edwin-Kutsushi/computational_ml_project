from scipy.sparse import coo_matrix


def train_test_split_sparse(R, test_size=0.2, seed=42):
    """
    Split a sparse user-item matrix into train and test sets.

    Parameters
    ----------
    R : scipy.sparse matrix (CSR or COO)
        Full sparse user-item rating matrix.

    test_size : float
        Proportion of observed entries to include in the test set.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    R_train : CSR matrix
        Training sparse matrix.

    R_test : CSR matrix
        Test sparse matrix.
    """

    # convert to COO format for easy indexing
    R_coo = R.tocoo()

    # number of observed ratings
    n = len(R_coo.data)

    # generate random permutation of indices
    import numpy as np
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    # determine split point
    test_size_int = int(test_size * n)

    test_idx = perm[:test_size_int]
    train_idx = perm[test_size_int:]

    # build training matrix
    R_train = coo_matrix(
        (R_coo.data[train_idx],
         (R_coo.row[train_idx], R_coo.col[train_idx])),
        shape=R_coo.shape
    ).tocsr()

    # build test matrix
    R_test = coo_matrix(
        (R_coo.data[test_idx],
         (R_coo.row[test_idx], R_coo.col[test_idx])),
        shape=R_coo.shape
    ).tocsr()

    return R_train, R_test