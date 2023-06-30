import numpy as np
import argparse
import torch
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import pairwise, pairwise_distances


def recall_at(src_npy, tgt_npy, topks=[1]):
    if isinstance(topks, int):
        topks = [topks]
    n_src = src_npy.shape[0]
    n_tgt = tgt_npy.shape[0]
    if n_src != n_tgt:
        raise ValueError(f"(src, tgt) has different length ({n_src}, {n_tgt})")
    pdist = pairwise_distances(src_npy, tgt_npy, metric="cosine")
    gold_dist = np.diag(pdist)
    pred_indices = pdist.argsort(axis=1)
    recalls = {}
    for k in topks:
        max_dist = pdist[np.arange(n_src), pred_indices[:,k - 1]]
        recall = (gold_dist <= max_dist).sum() / n_src
        recalls[k] = recall
    return recalls


def train_svd(X, n_components=None, n_iter=5, random_state=None):
    """
    Args:
        X (numpy.ndarray) :
            Shape = (n_data, n_features)
        n_components (int, optional) :
            If `n_components` is None, it set `n_components` = min(n_data, n_features) - 1
        n_iter (int) :
            Number of iteration. It is passed to `randomized_svd`
        random_state (int, optional)
    Returns:
        U (numpy.ndarray) : Shape is (n_data, n_components)
        Sigma (numpy.ndarray) : Shape is (n_components,)
        VT (numpy.ndarray) : Shape is (n_components, n_features)
    """
    if n_components is None:
        n_components = min(X.shape) - 1

    if (random_state == None) or isinstance(random_state, int):
        random_state = check_random_state(random_state)

    n_features = X.shape[1]
    if n_components >= n_features:
        raise ValueError(
            "n_components must be < n_features;"
            " got %d >= %d" % (n_components, n_features)
        )

    U, Sigma, VT = randomized_svd(
        X, n_components, n_iter=n_iter, random_state=random_state
    )
    return U, Sigma, VT


def train_basis(X, n_iter=5, random_state=None):
    """
    Args:
        X (numpy.ndarray) :
            Shape = (n_data, n_features)
        n_iter (int) :
            Number of iteration. It is passed to `randomized_svd`
        random_state (int, optional)
    Returns:
        truncated_basis (numpy.ndarray) :
            Shape = (n_components, n_features)
            where `n_components` is `min(n_data, n_features) - 1`
            and each row is orthonomal
    Examples:
        >>> X = np.random.random_sample((1000, 64))
        >>> basis = train_basis(X)
        >>> (basis ** 2).sum(axis=0)
        $ array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,  ...])
    """
    _, _, VT = train_svd(X, n_iter=n_iter, random_state=random_state)
    return VT.T


def debias(X, B):
    """
    Args:
        X (numpy.ndarray) :
            Shape = (n_data, n_features)
        B (numpy.ndarray) :
            Shape = (n_features, n_basis)
    Returns:
        X (numpy.ndarray) :
            Debiased X' = X - <<X, B>, B^T>
    Examples:
        >>> X = np.random.random_sample((1000, 64))
        >>> basis = train_basis(X)
        >>> X_ = debias(X, basis[:, :3])
        >>> np.set_printoptions(precision=3)
        >>> np.dot(X_, basis[:, :3]).sum(axis=1)
        $ array([-3.473e-15, -3.322e-15, -3.540e-15, -4.038e-15, -2.266e-15,
                 -2.216e-15, -2.829e-15, -3.780e-15, -1.289e-15, -2.700e-15,
                 -2.836e-15, -4.228e-15, -2.794e-15, -2.052e-15, -1.589e-15,
                 ...
                ])
    """
    X_ = np.dot(np.dot(X, B), B.T)
    return X - X_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_npy_path", type=str, required=True)
    parser.add_argument("-t", "--tgt_npy_path", type=str, required=True)
    args = parser.parse_args()

    src_npy = torch.load(args.src_npy_path)
    tgt_npy = torch.load(args.tgt_npy_path)

    src_npy = src_npy.cpu().detach().numpy()
    tgt_npy = tgt_npy.cpu().detach().numpy()

    # Train basis vectors
    src_basis = train_basis(src_npy)
    tgt_basis = train_basis(tgt_npy)
    
    # Eliminate selected number of components (language bias)
    print(" | ".join(f"{v}" for v in ["num_of_components", "Cosine distance (mean, std)", "Recall @1"]))
    for n in [0,1,2,3,5,10,20,30]:
        src_basis_ = debias(src_npy, src_basis[:, :n])
        tgt_basis_ = debias(tgt_npy, tgt_basis[:, :n])

        np.set_printoptions(precision=3)
        # Compute pairwise distances (Cosine, Euclidean)
        cosine_distances = pairwise.paired_distances(src_basis_,tgt_basis_, metric="cosine")    
        print(" | ".join(f"{v}" for v in [
            f"{n}",
            f"({np.average(cosine_distances)}, {np.std(cosine_distances)})",
            f"{recall_at(src_basis_, tgt_basis_)}"
            ])
        )


if __name__ == "__main__":
    main()