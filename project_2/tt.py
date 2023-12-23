import argparse
import itertools
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.spatial.distance import cdist


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/BOE-XUDLNKG.csv",
        help="Paht to data file",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help="Task dimenstion (1 - time series, 2 - matrix, ...)",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="Telerance",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=10332,
        help="Number of samples in time series",
    )
    parser.add_argument(
        "--usecols",
        type=str,
        nargs="+",
        default=["Value"],
        help="Columns to use in dataframe for time series",
    )

    args = parser.parse_args(args)

    return args


def reconstruct(A, G, axes=([-1], [0])):
    A_hat = G[0]
    for c in G[1:]:
        if c.ndim == 1:  # only for SVD!
            c = sla.diagsvd(c, c.shape[0], c.shape[0])

        A_hat = np.tensordot(A_hat, c, axes=axes)

    err = sla.norm((A - A_hat).ravel()) / sla.norm(A.ravel())
    compression_rate = int(A.size / sum(g.size for g in G))

    return A_hat, err, compression_rate


def svd(A, eps=0.05, ax=None):
    U, s, Vh = sla.svd(A, full_matrices=False)
    A_norm = sla.norm(A.ravel())
    eps_A = eps * A_norm

    rk = 1
    while True:
        A_hat = U[:, :rk] @ (s[:rk, None] * Vh[:rk])
        err = sla.norm((A - A_hat).ravel())
        if err >= eps_A:
            rk += 1
            print(
                f"Not enough precision for SVD {err}."
                f"The number of singular values  is increased to {rk}!"
            )
        else:
            break

    i = 0
    C = A
    if ax is not None:
        ax.plot(
            s,
            label=f"$C_{i}$ {C.shape[0]}x{C.shape[1]}: {s.shape[0]} singular values",
        )
        ax.set_yscale("log")
        ax.set_title(f"singular values SVD ({A.shape})")
        ax.grid(ls="--")
        ax.legend()

    G = [U[:, :rk], s[:rk], Vh[:rk]]
    A_hat, err, compression_rate = reconstruct(A, G)
    assert err < eps_A, f"Not enough precision ({err}). Increase the number of singluar values!"

    return A_hat, G, err, compression_rate


def hosvd(A, eps=0.05, ax=None):
    d = A.ndim
    A_norm = sla.norm(A.ravel())
    eps_A = eps * A_norm

    while True:
        rk = 1
        G = []
        for i, nk in enumerate(A.shape):
            perm = list(range(d))
            perm[0], perm[i] = i, 0
            C = A.transpose(perm).reshape(nk, -1)
            U, s, _, = sla.svd(C, full_matrices=False)
            if ax is not None:
                ax.plot(
                    s,
                    label=f"$C_{i}$ {C.shape[0]}x{C.shape[1]}: {s.shape[0]} singular values",
                )
                ax.set_yscale("log")

            G.append(U[:, :rk])

        core = A
        for c in G:
            core = np.tensordot(core, c, axes=([0], [0]))

        G = [core] + G

        A_hat, err, compression_rate = reconstruct(A, G, axes=([0], [1]))

        if err >= eps_A:
            rk += 1
            print(
                f"Not enough precision for HOSVD {err}."
                f"The number of singular values  is increased to {rk}!"
            )
        else:
            break

    if ax is not None:
        ax.set_title(f"singular values HOSVD ({A.shape})")
        ax.grid(ls="--")
        ax.legend()

    A_hat, err, compression_rate = reconstruct(A, G, axes=([0], [1]))
    assert err < eps_A, f"Not enough precision ({err}). Increase the number of singluar values!"

    return A_hat, G, err, compression_rate


def tt_svd(A, eps=0.05, ax=None):
    """
    https://sites.pitt.edu/~sjh95/related_papers/tensor_train_decomposition.pdf
    Algorithm 1: TT-svd
    """

    d = A.ndim
    A_norm = sla.norm(A.ravel())
    eps_A = eps * A_norm
    delta = eps_A / math.sqrt(d - 1)

    C = A.copy()
    r0 = 1

    G = []
    for i, nk in enumerate(A.shape[:-1]):
        C = C.reshape(r0 * nk, -1)

        U, s, Vh = sla.svd(C, full_matrices=False)
        if ax is not None:
            ax.plot(
                s,
                label=f"$C_{i}$ {C.shape[0]}x{C.shape[1]}: {s.shape[0]} singular values",
            )
            ax.set_yscale("log")

        S = sla.diagsvd(s, Vh.shape[0], Vh.shape[0])

        rk = 1
        while True:
            err = sla.norm((C - U[:, :rk] @ (S @ Vh)[:rk]).ravel())
            if err >= delta:
                rk += 1
                print(
                    f"Not enough precision for TT-SVD {err}."
                    f"The number of singular values  is increased to {rk}!"
                )
            else:
                break

        G.append(U[:, :rk].reshape(r0, nk, rk))
        C = (S @ Vh)[:rk]

        r0 = rk

    if ax is not None:
        ax.set_title(f"singular values TT-SVD ({A.shape})")
        ax.grid(ls="--")
        ax.legend()

    G.append(C)

    A_hat, err, compression_rate = reconstruct(A, G)
    assert err < eps_A, f"Not enough precision ({err}). Increase the number of singluar values!"

    return A_hat, G, err, compression_rate


def find_best_factorization(N, d, is_random=False):
    n = int(math.pow(N, 1 / d))

    eps = 2 * int(math.sqrt(n))
    sp = [
        range(max(1, n - eps), min(N + 1, n + eps + 1))
        for _ in range(d - 1)
    ]

    res = []
    for js in itertools.product(*sp):
        jsp = np.prod(js)
        if N % jsp == 0:
            res.append(js)

    if len(res) == 0:
        print(f"Not found factorization of {N} increase search space!")
        raise RuntimeError(f"Not found factorization of {N} increase search space!")

    res = np.array(res)
    res = np.hstack(
        [
            res,
            N // np.prod(res, axis=1, keepdims=True)
        ],
    )
    pw = cdist(
        res,
        [[n] * d],
        metric="cityblock",
    )
    i, _ = np.unravel_index(pw.argmin(), pw.shape)
    js = res[i]
    js = -np.sort(-js)

    if is_random:
        np.random.shuffle(js)

    return js


def is_prime(n):
    if n <= 1:
        return False

    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False

    return True


def find_best(N):
    if is_prime(N):
        N -= 1

    while True:
        try:
            ns = find_best_factorization(N, 3)
        except RuntimeError:
            N -= 1
            continue

        if is_prime(N) or max(ns) > N // 4:
            continue

        N -= 1
        break

    return N


def get_label(method, G, err, cr):
    label = (
        f"{method}"
        f", err:{err:.3} cr:{cr}"
        f", {','.join('x'.join(map(str, g.shape)) for g in G)}"
    )

    return label


def time_series(data_path, usecols, N, tol=0.05):
    x = pd.read_csv(
        data_path,
        usecols=usecols,
        nrows=N,
    )
    x = x.dropna()
    N = len(x)

    x = x.values.ravel()[:N]
    #x -= x.mean()

    mosaic = textwrap.dedent(
        '''
        abc
        ddd
        '''
    )
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 8))
    fig.suptitle(f"Compression for time series x: {len(x)}x1")

    axes["d"].plot(
        x,
        label="$x$",
    )

    n1, n2 = find_best_factorization(N, 2)
    X = x.reshape(n1, n2)
    X_svd, G_svd, error_svd, cr_svd = svd(X, eps=tol, ax=axes["a"])
    x_svd = X_svd.reshape(-1)
    axes["d"].plot(
        x_svd,
        label=get_label("$x_{svd}$  ", G_svd, error_svd, cr_svd),
    )

    n1, n2, n3 = find_best_factorization(N, 3)
    X = x.reshape(n1, n2, n3)
    X_hosvd, G_hosvd, error_hosvd, cr_hosvd = hosvd(X, eps=tol, ax=axes["b"])
    x_hosvd = X_hosvd.reshape(-1)
    axes["d"].plot(
        x_hosvd,
        label=get_label("$x_{hosvd}$", G_hosvd, error_hosvd, cr_hosvd),
    )

    X_tt, G_tt, error_tt, cr_tt = tt_svd(X, eps=tol, ax=axes["c"])
    x_tt = X_tt.reshape(-1)

    ax = axes["d"]
    ax.plot(
        x_tt,
        label=get_label("$x_{tt}$   ", G_tt, error_tt, cr_tt),
    )
    ax.set_title("Blessing of dimensionality of time series")
    ax.legend()
    ax.grid(ls="--")

    fig.text(
        0.5,
        0.005,
        "err=${\|x - \hat{x}\|}_F$, cr=compression rate",
        ha="center",
    )
    fig.tight_layout()

    data_path = Path(data_path)
    fig.savefig(f"{data_path.stem}_comp.png")
    plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    if args.dim == 1:
        time_series(
            data_path=args.data_path,
            usecols=args.usecols,
            N=args.n_samples,
            tol=args.tol,
        )


if __name__ == "__main__":
    main()
