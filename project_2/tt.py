import argparse
import itertools
import math
import textwrap
import time
from pathlib import Path

import cv2
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
        default="./BOE-XUDLNKG.csv",
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
    parser.add_argument(
        "--is-random",
        action="store_true",
        help="Random permutation",
    )

    parser.add_argument(
        "--use-img-dim",
        action="store_true",
        help="Use image dimensions",
    )

    args = parser.parse_args(args)

    return args


def reconstruct(A, G, axes=([-1], [0])):
    A_hat = G[0]
    for c in G[1:]:
        if c.ndim == 1:  # only for SVD!
            c = sla.diagsvd(c, c.shape[0], c.shape[0])

        A_hat = np.tensordot(A_hat, c, axes=axes)

    err = sla.norm((A - A_hat).ravel())
    compression_rate = int(A.size / sum(g.size for g in G))

    return A_hat, err, compression_rate


def svd(A, eps=0.05, ax=None):
    start = time.monotonic()

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

    elapsed = time.monotonic() - start

    i = 0
    C = A
    if ax is not None:
        ax.plot(
            s,
            label=f"$C_{i}$ {C.shape[0]}x{C.shape[1]}",
        )
        ax.set_yscale("log")
        ax.set_title(f"s.v. SVD ({A.shape})")
        ax.grid(ls="--")
        ax.legend()

    G = [U[:, :rk], s[:rk], Vh[:rk]]
    A_hat, err, compression_rate = reconstruct(A, G)
    err /= A_norm
    assert err < eps, f"Not enough precision ({err}). Increase the number of singluar values!"

    return A_hat, G, err, compression_rate, elapsed


def hosvd(A, eps=0.05, ax=None):
    start = time.monotonic()
    d = A.ndim
    A_norm = sla.norm(A.ravel())
    eps_A = eps * A_norm
    delta = eps_A / math.sqrt(d - 1)

    Us = []
    for i, nk in enumerate(A.shape):
        perm = list(range(d))
        perm[0], perm[i] = i, 0
        C = A.transpose(perm).reshape(nk, -1)
        U, s, _, = sla.svd(C, full_matrices=False)
        if ax is not None:
            ax.plot(
                s,
                label=f"$C_{i}$ {C.shape[0]}x{C.shape[1]}",
            )
            ax.set_yscale("log")
        Us.append(U)

    rk = 1
    while True:
        G = [
            U[:, :rk]
            for U in Us
        ]

        core = A
        for c in G:
            core = np.tensordot(core, c, axes=([0], [0]))

        G = [core] + G

        A_hat, err, compression_rate = reconstruct(A, G, axes=([0], [1]))

        if err >= delta:
            rk += 1
            print(
                f"Not enough precision for HOSVD {err}."
                f"The number of singular values  is increased to {rk}!"
            )
        else:
            break

    elapsed = time.monotonic() - start

    if ax is not None:
        ax.set_title(f"s.v. HOSVD ({A.shape})")
        ax.grid(ls="--")
        ax.legend()

    A_hat, err, compression_rate = reconstruct(A, G, axes=([0], [1]))
    err /= A_norm
    assert err < eps, f"Not enough precision ({err}). Increase the number of singluar values!"

    return A_hat, G, err, compression_rate, elapsed


def tt_svd(A, eps=0.05, ax=None):
    """
    https://sites.pitt.edu/~sjh95/related_papers/tensor_train_decomposition.pdf
    Algorithm 1: TT-svd
    """

    start = time.monotonic()
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
                label=f"$C_{i}$ {C.shape[0]}x{C.shape[1]}: {s.shape[0]}",
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

    elapsed = time.monotonic() - start

    if ax is not None:
        ax.set_title(f"s.v. TT-SVD ({A.shape})")
        ax.grid(ls="--")
        ax.legend()

    G.append(C)

    A_hat, err, compression_rate = reconstruct(A, G)
    err /= A_norm
    assert err < eps, f"Not enough precision ({err}). Increase the number of singluar values!"

    return A_hat, G, err, compression_rate, elapsed


def find_best_factorization(N, d, is_random=False):
    n = int(math.pow(N, 1 / d))

    eps = 10 * int(math.sqrt(n))
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
    if is_random:
        i = np.random.randint(0, len(res))
    else:
        pw = cdist(
            res,
            [[n] * d],
            metric="cityblock",
        )
        i, _ = np.unravel_index(pw.argmin(), pw.shape)

    js = res[i]

    if not is_random:
        js = -np.sort(-js)
        #js = np.sort(js)

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


def get_label(method, G, err, cr, t):
    r = max(min(g.shape) for g in G)
    label = f"{method} $\epsilon$:{err:.3f} $c_r$:{cr} $t$:{t:.3} $r$:{r}"

    return label


def time_series(data_path, usecols, N, tol=0.05, is_random=False):
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
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 6))
    fig.suptitle(f"Compression for time series x: {len(x)}x1 with tolerance {tol}")

    axes["d"].plot(
        x,
        label="$x$",
    )

    n1, n2 = find_best_factorization(N, 2, is_random=is_random)
    X = x.reshape(n1, n2)
    X_svd, G_svd, error_svd, cr_svd, t_svd = svd(X, eps=tol, ax=axes["a"])
    x_svd = X_svd.reshape(-1)
    axes["d"].plot(
        x_svd,
        label=get_label("$x_{svd}$", G_svd, error_svd, cr_svd, t_svd),
    )

    n1, n2, n3 = find_best_factorization(N, 3, is_random=is_random)
    X = x.reshape(n1, n2, n3)
    X_hosvd, G_hosvd, error_hosvd, cr_hosvd, t_hosvd = hosvd(X, eps=tol, ax=axes["b"])
    x_hosvd = X_hosvd.reshape(-1)
    axes["d"].plot(
        x_hosvd,
        label=get_label("$x_{hosvd}$", G_hosvd, error_hosvd, cr_hosvd, t_hosvd),
    )

    X_tt, G_tt, error_tt, cr_tt, t_tt = tt_svd(X, eps=tol, ax=axes["c"])
    x_tt = X_tt.reshape(-1)

    ax = axes["d"]
    ax.plot(
        x_tt,
        label=get_label("$x_{tt}$", G_tt, error_tt, cr_tt, t_tt),
    )
    ax.set_title("Blessing of dimensionality of time series")
    ax.legend()
    ax.grid(ls="--")

    fig.text(
        0.5,
        0.005,
        "$\epsilon$ - ${\|x - \hat{x}\|}_F$, $c_r$ - compression rate, $t$ - time $r$ - rank",
        ha="center",
    )
    fig.tight_layout()

    data_path = Path(data_path)
    out_path = f"{data_path.stem}_comp.png"
    if is_random:
        out_path = f"{data_path.stem}_comp_r.png"

    fig.savefig(out_path)
    plt.show()
    plt.close(fig)


def image(data_path, tol=0.05, is_random=False, use_img_dim=False):
    x = cv2.imread(data_path)
    m, n, c = x.shape
    x = x.ravel()
    N = len(x)

    mosaic = textwrap.dedent(
        '''
        .abc
        defg
        '''
    )
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 6))
    fig.suptitle(f"Compression for image x: {m}x{n}x{c} with tolerance {tol}")

    ax = axes["d"]
    ax.set_title("Original image")
    ax.imshow(
        x.reshape(m, n, c),
    )
    ax.axis("off")

    if use_img_dim:
        n1, n2 = m, n * c
    else:
        n1, n2 = find_best_factorization(N, 2, is_random=is_random)

    X = x.reshape(n1, n2)
    X_svd, G_svd, error_svd, cr_svd, t_svd = svd(X, eps=tol, ax=axes["a"])
    x_svd = X_svd.reshape(-1)
    ax = axes["e"]
    ax.set_title(get_label("$x_{svd}$", G_svd, error_svd, cr_svd, t_svd))
    ax.imshow(
        x_svd.reshape(m, n, c).astype("uint8"),
    )
    ax.axis("off")

    if use_img_dim:
        n1, n2, n3 = m, n, c
    else:
        n1, n2, n3 = find_best_factorization(N, 3, is_random=is_random)
    X = x.reshape(n1, n2, n3)
    X_hosvd, G_hosvd, error_hosvd, cr_hosvd, t_hosvd = hosvd(X, eps=tol, ax=axes["b"])
    x_hosvd = X_hosvd.reshape(-1)
    ax = axes["f"]
    ax.set_title(get_label("$x_{hosvd}$", G_hosvd, error_hosvd, cr_hosvd, t_hosvd))
    ax.imshow(
        x_hosvd.reshape(m, n, c).astype("uint8"),
    )
    ax.axis("off")

    X_tt, G_tt, error_tt, cr_tt, t_tt = tt_svd(X, eps=tol, ax=axes["c"])
    x_tt = X_tt.reshape(-1)

    ax = axes["g"]
    ax.set_title(get_label("$x_{tt}$", G_tt, error_tt, cr_tt, t_tt))
    ax.imshow(
        x_tt.reshape(m, n, c).astype("uint8"),
    )
    ax.axis("off")

    fig.text(
        0.5,
        0.005,
        "$\epsilon$ - ${\|x - \hat{x}\|}_F$, $c_r$ - compression rate, $t$ - time $r$ - rank",
        ha="center",
    )
    fig.tight_layout()

    data_path = Path(data_path)
    out_path = f"{data_path.stem}_comp.png"
    if is_random:
        out_path = f"{data_path.stem}_comp_r.png"

    if use_img_dim:
        out_path = f"{data_path.stem}_comp_i.png"

    fig.savefig(out_path)
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
            is_random=args.is_random,
        )
    elif args.dim == 2:
        image(
            data_path=args.data_path,
            tol=args.tol,
            is_random=args.is_random,
            use_img_dim=args.use_img_dim,
        )


if __name__ == "__main__":
    main()
