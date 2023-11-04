import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as sla


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        required=True,
        type=str,
        help="Path to train data",
    )
    parser.add_argument(
        "--test-path",
        required=False,
        type=str,
        help="Path to test data",
    )
    parser.add_argument(
        "--site-id",
        required=False,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train-year",
        required=False,
        type=int,
        default=1991,
    )
    parser.add_argument(
        "--test-year",
        required=False,
        type=int,
        default=2005,
    )

    args = parser.parse_args(args)

    return args


def read_water_supply(data_path, site_id=None):
    df = pd.read_csv(data_path)

    if site_id is not None:
        df = df[df.site_id.isin([site_id])]

    df = df.set_index(
        pd.to_datetime(
            df.year.astype("string") + "/" + df.month.astype("string"),
            format="%Y/%m",
        )
    ).sort_index()

    return df


def train_dev_split(df, train_year, test_year, test_df=None):
    if test_df is not None:
        test_year = test_df.year.min()
        train_df = df[str(train_year):str(test_year - 1)]
        val_df = df[str(test_year):]
        val_df = pd.concat(
            [
                val_df,
                test_df,
            ]
        ).sort_index()
    else:
        train_df = df[str(train_year):str(test_year - 1)]
        val_df = df[str(test_year):]

    return train_df, val_df


def get_data(volume, window_len=3):
    # volume: [n]
    # [
    #   [0, 1, 2],
    #   [1, 2, 3],
    #   [2, 3, 4],
    #   ...
    # ]
    n = len(volume)
    i = np.arange(window_len + 1)
    n_samples = n - window_len - 1
    i = np.tile(i, (n_samples, 1))
    j = np.arange(n_samples)[:, None]
    ij = i + j
    x = volume[ij]
    y = x[:, -1]
    x = x[:, :-1]

    return x, y



def predict_linear_model(X_val, w):
    y_hat = X_val @ w

    return y_hat


def train_linear_model(X_train, y_train):
    # X w = y => w = pinv(X) y
    w, res, rank, s = np.linalg.lstsq(X_train, y_train)

    return w


def train_qr_model(X_train, y_train):
    # X w = y => QR w = y => w = Rm Qt y
    Q, R = np.linalg.qr(X_train)
    w = sla.solve_triangular(R, Q.T @ y_train, lower=False)

    return w


def train_svd_model(X_train, y_train):
    # X w = y => USVt w = y => w = V Sm Ut y
    U, s, Vt = np.linalg.svd(X_train, full_matrices=False)
    w = U.T @ y_train
    w /= s
    w = Vt.T @ w

    return w


def error(y, y_hat, ord=2):
    diff = y - y_hat
    error = np.linalg.norm(diff, ord=ord)

    return error


def measure_time(method, X_train, y_train, n_iters=10_000):
    for _ in range(10):
        method(X_train, y_train)

    start = time.monotonic()
    for _ in range(n_iters):
        method(X_train, y_train)

    elapsed = time.monotonic() - start
    elapsed /= n_iters

    return elapsed


def main():
    args = parse_args(
        """
        --train-path ./data/water_supply/train_monthly_naturalized_flow.csv
        --test-path ./data/water_supply/test_monthly_naturalized_flow.csv
        --site-id missouri_r_at_toston
        --train-year 1991
        --test-year 2005
        """.split()
    )

    site_id = args.site_id
    df = read_water_supply(args.train_path, site_id=site_id)
    if args.test_path is not None:
        test_df = read_water_supply(args.test_path, site_id=site_id)
    else:
        test_df = None

    train_df, val_df = train_dev_split(
        df,
        train_year=args.train_year,
        test_year=args.test_year,
        test_df=test_df,
    )

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"water supply volume for {site_id}")

    train_df.plot(y="volume", ax=ax, label="train")
    val_df.plot(y="volume", ax=ax, label="val")

    plt.show()
    plt.close()

    X_train, y_train = get_data(train_df.volume.values)
    X_val, y_val = get_data(val_df.volume.values)

    w_lin = train_linear_model(X_train, y_train)
    #print(w_lin)
    print(error(y_val, predict_linear_model(X_val, w_lin)))

    w_qr = train_qr_model(X_train, y_train)
    #print(w_qr)
    print(error(y_val, predict_linear_model(X_val, w_qr)))

    w_svd = train_svd_model(X_train, y_train)
    #print(w_svd)
    print(error(y_val, predict_linear_model(X_val, w_svd)))

    print(f"lin vs qr : {error(w_lin, w_qr)}")
    print(f"lin vs svd: {error(w_lin, w_svd)}")
    print(f"qr vs svd : {error(w_qr, w_svd)}")

    elapsed = []
    for method in [
        train_linear_model,
        train_qr_model,
        train_svd_model,
    ]:
        elapsed.append(measure_time(method, X_train, y_train))
        print(method.__name__, elapsed[-1])


if __name__ == "__main__":
    main()
