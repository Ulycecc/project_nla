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


def get_topelitz_col_row(x, m):
    c = x[m - 1:-1]
    r = x[:m][::-1]

    return c, r


def get_data(volume, window_len=20):
    # volume: [n]
    # [
    #   [0, 1, 2],
    #   [1, 2, 3],
    #   [2, 3, 4],
    #   ...
    # ]

    #n = len(volume)
    #i = np.arange(window_len + 1)
    #n_samples = n - window_len - 1
    #i = np.tile(i, (n_samples, 1))
    #j = np.arange(n_samples)[:, None]
    #ij = i + j
    #x = volume[ij]
    #y = x[:, -1]
    #x = x[:, :-1]

    c, r = get_topelitz_col_row(volume, window_len)
    x = sla.toeplitz(c, r)
    y = volume[window_len:]

    return x, y


def predict_linear_model(X, w):
    y = X @ w

    return y


def padone(X):
    X = np.pad(
        X,
        ((0, 0), (1, 0)),
        mode="constant",
        constant_values=1,
    )

    return X


def linear_model(X, y):
    w, res, rank, s = np.linalg.lstsq(X, y, rcond=None)

    return w


def qr_model(X, y):
    # X w = y => QR w = y => w = Rm Qt y
    Q, R = np.linalg.qr(X)
    w = sla.solve_triangular(R, Q.T @ y, lower=False)

    return w


def svd_model(X, y):
    # X w = y => USVt w = y => w = V Sm Ut y
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    w = U.T @ y
    w /= s
    w = Vt.T @ w

    return w


def toeplitz_model(X, y, window_len):
    c, r = get_topelitz_col_row(X, window_len)
    w = sla.solve_toeplitz((c, r), y)

    return w


def error(y, y_hat, ord=2):
    diff = y - y_hat
    error = np.linalg.norm(diff, ord=ord)

    return error


def measure_time(method, X, y, n_iters=10, **kwargs):
    for _ in range(10):
        method(X, y, **kwargs)

    start = time.monotonic()
    for _ in range(n_iters):
        method(X, y, **kwargs)

    elapsed = time.monotonic() - start
    elapsed /= n_iters

    return elapsed


def make_predictions(x, w, n):
    x = x.copy()
    y = []
    for _ in range(n):
        a = x @ w
        y.append(a)

        x[:-1] = x[1:]
        x[-1] = a

    y = np.array(y)

    return y


def get_water_supply(path, test_path, site_id, train_year, test_year):
    df = read_water_supply(path, site_id=site_id)
    if test_path is not None:
        test_df = read_water_supply(test_path, site_id=site_id)
    else:
        test_df = None

    train_df, val_df = train_dev_split(
        df,
        train_year=train_year,
        test_year=test_year,
        test_df=test_df,
    )

    return train_df, val_df


def get_daily(path, usecol, split_ratio=0.8):
    df = pd.read_csv(
        path,
        usecols=[usecol],
    )
    df.rename({usecol: "volume"}, axis=1, inplace=True)

    n = len(df)
    n_train = int(split_ratio * n)
    train_df = df[:n_train]
    val_df = df[n_train:]

    return train_df, val_df


def get_taxi(path, test_year="2018-08-01"):
    df = pd.read_csv(path)
    df = df.set_index(pd.to_datetime(df["datetime"])).sort_index()
    df.rename({"num_orders": "volume"}, axis=1, inplace=True)
    train_df = df[:test_year]
    val_df = df[test_year:]

    return train_df, val_df


def main():
    args = parse_args(
            #--train-path ./data/water_supply/train_monthly_naturalized_flow.csv
            #--train-path ./data/Daily-train.csv
        """
        --train-path ./data/taxi.csv
        --test-path ./data/water_supply/test_monthly_naturalized_flow.csv
        --site-id missouri_r_at_toston
        --train-year 1951
        --test-year 2005
        """.split()
    )

    #train_df, val_df = get_water_supply(
    #    args.train_path,
    #    args.test_path,
    #    args.site_id,
    #    args.train_year,
    #    args.test_year,
    #)

    #train_df, val_df = get_daily(args.train_path, "V8")

    train_df, val_df = get_taxi(args.train_path)

    to_plot = {}
    window_lens = [5, 10, 50]#, 100, 200, 300]
    for window_len in window_lens:
        X_train, y_train = get_data(train_df.volume.values, window_len=window_len)
        X_val, y_val = get_data(val_df.volume.values, window_len=window_len)
        X_train = padone(X_train)
        X_val = padone(X_val)
        print(X_train.shape, np.linalg.cond(X_train))

        w_lin = linear_model(X_train, y_train)
        #print(w_lin)
        print(error(y_val, predict_linear_model(X_val, w_lin)))

        w_qr = qr_model(X_train, y_train)
        #print(w_qr)
        print(error(y_val, predict_linear_model(X_val, w_qr)))

        w_svd = svd_model(X_train, y_train)
        #print(w_svd)
        print(error(y_val, predict_linear_model(X_val, w_svd)))

        #w_toe = train_toeplitz_model(train_df.volume.values, y_train, window_len=window_len)
        #print(w_toe)
        #print(error(y_val, predict_linear_model(X_val, w_toe)))

        print(f"lin vs qr : {error(w_lin, w_qr)}")
        print(f"lin vs svd: {error(w_lin, w_svd)}")
        print(f"qr vs svd : {error(w_qr, w_svd)}")

        elapsed = []
        for method in [
            linear_model,
            qr_model,
            svd_model,
        ]:
            elapsed.append(measure_time(method, X_train, y_train))
            print(method.__name__, elapsed[-1])
            to_plot.setdefault(method.__name__, []).append(elapsed[-1])

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("lag vs. time")
    for key, value in to_plot.items():
        ax.plot(window_lens, value, label=key)

    ax.set_xlabel("number of lags (features)")
    ax.set_ylabel("time (s)")
    ax.legend()
    ax.grid(ls="--")
    fig.tight_layout()

    #ax.set_title(f"water supply volume")

    #train_df.plot(y="volume", ax=ax, label="train")
    #val_df.plot(y="volume", ax=ax, label="val")
    #val_df["preds"] = 0
    #val_df.preds.values[window_len:] = predict_linear_model(X_val, w_svd)
    #val_df.plot(y="preds", ax=ax, label="pred")

    plt.savefig("window_lens.png")
    plt.show()
    plt.close(fig)

    to_plot = {}
    window_len = 10
    data_sizes = [0.1, 0.25, 0.5, 1]
    for data_size in data_sizes:
        n = int(data_size * len(train_df))
        X_train, y_train = get_data(train_df.volume.values[:n], window_len=window_len)
        X_val, y_val = get_data(val_df.volume.values, window_len=window_len)

        X_train = padone(X_train)
        X_val = padone(X_val)
        print(X_train.shape, np.linalg.cond(X_train))

        w_lin = linear_model(X_train, y_train)
        #print(w_lin)
        print(error(y_val, predict_linear_model(X_val, w_lin)))

        w_qr = qr_model(X_train, y_train)
        #print(w_qr)
        print(error(y_val, predict_linear_model(X_val, w_qr)))

        w_svd = svd_model(X_train, y_train)
        #print(w_svd)
        print(error(y_val, predict_linear_model(X_val, w_svd)))

        #w_toe = train_toeplitz_model(train_df.volume.values, y_train, window_len=window_len)
        #print(w_toe)
        #print(error(y_val, predict_linear_model(X_val, w_toe)))

        print(f"lin vs qr : {error(w_lin, w_qr)}")
        print(f"lin vs svd: {error(w_lin, w_svd)}")
        print(f"qr vs svd : {error(w_qr, w_svd)}")

        elapsed = []
        for method in [
            linear_model,
            qr_model,
            svd_model,
        ]:
            elapsed.append(measure_time(method, X_train, y_train))
            print(method.__name__, elapsed[-1])
            to_plot.setdefault(method.__name__, []).append(elapsed[-1])

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"data size ({len(train_df)}) vs. time with fixed lag ({window_len})")
    for key, value in to_plot.items():
        ax.plot(data_sizes, value, label=key)

    ax.set_xlabel("number of samples (% of full)")
    ax.set_ylabel("time (s)")
    ax.legend()
    ax.grid(ls="--")
    fig.tight_layout()

    #ax.set_title(f"water supply volume")

    #train_df.plot(y="volume", ax=ax, label="train")
    #val_df.plot(y="volume", ax=ax, label="val")
    #val_df["preds"] = 0
    #val_df.preds.values[window_len:] = predict_linear_model(X_val, w_svd)
    #val_df.plot(y="preds", ax=ax, label="pred")

    plt.savefig("data_lens.png")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
