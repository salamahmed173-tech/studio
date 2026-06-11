"""Extract GAC Motors imports to UAE from CAAM (CSV/URL), visualize, and forecast with Prophet.

Usage examples:
  python analyze_gac_uae.py --input caam_data.csv --outdir outputs
  python analyze_gac_uae.py --input https://example.com/caam.csv --outdir outputs

If no valid data file is provided the script exits with an explanation so you can
provide the CAAM export file or URL.
"""
import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from prophet import Prophet



def download_csv(url, dst):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(dst, "wb") as f:
        f.write(r.content)


def detect_columns(df):
    cols = {c.lower(): c for c in df.columns}
    # heuristics
    date_col = None
    maker_col = None
    country_col = None
    units_col = None

    for k, v in cols.items():
        if any(x in k for x in ("date", "shipment_date", "time")):
            date_col = v
        if any(x in k for x in ("maker", "manufacturer", "brand", "makername", "exporter", "importer")):
            maker_col = v
        if any(x in k for x in ("country", "destination", "to_country", "consignee", "import_country")):
            country_col = v
        if any(x in k for x in ("unit", "units", "quantity", "qty")):
            units_col = v

    return date_col, maker_col, country_col, units_col


def load_data(path_or_url):
    is_url = str(path_or_url).lower().startswith("http")
    tmp_path = path_or_url
    if is_url:
        tmp_path = "__caam_tmp.csv"
        download_csv(path_or_url, tmp_path)

    df = pd.read_csv(tmp_path)

    date_col, maker_col, country_col, units_col = detect_columns(df)
    if not date_col or not maker_col or not country_col or not units_col:
        print("Could not auto-detect required columns in the CAAM CSV. Columns found:", list(df.columns))
        print("Expected a date column, a maker/manufacturer column, a country/destination column, and a units column.")
        sys.exit(1)

    df = df[[date_col, maker_col, country_col, units_col]].copy()
    df.columns = ["date", "maker", "country", "units"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # drop invalid dates
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0)
    return df


def filter_gac_uae(df, last_n_years=2):
    mask_maker = df["maker"].astype(str).str.contains("gac", case=False, na=False)
    mask_country = df["country"].astype(str).str.contains("uae|united arab emirates|ae", case=False, na=False)
    df = df[mask_maker & mask_country].copy()
    if df.empty:
        print("No rows found for GAC and UAE in the provided data.")
        sys.exit(1)

    latest = df["date"].max()
    cutoff = latest - pd.DateOffset(years=last_n_years)
    df = df[df["date"] >= cutoff]
    return df


def aggregate_monthly(df):
    df = df.set_index("date")
    monthly = df["units"].resample("MS").sum().reset_index()
    monthly.columns = ["ds", "y"]
    return monthly


def grid_search_prophet(train_df, val_df):
    # Randomized limited grid search over a small hyperparameter space.
    cps = [0.001, 0.01, 0.05, 0.1]
    sps = [1.0, 5.0, 10.0]
    modes = ["additive", "multiplicative"]
    params = []
    for a in cps:
        for b in sps:
            for m in modes:
                params.append((a, b, m))

    # shuffle and limit evaluations to keep fitting time reasonable
    import random

    random.shuffle(params)
    max_evals = min(9, len(params))
    best = None
    best_rmse = float("inf")

    for (cps_val, sps_val, mode_val) in params[:max_evals]:
        try:
            m = Prophet(changepoint_prior_scale=cps_val, seasonality_prior_scale=sps_val, seasonality_mode=mode_val, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(train_df)
            future = m.make_future_dataframe(periods=len(val_df), freq="MS")
            fc = m.predict(future)
            preds = fc.set_index("ds")["yhat"].reindex(val_df["ds"]).values
            # compute RMSE using numpy to avoid heavy sklearn import
            rmse = float(np.sqrt(np.mean((val_df["y"].values - preds) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best = dict(changepoint_prior_scale=cps_val, seasonality_prior_scale=sps_val, seasonality_mode=mode_val, rmse=rmse)
        except Exception:
            continue

    # if nothing worked, fall back to a default
    if not best:
        return dict(changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, seasonality_mode="additive", rmse=float("inf"))
    return best


def fit_and_forecast(monthly, periods=6, outdir="outputs"):
    monthly = monthly.sort_values("ds")
    # split last 6 months for validation if enough data
    if len(monthly) < 12:
        print("Warning: less than 12 months of data — model may be unreliable.")

    val_periods = min(6, int(len(monthly) * 0.2) or 1)
    train = monthly.iloc[:-val_periods]
    val = monthly.iloc[-val_periods:]

    best = grid_search_prophet(train, val)
    print("Best params:", best)

    m = Prophet(changepoint_prior_scale=best["changepoint_prior_scale"], seasonality_prior_scale=best["seasonality_prior_scale"], yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(monthly)
    future = m.make_future_dataframe(periods=periods, freq="MS")
    fc = m.predict(future)

    # save forecast CSV
    os.makedirs(outdir, exist_ok=True)
    fc.to_csv(os.path.join(outdir, "forecast.csv"), index=False)

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["ds"], monthly["y"], "k.-", label="history")
    plt.plot(fc["ds"], fc["yhat"], "b-", label="forecast")
    plt.fill_between(fc["ds"].dt.to_pydatetime(), fc["yhat_lower"], fc["yhat_upper"], color="b", alpha=0.2)
    plt.axvline(monthly["ds"].max(), color="gray", linestyle="--", label="forecast start")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.title("GAC Motors imports to UAE - history and Prophet forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "forecast.png"))

    return best, fc


def prepare_lags(monthly, lags=3):
    df = monthly.copy().sort_values("ds").reset_index(drop=True)
    df_idx = df.set_index("ds")
    # lag features
    for lag in range(1, lags + 1):
        df_idx[f"lag_{lag}"] = df_idx["y"].shift(lag)

    # time-based features
    df_idx["month"] = df_idx.index.month
    # cyclical encoding for month
    df_idx["month_sin"] = np.sin(2 * np.pi * df_idx["month"] / 12.0)
    df_idx["month_cos"] = np.cos(2 * np.pi * df_idx["month"] / 12.0)

    # rolling statistics (use past 3 months by default)
    win = min(3, max(1, lags))
    df_idx["roll_mean_3"] = df_idx["y"].shift(1).rolling(window=win, min_periods=1).mean()
    df_idx["roll_std_3"] = df_idx["y"].shift(1).rolling(window=win, min_periods=1).std().fillna(0.0)

    df_idx = df_idx.dropna().reset_index()
    return df_idx


def train_gbm_forecast(monthly, periods=6, outdir="outputs"):
    monthly = monthly.sort_values("ds").reset_index(drop=True)
    lags = 3
    df = prepare_lags(monthly, lags=lags)
    if df.empty:
        return dict(rmse=float("inf")), None

    val_periods = min(6, int(len(monthly) * 0.2) or 1)
    train_df = df.iloc[:-val_periods]
    val_df = df.iloc[-val_periods:]

    X_train = train_df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y_train = train_df["y"].values
    X_val = val_df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y_val = val_df["y"].values

    best = None
    best_rmse = float("inf")
    # small grid
    for n_estimators in (50, 100):
        for max_depth in (2, 4):
            for lr in (0.05, 0.1):
                try:
                    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, rmse=rmse)
                        best_model = model
                except Exception:
                    continue

    if not best:
        return dict(rmse=float("inf")), None

    # refit on full available lagged data
    X_full = df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y_full = df["y"].values
    best_model.fit(X_full, y_full)

    # iterative forecasting using last observed values
    last_vals = monthly["y"].values[-lags:].tolist()
    preds = []
    for i in range(periods):
        x_in = np.array(last_vals[-lags:])[::-1] if False else np.array(last_vals[-lags:])
        # ensure shape (1, lags)
        x_in = x_in.reshape(1, -1)
        p = float(best_model.predict(x_in)[0])
        preds.append(p)
        last_vals.append(p)

    # build forecast dataframe
    last_date = monthly["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    fc = pd.DataFrame({"ds": future_dates, "yhat": preds})
    # add placeholder bounds
    fc["yhat_lower"] = fc["yhat"]
    fc["yhat_upper"] = fc["yhat"]

    os.makedirs(outdir, exist_ok=True)
    fc.to_csv(os.path.join(outdir, "forecast_gbm.csv"), index=False)

    # plot combined
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["ds"], monthly["y"], "k.-", label="history")
    plt.plot(fc["ds"], fc["yhat"], "g-", label="gbm_forecast")
    plt.axvline(monthly["ds"].max(), color="gray", linestyle="--", label="forecast start")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.title("GAC Motors imports to UAE - GBM forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "forecast_gbm.png"))

    return best, fc


def train_xgb_forecast(monthly, periods=6, outdir="outputs"):
    monthly = monthly.sort_values("ds").reset_index(drop=True)
    lags = 6
    df = prepare_lags(monthly, lags=lags)
    if df.empty:
        return dict(rmse=float("inf")), None

    val_periods = min(6, int(len(monthly) * 0.2) or 1)
    train_df = df.iloc[:-val_periods]
    val_df = df.iloc[-val_periods:]

    X_train = train_df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y_train = train_df["y"].values
    X_val = val_df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y_val = val_df["y"].values

    best = None
    best_rmse = float("inf")
    best_model = None
    for n in (50, 100):
        for d in (3, 6):
            for lr in (0.05, 0.1):
                try:
                    model = XGBRegressor(n_estimators=n, max_depth=3, learning_rate=lr, verbosity=0)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best = dict(n_estimators=n, max_depth=3, learning_rate=lr, rmse=rmse)
                        best_model = model
                except Exception:
                    continue

    if not best:
        return dict(rmse=float("inf")), None

    # refit on full lagged data
    X_full = df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y_full = df["y"].values
    best_model.fit(X_full, y_full)

    # iterative forecast
    last_vals = monthly["y"].values[-lags:].tolist()
    preds = []
    for i in range(periods):
        x_in = np.array(last_vals[-lags:]).reshape(1, -1)
        p = float(best_model.predict(x_in)[0])
        preds.append(p)
        last_vals.append(p)

    future_dates = pd.date_range(start=monthly["ds"].max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    fc = pd.DataFrame({"ds": future_dates, "yhat": preds})
    fc["yhat_lower"] = fc["yhat"]
    fc["yhat_upper"] = fc["yhat"]

    os.makedirs(outdir, exist_ok=True)
    fc.to_csv(os.path.join(outdir, "forecast_xgb.csv"), index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["ds"], monthly["y"], "k.-", label="history")
    plt.plot(fc["ds"], fc["yhat"], "m-", label="xgb_forecast")
    plt.axvline(monthly["ds"].max(), color="gray", linestyle="--", label="forecast start")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.title("GAC Motors imports to UAE - XGBoost forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "forecast_xgb.png"))

    return best, fc


def train_sarima_forecast(monthly, periods=6, outdir="outputs"):
    # simple SARIMA grid with small combos to limit runtime
    monthly = monthly.set_index("ds").asfreq('MS')
    val_periods = min(6, int(len(monthly) * 0.2) or 1)
    train = monthly.iloc[:-val_periods]["y"]
    val = monthly.iloc[-val_periods:]["y"]

    orders = [(1, 1, 1), (2, 1, 1)]
    seasonal_orders = [(0, 1, 1, 12), (1, 1, 1, 12)]
    best = None
    best_rmse = float("inf")
    best_model = None

    for o in orders:
        for so in seasonal_orders:
            try:
                model = SARIMAX(train, order=o, seasonal_order=so, enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                preds = res.get_forecast(steps=val_periods).predicted_mean.values
                rmse = float(np.sqrt(np.mean((val.values - preds) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best = dict(order=o, seasonal_order=so, rmse=rmse)
                    best_model = res
            except Exception:
                continue

    if not best:
        return dict(rmse=float("inf")), None

    # refit on full series
    full = monthly["y"]
    model = SARIMAX(full, order=best["order"], seasonal_order=best["seasonal_order"], enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.get_forecast(steps=periods)
    fc_mean = preds.predicted_mean
    fc_index = pd.date_range(start=monthly.index.max() + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
    fc = pd.DataFrame({"ds": fc_index, "yhat": fc_mean.values})
    ci = preds.conf_int()
    fc["yhat_lower"] = ci.iloc[:, 0].values
    fc["yhat_upper"] = ci.iloc[:, 1].values

    os.makedirs(outdir, exist_ok=True)
    fc.to_csv(os.path.join(outdir, "forecast_sarima.csv"), index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(monthly.index, monthly["y"], "k.-", label="history")
    plt.plot(fc["ds"], fc["yhat"], "r-", label="sarima_forecast")
    plt.fill_between(fc["ds"].dt.to_pydatetime(), fc["yhat_lower"], fc["yhat_upper"], color="r", alpha=0.2)
    plt.axvline(monthly.index.max(), color="gray", linestyle="--", label="forecast start")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.title("GAC Motors imports to UAE - SARIMA forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "forecast_sarima.png"))

    return best, fc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to CAAM CSV or URL")
    p.add_argument("--outdir", default="outputs", help="Output folder for plots and forecasts")
    p.add_argument("--model", choices=["prophet", "gbm", "auto"], default="auto", help="Model to use: prophet, gbm, or auto to pick best")
    p.add_argument("--deep", action="store_true", help="Run deeper rolling-window CV hyperparameter search (slower)")
    p.add_argument("--deeper", action="store_true", help="Run expanded grids and build a weighted ensemble (much slower)")
    p.add_argument("--years", type=int, default=2, help="Number of past years to analyze")
    args = p.parse_args()

    df = load_data(args.input)
    df = filter_gac_uae(df, last_n_years=args.years)
    monthly = aggregate_monthly(df)

    # save processed
    os.makedirs(args.outdir, exist_ok=True)
    monthly.to_csv(os.path.join(args.outdir, "monthly_units.csv"), index=False)

    # plot history
    plt.figure(figsize=(10, 4))
    plt.plot(monthly["ds"], monthly["y"], marker="o")
    plt.title("Monthly imported units: GAC Motors -> UAE")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "monthly_history.png"))

    results = {}
    if args.model in ("prophet", "auto"):
        best_p, fc_p = fit_and_forecast(monthly, periods=6, outdir=args.outdir)
        results['prophet'] = (best_p, fc_p)
    if args.model in ("gbm", "auto"):
        best_g, fc_g = train_gbm_forecast(monthly, periods=6, outdir=args.outdir)
        results['gbm'] = (best_g, fc_g)
    if args.model in ("xgb", "auto"):
        best_x, fc_x = train_xgb_forecast(monthly, periods=6, outdir=args.outdir)
        results['xgb'] = (best_x, fc_x)
    if args.model in ("sarima", "auto"):
        best_s, fc_s = train_sarima_forecast(monthly, periods=6, outdir=args.outdir)
        results['sarima'] = (best_s, fc_s)

    # If deep search requested, run rolling-window CV for each model with expanded grids
    if args.deep:
        def rolling_cv_prophet(grid, n_splits=3, horizon=3):
            best = None
            best_rmse = float('inf')
            for cps in grid['cps']:
                for sps in grid['sps']:
                    rmses = []
                    # rolling splits
                    for split in range(n_splits):
                        # define train end index
                        split_end = len(monthly) - (n_splits - split) * horizon
                        train_df = monthly.iloc[:split_end]
                        val_df = monthly.iloc[split_end:split_end + horizon]
                        try:
                            m = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps, seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                            m.fit(train_df)
                            future = m.make_future_dataframe(periods=horizon, freq='MS')
                            fc = m.predict(future)
                            preds = fc.set_index('ds')['yhat'].reindex(val_df['ds']).values
                            rmse = float(np.sqrt(np.mean((val_df['y'].values - preds) ** 2)))
                            rmses.append(rmse)
                        except Exception:
                            rmses.append(float('inf'))
                    mean_rmse = float(np.mean(rmses))
                    if mean_rmse < best_rmse:
                        best_rmse = mean_rmse
                        best = dict(changepoint_prior_scale=cps, seasonality_prior_scale=sps, rmse=mean_rmse)
            return best

        def rolling_cv_gbm(grid, n_splits=3, horizon=3):
            best = None
            best_rmse = float('inf')
            lags = 6
            df_lag = prepare_lags(monthly, lags=lags)
            for n in grid['n_estimators']:
                for md in grid['max_depth']:
                    for lr in grid['learning_rate']:
                        rmses = []
                        for split in range(n_splits):
                            split_end = len(df_lag) - (n_splits - split) * horizon
                            if split_end <= 0:
                                rmses.append(float('inf'))
                                continue
                            train_df = df_lag.iloc[:split_end]
                            val_df = df_lag.iloc[split_end:split_end + horizon]
                            X_train = train_df[[f'lag_{i}' for i in range(1, lags + 1)]].values
                            y_train = train_df['y'].values
                            X_val = val_df[[f'lag_{i}' for i in range(1, lags + 1)]].values
                            y_val = val_df['y'].values
                            try:
                                model = GradientBoostingRegressor(n_estimators=n, max_depth=md, learning_rate=lr)
                                model.fit(X_train, y_train)
                                preds = model.predict(X_val)
                                rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
                                rmses.append(rmse)
                            except Exception:
                                rmses.append(float('inf'))
                        mean_rmse = float(np.mean(rmses))
                        if mean_rmse < best_rmse:
                            best_rmse = mean_rmse
                            best = dict(n_estimators=n, max_depth=md, learning_rate=lr, rmse=mean_rmse)
            return best

        def rolling_cv_xgb(grid, n_splits=3, horizon=3):
            best = None
            best_rmse = float('inf')
            lags = 6
            df_lag = prepare_lags(monthly, lags=lags)
            for n in grid['n_estimators']:
                for lr in grid['learning_rate']:
                    rmses = []
                    for split in range(n_splits):
                        split_end = len(df_lag) - (n_splits - split) * horizon
                        if split_end <= 0:
                            rmses.append(float('inf'))
                            continue
                        train_df = df_lag.iloc[:split_end]
                        val_df = df_lag.iloc[split_end:split_end + horizon]
                        X_train = train_df[[f'lag_{i}' for i in range(1, lags + 1)]].values
                        y_train = train_df['y'].values
                        X_val = val_df[[f'lag_{i}' for i in range(1, lags + 1)]].values
                        y_val = val_df['y'].values
                        try:
                            model = XGBRegressor(n_estimators=n, learning_rate=lr, verbosity=0)
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
                            rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
                            rmses.append(rmse)
                        except Exception:
                            rmses.append(float('inf'))
                    mean_rmse = float(np.mean(rmses))
                    if mean_rmse < best_rmse:
                        best_rmse = mean_rmse
                        best = dict(n_estimators=n, learning_rate=lr, rmse=mean_rmse)
            return best

        def rolling_cv_sarima(grid, n_splits=2, horizon=3):
            best = None
            best_rmse = float('inf')
            monthly_idx = monthly.set_index('ds').asfreq('MS')
            for o in grid['orders']:
                for so in grid['seasonal_orders']:
                    rmses = []
                    for split in range(n_splits):
                        split_end = len(monthly_idx) - (n_splits - split) * horizon
                        if split_end <= 12:
                            rmses.append(float('inf'))
                            continue
                        train = monthly_idx.iloc[:split_end]['y']
                        val = monthly_idx.iloc[split_end:split_end + horizon]['y']
                        try:
                            model = SARIMAX(train, order=o, seasonal_order=so, enforce_stationarity=False, enforce_invertibility=False)
                            res = model.fit(disp=False)
                            preds = res.get_forecast(steps=horizon).predicted_mean.values
                            rmse = float(np.sqrt(np.mean((val.values - preds) ** 2)))
                            rmses.append(rmse)
                        except Exception:
                            rmses.append(float('inf'))
                    mean_rmse = float(np.mean(rmses))
                    if mean_rmse < best_rmse:
                        best_rmse = mean_rmse
                        best = dict(order=o, seasonal_order=so, rmse=mean_rmse)
            return best

        # define grids
        prop_grid = {'cps': [0.01, 0.05], 'sps': [5.0, 10.0]}
        gbm_grid = {'n_estimators': [50, 100], 'max_depth': [2, 4], 'learning_rate': [0.05, 0.1]}
        xgb_grid = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
        sarima_grid = {'orders': [(1,1,1), (2,1,1)], 'seasonal_orders': [(0,1,1,12), (1,1,1,12)]}

        print('Running deep rolling CV: Prophet...')
        best_prop_cv = rolling_cv_prophet(prop_grid)
        print('Prophet CV best:', best_prop_cv)
        print('Running deep rolling CV: GBM...')
        best_gbm_cv = rolling_cv_gbm(gbm_grid)
        print('GBM CV best:', best_gbm_cv)
        print('Running deep rolling CV: XGB...')
        best_xgb_cv = rolling_cv_xgb(xgb_grid)
        print('XGB CV best:', best_xgb_cv)
        print('Running deep rolling CV: SARIMA...')
        best_sarima_cv = rolling_cv_sarima(sarima_grid)
        print('SARIMA CV best:', best_sarima_cv)

        # Prefer model with lowest CV RMSE
        candidates = [('prophet', best_prop_cv), ('gbm', best_gbm_cv), ('xgb', best_xgb_cv), ('sarima', best_sarima_cv)]
        chosen = min(candidates, key=lambda x: x[1].get('rmse', float('inf')))[0]
        print('Deep CV chosen model:', chosen)
        # If user asked for even deeper search and ensemble, proceed
        if args.deeper:
            print('Running expanded grids and ensemble training...')
            # expanded grids
            prop_grid_big = {'cps': [0.01, 0.05, 0.1], 'sps': [1.0, 5.0, 10.0]}
            gbm_grid_big = {'n_estimators': [100, 200], 'max_depth': [2, 4, 6], 'learning_rate': [0.01, 0.05, 0.1]}
            xgb_grid_big = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1]}
            sarima_grid_big = {'orders': [(1,1,1), (2,1,1), (2,1,2)], 'seasonal_orders': [(0,1,1,12), (1,1,1,12)]}

            best_prop_big = rolling_cv_prophet(prop_grid_big, n_splits=3, horizon=6)
            best_gbm_big = rolling_cv_gbm(gbm_grid_big, n_splits=3, horizon=6)
            best_xgb_big = rolling_cv_xgb(xgb_grid_big, n_splits=3, horizon=6)
            best_sarima_big = rolling_cv_sarima(sarima_grid_big, n_splits=2, horizon=6)

            print('Expanded CV results:')
            print('Prophet:', best_prop_big)
            print('GBM:', best_gbm_big)
            print('XGB:', best_xgb_big)
            print('SARIMA:', best_sarima_big)

            # Use last 6 months as holdout for final evaluation and ensemble
            horizon = 6
            train_df = monthly.iloc[:-horizon].copy()
            holdout_df = monthly.iloc[-horizon:].copy()

            preds_map = {}
            rmse_map = {}

            # Prophet final
            try:
                m = Prophet(changepoint_prior_scale=best_prop_big.get('changepoint_prior_scale', 0.05), seasonality_prior_scale=best_prop_big.get('seasonality_prior_scale', 5.0), seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                m.fit(train_df)
                future = m.make_future_dataframe(periods=horizon, freq='MS')
                fc = m.predict(future)
                p_prop = fc.set_index('ds')['yhat'].reindex(holdout_df['ds']).values
                rmse_prop = float(np.sqrt(np.mean((holdout_df['y'].values - p_prop) ** 2)))
                preds_map['prophet'] = p_prop
                rmse_map['prophet'] = rmse_prop
            except Exception as e:
                rmse_map['prophet'] = float('inf')

            # GBM final
            try:
                lags = 6
                df_lag = prepare_lags(monthly, lags=lags)
                train_lag = df_lag.iloc[:len(df_lag)-horizon]
                hold_lag = df_lag.iloc[len(df_lag)-horizon:]
                X_train = train_lag[[f'lag_{i}' for i in range(1, lags+1)]].values
                y_train = train_lag['y'].values
                X_hold = hold_lag[[f'lag_{i}' for i in range(1, lags+1)]].values
                gbm_params = best_gbm_big or {'n_estimators':100,'max_depth':2,'learning_rate':0.05}
                model_g = GradientBoostingRegressor(n_estimators=gbm_params.get('n_estimators',100), max_depth=gbm_params.get('max_depth',2), learning_rate=gbm_params.get('learning_rate',0.05))
                model_g.fit(X_train, y_train)
                p_gbm = model_g.predict(X_hold)
                rmse_gbm = float(np.sqrt(np.mean((hold_lag['y'].values - p_gbm) ** 2)))
                preds_map['gbm'] = p_gbm
                rmse_map['gbm'] = rmse_gbm
            except Exception:
                rmse_map['gbm'] = float('inf')

            # XGB final
            try:
                xgb_params = best_xgb_big or {'n_estimators':100,'learning_rate':0.1}
                model_x = XGBRegressor(n_estimators=xgb_params.get('n_estimators',100), learning_rate=xgb_params.get('learning_rate',0.1), verbosity=0)
                model_x.fit(X_train, y_train)
                p_xgb = model_x.predict(X_hold)
                rmse_xgb = float(np.sqrt(np.mean((hold_lag['y'].values - p_xgb) ** 2)))
                preds_map['xgb'] = p_xgb
                rmse_map['xgb'] = rmse_xgb
            except Exception:
                rmse_map['xgb'] = float('inf')

            # SARIMA final
            try:
                order = best_sarima_big.get('order', (1,1,1))
                seasonal = best_sarima_big.get('seasonal_order', (0,1,1,12))
                monthly_idx = monthly.set_index('ds').asfreq('MS')
                train_series = monthly_idx.iloc[:-horizon]['y']
                sar = SARIMAX(train_series, order=order, seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False)
                res = sar.fit(disp=False)
                p_sar = res.get_forecast(steps=horizon).predicted_mean.values
                hold_series = monthly_idx.iloc[-horizon:]['y'].values
                rmse_sar = float(np.sqrt(np.mean((hold_series - p_sar) ** 2)))
                preds_map['sarima'] = p_sar
                rmse_map['sarima'] = rmse_sar
            except Exception:
                rmse_map['sarima'] = float('inf')

            print('Final holdout RMSEs:', rmse_map)

            # Build weighted ensemble using inverse RMSE weights
            weights = {}
            for k, v in rmse_map.items():
                weights[k] = 0.0 if not (v and np.isfinite(v) and v>0) else 1.0 / v
            s = sum(weights.values())
            if s == 0:
                print('Ensemble not possible (no valid models).')
            else:
                for k in weights:
                    weights[k] = weights[k] / s
                print('Ensemble weights:', weights)
                # align preds and compute ensemble
                keys = list(preds_map.keys())
                stacked = np.vstack([preds_map[k] for k in keys])
                ens = np.zeros(stacked.shape[1]) if stacked.ndim==2 else np.zeros(stacked.shape[0])
                # handle 2D stacked (models x horizon)
                if stacked.ndim == 2:
                    for i,k in enumerate(keys):
                        ens += weights.get(k,0.0) * stacked[i]
                else:
                    for i,k in enumerate(keys):
                        ens += weights.get(k,0.0) * stacked[i]
                # compute ensemble rmse
                hold_vals = holdout_df['y'].values
                ens_rmse = float(np.sqrt(np.mean((hold_vals - ens) ** 2)))
                print('Ensemble RMSE on holdout:', ens_rmse)

                # Save ensemble forecast CSV
                future_index = pd.date_range(start=monthly['ds'].max() + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
                out_df = pd.DataFrame({'ds': future_index, 'yhat_ensemble': ens})
                out_csv = os.path.join(args.outdir, 'forecast_ensemble.csv')
                out_df.to_csv(out_csv, index=False)
                print('Ensemble forecast saved to', out_csv)
                # Attempt stacking (OOF stacking with Ridge meta-learner)
                try:
                    from sklearn.linear_model import RidgeCV

                    # adapt folds and horizon to available history
                    horizon_oof = min(6, max(1, len(monthly) // 8))
                    n_splits = min(5, max(2, len(monthly) // (horizon_oof * 2)))
                    print(f'Running OOF stacking with Ridge meta-learner (n_splits={n_splits}, horizon={horizon_oof})...')

                    oof_X = []
                    oof_y = []
                    for split in range(n_splits):
                        split_end = len(monthly) - (n_splits - split) * horizon_oof
                        if split_end <= 6:
                            continue
                        train_df = monthly.iloc[:split_end].copy()
                        val_df = monthly.iloc[split_end:split_end + horizon_oof].copy()
                        preds_split = []

                        # Prophet
                        try:
                            m = Prophet(changepoint_prior_scale=best_prop_big.get('changepoint_prior_scale', 0.05), seasonality_prior_scale=best_prop_big.get('seasonality_prior_scale', 5.0), seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                            m.fit(train_df)
                            future = m.make_future_dataframe(periods=horizon_oof, freq='MS')
                            fc = m.predict(future)
                            p_prop = fc.set_index('ds')['yhat'].reindex(val_df['ds']).values
                        except Exception:
                            p_prop = np.full(len(val_df), np.nan)
                        preds_split.append(p_prop)

                        # lag-based GBM/XGB
                        try:
                            lags = 6
                            df_lag = prepare_lags(monthly, lags=lags)
                            train_lag = df_lag.iloc[:split_end]
                            val_lag = df_lag.iloc[split_end:split_end + horizon_oof]
                            X_tr = train_lag[[f'lag_{i}' for i in range(1, lags+1)]].values
                            y_tr = train_lag['y'].values
                            X_val = val_lag[[f'lag_{i}' for i in range(1, lags+1)]].values

                            g = GradientBoostingRegressor(n_estimators=best_gbm_big.get('n_estimators',100), max_depth=best_gbm_big.get('max_depth',2), learning_rate=best_gbm_big.get('learning_rate',0.05))
                            g.fit(X_tr, y_tr)
                            p_g = g.predict(X_val)
                        except Exception:
                            p_g = np.full(len(val_df), np.nan)
                        preds_split.append(p_g)

                        try:
                            x = XGBRegressor(n_estimators=best_xgb_big.get('n_estimators',100), learning_rate=best_xgb_big.get('learning_rate',0.1), verbosity=0)
                            x.fit(X_tr, y_tr)
                            p_x = x.predict(X_val)
                        except Exception:
                            p_x = np.full(len(val_df), np.nan)
                        preds_split.append(p_x)

                        # SARIMA
                        try:
                            monthly_idx = monthly.set_index('ds').asfreq('MS')
                            train_series = monthly_idx.iloc[:split_end]['y']
                            sar = SARIMAX(train_series, order=best_sarima_big.get('order',(1,1,1)), seasonal_order=best_sarima_big.get('seasonal_order',(0,1,1,12)), enforce_stationarity=False, enforce_invertibility=False)
                            res = sar.fit(disp=False)
                            p_s = res.get_forecast(steps=horizon_oof).predicted_mean.values
                        except Exception:
                            p_s = np.full(len(val_df), np.nan)
                        preds_split.append(p_s)

                        stacked = np.vstack(preds_split).T
                        oof_X.append(stacked)
                        oof_y.append(val_df['y'].values.reshape(-1,1))

                    if len(oof_X) == 0:
                        print('OOF stacking had no valid folds; skipping.')
                    else:
                        X_meta = np.vstack(oof_X)
                        y_meta = np.vstack(oof_y).ravel()
                        col_mask = ~np.all(np.isnan(X_meta), axis=0)
                        X_meta = X_meta[:, col_mask]
                        col_mean = np.nanmean(X_meta, axis=0)
                        inds = np.where(np.isnan(X_meta))
                        X_meta[inds] = np.take(col_mean, inds[1])

                        meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
                        meta.fit(X_meta, y_meta)
                        print('Stacking meta coefficients:', meta.coef_)

                        # prepare holdout stacked inputs
                        hold_preds = []
                        try:
                            m = Prophet(changepoint_prior_scale=best_prop_big.get('changepoint_prior_scale', 0.05), seasonality_prior_scale=best_prop_big.get('seasonality_prior_scale', 5.0), seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                            m.fit(monthly.iloc[:-horizon])
                            future = m.make_future_dataframe(periods=horizon, freq='MS')
                            fc = m.predict(future)
                            p_prop_full = fc.set_index('ds')['yhat'].reindex(holdout_df['ds']).values
                        except Exception:
                            p_prop_full = np.full(horizon, np.nan)
                        hold_preds.append(p_prop_full)

                        try:
                            lags = 6
                            df_lag = prepare_lags(monthly, lags=lags)
                            train_lag = df_lag.iloc[:len(df_lag)-horizon]
                            hold_lag = df_lag.iloc[len(df_lag)-horizon:]
                            X_trf = train_lag[[f'lag_{i}' for i in range(1, lags+1)]].values
                            y_trf = train_lag['y'].values
                            g = GradientBoostingRegressor(n_estimators=best_gbm_big.get('n_estimators',100), max_depth=best_gbm_big.get('max_depth',2), learning_rate=best_gbm_big.get('learning_rate',0.05))
                            g.fit(X_trf, y_trf)
                            p_g_full = g.predict(hold_lag[[f'lag_{i}' for i in range(1, lags+1)]].values)
                        except Exception:
                            p_g_full = np.full(horizon, np.nan)
                        hold_preds.append(p_g_full)

                        try:
                            x = XGBRegressor(n_estimators=best_xgb_big.get('n_estimators',100), learning_rate=best_xgb_big.get('learning_rate',0.1), verbosity=0)
                            x.fit(X_trf, y_trf)
                            p_x_full = x.predict(hold_lag[[f'lag_{i}' for i in range(1, lags+1)]].values)
                        except Exception:
                            p_x_full = np.full(horizon, np.nan)
                        hold_preds.append(p_x_full)

                        try:
                            sar = SARIMAX(monthly.set_index('ds').asfreq('MS')[:-horizon]['y'], order=best_sarima_big.get('order',(1,1,1)), seasonal_order=best_sarima_big.get('seasonal_order',(0,1,1,12)), enforce_stationarity=False, enforce_invertibility=False)
                            res = sar.fit(disp=False)
                            p_s_full = res.get_forecast(steps=horizon).predicted_mean.values
                        except Exception:
                            p_s_full = np.full(horizon, np.nan)
                        hold_preds.append(p_s_full)

                        stacked_hold = np.vstack(hold_preds).T
                        stacked_hold = stacked_hold[:, col_mask]
                        inds = np.where(np.isnan(stacked_hold))
                        stacked_hold[inds] = np.take(col_mean, inds[1])
                        ens_meta = meta.predict(stacked_hold)
                        ens_meta_rmse = float(np.sqrt(np.mean((holdout_df['y'].values - ens_meta) ** 2)))
                        print('Stacking ensemble RMSE on holdout:', ens_meta_rmse)
                        out_df2 = pd.DataFrame({'ds': future_index, 'yhat_stacked': ens_meta})
                        out_csv2 = os.path.join(args.outdir, 'forecast_stacked.csv')
                        out_df2.to_csv(out_csv2, index=False)
                        print('Stacked forecast saved to', out_csv2)
                except Exception as e:
                    print('Stacking failed:', e)

    # choose best when auto
    chosen = args.model
    if args.model == 'auto':
        # pick model with lower RMSE
        best_choice = None
        best_rmse = float('inf')
        for k, (b, fc) in results.items():
            if b and b.get('rmse', float('inf')) < best_rmse:
                best_rmse = b.get('rmse', float('inf'))
                best_choice = k
        chosen = best_choice or 'prophet'

    print(f"Chosen model: {chosen}")
    print("Forecast saved to", args.outdir)


if __name__ == "__main__":
    main()
