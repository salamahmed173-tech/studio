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

try:
    from prophet import Prophet
except Exception:
    print("Error: Prophet is not installed. Run: pip install -r requirements.txt")
    raise


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
    for lag in range(1, lags + 1):
        df_idx[f"lag_{lag}"] = df_idx["y"].shift(lag)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to CAAM CSV or URL")
    p.add_argument("--outdir", default="outputs", help="Output folder for plots and forecasts")
    p.add_argument("--model", choices=["prophet", "gbm", "auto"], default="auto", help="Model to use: prophet, gbm, or auto to pick best")
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
