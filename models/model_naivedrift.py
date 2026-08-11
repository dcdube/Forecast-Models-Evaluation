import argparse
import os
import sys
import warnings
import logging
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.metrics import calculate_metrics, forecast_plot_and_csv, plot_model_metrics
from utils.dataset_config import DatasetBelgiumNF, DatasetGermanyNF, DatasetLondonNF, DatasetZonnedaelNF
from utils.device import GPU_HELP, execution_target
import time
import gc
import numpy as np


class NaiveDrift:
    def __init__(self):
        self.last_value = None
        self.drift = 0
        self.noise_std = 0
        self.last_ds = None
        self.freq = None

    def fit(self, df):
        y = df['y'].values
        n = len(y)
        if n < 2:
            raise ValueError("Need at least 2 observations to estimate drift")
        self.last_value = y[-1]
        self.drift = (y[-1] - y[0]) / (n - 1)
        self.last_ds = df['ds'].iloc[-1]
        # Estimate noise std as std of first differences residuals around drift
        diffs = np.diff(y)
        residuals = diffs - self.drift
        self.noise_std = np.std(residuals) if len(residuals) > 1 else 0

    def predict(self, h):
        if self.last_ds is None or self.freq is None:
            raise ValueError("Must call fit and set frequency before predict")
        future_dates = pd.date_range(start=self.last_ds + pd.Timedelta(self.freq), periods=h, freq=self.freq)
        # Generate forecast values: deterministic drift + Gaussian noise
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=h)
        forecast_values = [self.last_value + self.drift * (i + 1) + noise[i] for i in range(h)]
        return pd.DataFrame({'ds': future_dates, 'NaiveDrift': forecast_values})

    def fit_predict(self, df, h, freq='1min'):
        self.freq = freq
        self.fit(df)
        return self.predict(h)

def setup_model_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_log.txt")
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )
    logging.info(f"Logger initialized at {log_file}")

def generic_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, sampling_rate):
    logging.info("Using NaiveDrift model")
    y_df = y_df.iloc[::int(100 / sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]
    # Instantiate your custom model and forecast
    # For your NaiveDrift, use fit_predict, otherwise fallback (if needed)
    forecast_df = NaiveDrift().fit_predict(train_df, forecast_horizon, freq=freq)
    # Placeholder for other models if needed
    # (e.g., your original StatsForecast usage)
    logging.info(f"Forecast columns: {forecast_df.columns.tolist()}")  # DEBUG: show columns
    # Pick forecast column dynamically
    forecast_cols = [col for col in forecast_df.columns if col != "ds"]
    if not forecast_cols:
        raise ValueError("No forecast columns found in forecast_df")
    y_pred = forecast_df[forecast_cols[0]].values
    y_true = test_df["y"].values
    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} (NaiveDrift) - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        f"{model_name}",
        save_dir
    )
    return None, mae, rmse

def train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, freq, forecast_horizon, sampling_rate):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    logging.info("...Start training using NaiveDrift...")
    if dataset_name == "belgium":
        logging.info("Training PV models")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt)
            _, pv_mae, pv_rmse = generic_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, sampling_rate)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        logging.info("Training BESS models")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt)
            _, battery_mae, battery_rmse = generic_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon, sampling_rate)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})
    elif dataset_name == "germany":
        logging.info("Training Germany load model")
        germany_data = dataset.get_inputs_for_load(start_dt, end_dt)
        _, load_mae, load_rmse = generic_forecast_model(germany_data, "germany_load", save_dir, freq, forecast_horizon, sampling_rate)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        logging.info("Training London load model")
        london_data = dataset.get_inputs_for_load()
        _, load_mae, load_rmse = generic_forecast_model(london_data, "london_load", save_dir, freq, forecast_horizon, sampling_rate)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        logging.info("Training Zonnedael customer models")
        for customer_id in [8, 9, 43]:
            customer_data = dataset.get_inputs_for_zonnedael_consumption(customer_id)
            _, cust_mae, cust_rmse = generic_forecast_model(customer_data, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon, sampling_rate)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info("...End training...")
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate, results_dir=PROJECT_ROOT / "results"):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    freq_str = f"{int(15 * (100 / sampling_rate))}min"
    forecast_horizon = int(192 / (100 / sampling_rate))
    try:
        gc.collect()
        save_dir = results_dir / f"results_{dataset_name}/NaiveDrift/Sampling_{sampling_rate:.0f}/Run_{run_num}"
        train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, freq_str, forecast_horizon, sampling_rate)
        gc.collect()
    except Exception as e:
        logging.error(f"Skipping model NaiveDrift due to error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    dataset_classes = dict(belgium=DatasetBelgiumNF, 
                           germany=DatasetGermanyNF, 
                           london=DatasetLondonNF, 
                           zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run the naive-drift forecasting benchmark.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    if execution_target(args.gpu) != "CPU":
        logging.info("Naive Drift has no GPU operations and will run on CPU.")
    dataset = dataset_classes[args.dataset]()
    for sampling_rate in args.sampling_rates:
        for run_num in range(1, args.runs + 1):
            paper_forecasting_train(args.dataset, dataset, run_num, sampling_rate, args.results_dir)
