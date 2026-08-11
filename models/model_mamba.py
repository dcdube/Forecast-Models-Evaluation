import os
import sys
import warnings
import logging
import time
import gc
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm import Mamba
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.metrics import calculate_metrics, forecast_plot_and_csv, plot_model_metrics
from utils.dataset_config import DatasetBelgiumNF, DatasetGermanyNF, DatasetLondonNF, DatasetZonnedaelNF
from utils.device import GPU_HELP, torch_device, uses_gpu


class MambaForecaster(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, forecast_horizon):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.head = nn.Linear(d_model, forecast_horizon)

    def forward(self, x):
        # x: (batch, length, d_model)
        y = self.mamba(x)
        last_state = y[:, -1, :]
        return self.head(last_state)

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

def build_windows(series, context_length, forecast_horizon):
    values = series.values.astype("float32")
    total_length = context_length + forecast_horizon
    if len(values) < total_length:
        return None, None
    X = []
    y = []
    for i in range(len(values) - total_length + 1):
        window = values[i:i + total_length]
        X.append(window[:context_length])
        y.append(window[context_length:])
    X = torch.tensor(X).unsqueeze(-1)  # (batch, length, 1)
    y = torch.tensor(y)  # (batch, horizon)
    return X, y

# Core function for Mamba forecasting
def mamba_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, sampling_rate, epochs, device):
    y_df = y_df.iloc[::int(100 / sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]
    context_length = min(forecast_horizon * 2, max(1, len(train_df) - forecast_horizon))
    logging.info(
        f"Running Mamba {model_name} for {len(train_df)} samples "
        f"(sampling {sampling_rate:.0f}%, context {context_length}, horizon {forecast_horizon})"
    )
    X_train, y_train = build_windows(train_df["y"], context_length, forecast_horizon)
    if X_train is None:
        raise ValueError("Not enough data to build training windows.")
    model = MambaForecaster(
        d_model=1,
        d_state=16,
        d_conv=4,
        expand=2,
        forecast_horizon=forecast_horizon
    ).to(device)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            logging.info(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss / len(train_loader):.6f}")
    model.eval()
    with torch.no_grad():
        context_values = train_df["y"].values.astype("float32")
        context_values = context_values[-context_length:]
        context_tensor = torch.tensor(context_values).unsqueeze(0).unsqueeze(-1).to(device)
        forecast = model(context_tensor).cpu().numpy().flatten()
    y_true = test_df["y"].values
    y_pred = forecast[:forecast_horizon]
    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name,
        save_dir
    )
    return mae, rmse

# Full pipeline for Mamba
def train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, freq, forecast_horizon, sampling_rate, epochs, device):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    logging.info("...Start Mamba forecasting...")
    if dataset_name == "belgium":
        logging.info("Forecasting PV")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt)
            pv_mae, pv_rmse = mamba_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, sampling_rate, epochs, device)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        logging.info("Forecasting BESS")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt)
            battery_mae, battery_rmse = mamba_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon, sampling_rate, epochs, device)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})
    elif dataset_name == "germany":
        logging.info("Forecasting Germany load")
        load_data = dataset.get_inputs_for_load(start_dt, end_dt)
        load_mae, load_rmse = mamba_forecast_model(load_data, "germany_load", save_dir, freq, forecast_horizon, sampling_rate, epochs, device)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        logging.info("Forecasting London load")
        load_data = dataset.get_inputs_for_load()
        load_mae, load_rmse = mamba_forecast_model(load_data, "london_load", save_dir, freq, forecast_horizon, sampling_rate, epochs, device)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        logging.info("Forecasting Zonnedael customers")
        for customer_id in [8, 9, 43]:
            data_df = dataset.get_inputs_for_zonnedael_consumption(customer_id)
            cust_mae, cust_rmse = mamba_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon, sampling_rate, epochs, device)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info("...End Mamba forecasting...")
    logging.info(f"Forecasting completed in {elapsed_time:.2f} seconds.")

def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate, epochs, device, gpu, results_dir=PROJECT_ROOT / "results"):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    freq_str = f"{int(15 * (100 / sampling_rate))}min"
    forecast_horizon = int(192 / (100 / sampling_rate))
    try:
        gc.collect()
        if uses_gpu(gpu):
            torch.cuda.empty_cache()
        save_dir = results_dir / f"results_{dataset_name}/Mamba/Sampling_{sampling_rate:.0f}/Run_{run_num}"
        train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, freq_str, forecast_horizon, sampling_rate, epochs, device)
        gc.collect()
        if uses_gpu(gpu):
            torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"Skipping Mamba due to error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    dataset_classes = dict(belgium=DatasetBelgiumNF, 
                           germany=DatasetGermanyNF, 
                           london=DatasetLondonNF, 
                           zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run the Mamba forecasting benchmark.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    device = torch.device(torch_device(args.gpu))
    # Run all sampling rates and seeds
    for sampling_rate in args.sampling_rates:
        for run_num in range(1, args.runs + 1):
            paper_forecasting_train(args.dataset, dataset, run_num, sampling_rate, args.epochs, device, args.gpu, args.results_dir)
