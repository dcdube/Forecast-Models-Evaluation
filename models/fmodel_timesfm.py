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
from utils.device import GPU_HELP, uses_gpu
import time
import gc


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

# Core function for TimesFM
def timesfm_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, tfm, sampling_rate):
    y_df = y_df.iloc[::int(100 / sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]
    logging.info(f"Running TimesFM {model_name} for {len(train_df)} samples ({sampling_rate:.0f}% of training set)...")
    train_values = train_df["y"].values.astype("float32")
    freq_input = [0]
    forecast_outputs = tfm.forecast([train_values], freq=freq_input)
    mean_forecast = forecast_outputs[0]
    y_pred = mean_forecast[0]
    y_true = test_df["y"].values
    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name,
        save_dir
    )
    return mae, rmse

# Full pipeline for TimesFM with dataset toggle
def train_all_models(
    dataset_name, dataset, start_dt, end_dt, save_dir, freq, forecast_horizon,
    sampling_rate, gpu, batch_size, model_id
):
    # TimesFM v1 selects a backend rather than a CUDA index. Limit visibility
    # before importing it so its GPU backend maps to the requested physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) if uses_gpu(gpu) else ""
    import timesfm

    backend = "gpu" if uses_gpu(gpu) else "cpu"
    setup_model_logger(save_dir)
    metrics = []
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=batch_size,
            horizon_len=forecast_horizon
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id=model_id
        )
    )
    start_time = time.time()
    logging.info("...Start TimesFM forecasting...")
    if dataset_name == "belgium":
        logging.info("Forecasting PV")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt)
            pv_mae, pv_rmse = timesfm_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, tfm, sampling_rate)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        logging.info("Forecasting BESS")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt)
            battery_mae, battery_rmse = timesfm_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon, tfm, sampling_rate)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})
    elif dataset_name == "germany":
        logging.info("Forecasting Germany load")
        load_data = dataset.get_inputs_for_load(start_dt, end_dt)
        load_mae, load_rmse = timesfm_forecast_model(load_data, "germany_load", save_dir, freq, forecast_horizon, tfm, sampling_rate)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        logging.info("Forecasting London load")
        load_data = dataset.get_inputs_for_load()
        load_mae, load_rmse = timesfm_forecast_model(load_data, "london_load", save_dir, freq, forecast_horizon, tfm, sampling_rate)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        logging.info("Forecasting Zonnedael customers")
        for customer_id in [8, 9, 43]:
            data_df = dataset.get_inputs_for_zonnedael_consumption(customer_id)
            cust_mae, cust_rmse = timesfm_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon, tfm, sampling_rate)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info("...End TimesFM forecasting...")
    logging.info(f"Forecasting completed in {elapsed_time:.2f} seconds.")

def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate_local, gpu, batch_size, model_id, results_dir=PROJECT_ROOT / "results"):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    freq_str = f"{int(15 * (100 / sampling_rate_local))}MIN"
    forecast_horizon = int(192 / (100 / sampling_rate_local))
    try:
        gc.collect()
        save_dir = results_dir / f"results_{dataset_name}/TimesFM/Sampling_{sampling_rate_local:.0f}/Run_{run_num}"
        train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, freq_str, forecast_horizon, sampling_rate_local, gpu, batch_size, model_id)
        gc.collect()
    except Exception as e:
        logging.error(f"Skipping TimesFM due to error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    dataset_classes = dict(belgium=DatasetBelgiumNF, 
                           germany=DatasetGermanyNF, 
                           london=DatasetLondonNF, 
                           zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run the TimesFM forecasting benchmark.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--batch_size", type=int, default=32, help="Per-core batch size.")
    parser.add_argument("--model_id", default="google/timesfm-1.0-200m-pytorch", help="Hugging Face checkpoint ID.")
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    # Run all sampling rates and seeds
    for sampling_rate in args.sampling_rates:
        for run_num in range(1, args.runs + 1):
            paper_forecasting_train(args.dataset, dataset, run_num, sampling_rate, args.gpu, args.batch_size, args.model_id, args.results_dir)
