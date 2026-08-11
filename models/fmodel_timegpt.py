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
import time
import gc
from nixtla import NixtlaClient


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

# Core function for TimeGPT
def timegpt_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, client,
    timegpt_model, confidence_level, finetune_steps, finetune_depth):
    y_df['ds'] = pd.to_datetime(y_df['ds'])
    y_df = y_df.set_index('ds')
    y_df = y_df[~y_df.index.duplicated(keep='first')]  # Remove duplicate timestamps
    # Resample to the target frequency. This is more robust than iloc slicing.
    # It creates a regular time index, which is required by the TimeGPT API.
    y_df = y_df.resample(freq).first()
    # Interpolate to fill any gaps created by resampling or present in the original data.
    y_df.interpolate(method='linear', inplace=True)
    y_df.ffill(inplace=True)  # Fill any remaining NaNs at the start/end
    y_df.bfill(inplace=True)
    y_df = y_df.reset_index()
    # The original iloc slicing for sampling has been replaced by the robust resampling method above.
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]
    logging.info(f"Running TimeGPT {model_name} for {len(train_df)} samples with frequency {freq}...")
    forecast = client.forecast(
        df=train_df,
        h=forecast_horizon,
        model=timegpt_model,
        freq=freq,
        finetune_steps=finetune_steps,
        finetune_depth=finetune_depth,
        level=[confidence_level]
    )
    y_pred = forecast["TimeGPT"].values
    y_true = test_df["y"].values
    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name,
        save_dir
    )
    return mae, rmse

# Full pipeline for TimeGPT
def train_all_models(
    dataset_name, dataset, start_dt, end_dt, save_dir, freq, forecast_horizon,
    client, timegpt_model, confidence_level, finetune_steps, finetune_depth
):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    logging.info("...Start TimeGPT forecasting...")
    if dataset_name == "belgium":
        logging.info("Forecasting PV")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt)
            pv_mae, pv_rmse = timegpt_forecast_model(
                pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon,
                client, timegpt_model, confidence_level, finetune_steps, finetune_depth
            )
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        logging.info("Forecasting BESS")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt)
            battery_mae, battery_rmse = timegpt_forecast_model(
                battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon,
                client, timegpt_model, confidence_level, finetune_steps, finetune_depth
            )
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})
    elif dataset_name == "germany":
        logging.info("Forecasting Germany load model")
        germany_data = dataset.get_inputs_for_load(start_dt, end_dt)
        load_mae, load_rmse = timegpt_forecast_model(
            germany_data, "germany_load", save_dir, freq, forecast_horizon,
            client, timegpt_model, confidence_level, finetune_steps, finetune_depth
        )
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        logging.info("Forecasting London load model")
        london_data = dataset.get_inputs_for_load()
        load_mae, load_rmse = timegpt_forecast_model(
            london_data, "london_load", save_dir, freq, forecast_horizon,
            client, timegpt_model, confidence_level, finetune_steps, finetune_depth
        )
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        logging.info("Forecasting Zonnedael customer models")
        for customer_id in [8, 9, 43]:
            customer_data = dataset.get_inputs_for_zonnedael_consumption(customer_id)
            cust_mae, cust_rmse = timegpt_forecast_model(
                customer_data, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon,
                client, timegpt_model, confidence_level, finetune_steps, finetune_depth
            )
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info("...End TimeGPT forecasting...")
    logging.info(f"Forecasting completed in {elapsed_time:.2f} seconds.")

def paper_forecasting_train(
    dataset_name, dataset, run_num, sampling_rate, client, timegpt_model,
    confidence_level, finetune_steps, finetune_depth, results_dir=PROJECT_ROOT / "results"
):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    freq_str = f"{int(15 * (100 / sampling_rate))}T"
    forecast_horizon = int(192/(100/sampling_rate))
    try:
        gc.collect()
        save_dir = results_dir / f"results_{dataset_name}/TimeGPT/Sampling_{sampling_rate:.0f}/Run_{run_num}"
        train_all_models(
            dataset_name, dataset, start_dt, end_dt, save_dir, freq_str,
            forecast_horizon, client, timegpt_model, confidence_level, finetune_steps, finetune_depth
        )
        gc.collect()
    except Exception as e:
        logging.error(f"Skipping TimeGPT due to error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    dataset_classes = dict(belgium=DatasetBelgiumNF, 
                           germany=DatasetGermanyNF, 
                           london=DatasetLondonNF, 
                           zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run the TimeGPT forecasting benchmark.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs.")
    parser.add_argument("--api_key", default="nixak-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", help="TimeGPT API key.")
    parser.add_argument("--model", choices=["timegpt-1", "timegpt-1-long-horizon"], default="timegpt-1-long-horizon", help="TimeGPT model.")
    parser.add_argument("--confidence_level", type=int, default=95, help="Prediction interval confidence level.")
    parser.add_argument("--finetune_steps", type=int, default=0, help="Fine-tuning steps.")
    parser.add_argument("--finetune_depth", type=int, default=1, help="Fine-tuning depth (finetune_steps > 0).")
    # =========================================================================================
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    client = NixtlaClient(api_key=args.api_key)
    # Run all sampling rates and seeds
    for sampling_rate in args.sampling_rates:
        # Loop for run_num
        for run_num in range(1, args.runs + 1):
            paper_forecasting_train(
                args.dataset,
                dataset,
                run_num,
                sampling_rate,
                client,
                args.model,
                args.confidence_level,
                args.finetune_steps,
                args.finetune_depth,
                args.results_dir
            )
