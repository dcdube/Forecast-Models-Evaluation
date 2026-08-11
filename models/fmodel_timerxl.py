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
import torch
from transformers import AutoModelForCausalLM
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")


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

# Core forecasting function
def hf_causal_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, hf_model_name, sampling_rate, device):
    y_df = y_df.iloc[::int(100/sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]
    logging.info(f"Running {hf_model_name} ({model_name}) for {len(train_df)} samples ({sampling_rate:.0f}% of original training set)...")
    context_values = train_df["y"].values.astype("float32")
    context_tensor = torch.tensor(context_values).unsqueeze(0).to(device)
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, trust_remote_code=True).to(device)
    model.generation_config.use_cache = False
    with torch.no_grad():
        output = model.generate(
            context_tensor,
            max_new_tokens=forecast_horizon,
            use_cache=False
        )
    y_pred = output[0, -forecast_horizon:].detach().cpu().numpy()
    y_true = test_df["y"].values
    if "sundial" in hf_model_name.lower():
        y_pred = y_pred.ravel()
        y_true = y_true.ravel()
    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name,
        save_dir
    )
    return mae, rmse

# Full pipeline with dataset toggle
def train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, freq, forecast_horizon, model_id, sampling_rate, device):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    logging.info(f"...Start {model_id} forecasting on {dataset_name} dataset...")
    if dataset_name == "belgium":
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt)
            pv_mae, pv_rmse = hf_causal_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, model_id, sampling_rate, device)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt)
            battery_mae, battery_rmse = hf_causal_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon, model_id, sampling_rate, device)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})
    elif dataset_name == "germany":
        load_data = dataset.get_inputs_for_load(start_dt, end_dt)
        load_mae, load_rmse = hf_causal_forecast_model(load_data, "germany_load", save_dir, freq, forecast_horizon, model_id, sampling_rate, device)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        load_data = dataset.get_inputs_for_load()
        load_mae, load_rmse = hf_causal_forecast_model(load_data, "london_load", save_dir, freq, forecast_horizon, model_id, sampling_rate, device)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        for customer_id in [8, 9, 43]:
            data_df = dataset.get_inputs_for_zonnedael_consumption(customer_id)
            cust_mae, cust_rmse = hf_causal_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon, model_id, sampling_rate, device)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info(f"...End {model_id} forecasting on {dataset_name} dataset...")
    logging.info(f"Forecasting completed in {elapsed_time:.2f} seconds.")

def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate, model_id, save_folder, device, results_dir=PROJECT_ROOT / "results"):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    freq_str = f"{int(15 * (100 / sampling_rate))}T"
    forecast_horizon = int(192/(100/sampling_rate))
    try:
        gc.collect()
        save_dir = results_dir / f"results_{dataset_name}/{save_folder}/Sampling_{sampling_rate:.0f}/Run_{run_num}"
        train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, freq_str, forecast_horizon, model_id, sampling_rate, device)
        gc.collect()
    except Exception as e:
        logging.error(f"Skipping {model_id} on {dataset_name} due to error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    dataset_classes = dict(belgium=DatasetBelgiumNF, 
                           germany=DatasetGermanyNF, 
                           london=DatasetLondonNF, 
                           zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run the Timer-XL forecasting benchmark.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs.")
    parser.add_argument("--model_id", default="thuml/timer-base-84m", help="Hugging Face model ID.")
    parser.add_argument("--save_folder", default="TimerXL", help="Result folder name.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device.")
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    # Run all sampling rates and models
    for sampling_rate in args.sampling_rates:
        # Loop for run_num
        for run_num in range(1, args.runs + 1):
            paper_forecasting_train(args.dataset, dataset, run_num, sampling_rate, args.model_id, args.save_folder, device, args.results_dir)
