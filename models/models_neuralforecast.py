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
from utils.device import GPU_HELP, lightning_device_config, uses_gpu
import torch
import time
import gc
from neuralforecast import NeuralForecast
from neuralforecast.models import (BiTCN, DeepNPTS, Informer, NBEATS, NHITS, NLinear, PatchTST,
    TCN, TiDE, TimesNet, TimeXer, VanillaTransformer, iTransformer)


# Custom logger setup per model
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

# Core training function for each model type
def neural_forecast_model(
    y_df, model_name, save_dir, freq, forecast_horizon, sampling_rate, epochs,
    gpu, nf_model_name="TimesNet", NFmodel=None
):
    current_seed = int(time.time()) % (2**32 - 1)  # Use current time for true randomness
    logging.info(f"Using random seed: {current_seed}")
    # Split data
    y_df = y_df.iloc[::int(100/sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]
    # train_df = full_train_df.iloc[::int(100/sampling_rate)]
    # plt.plot(train_df['ds'], train_df['y'])
    # plt.show()
    logging.info(f"Training {nf_model_name} {model_name} model on {len(train_df)} samples ({sampling_rate}% of original training set)...")
    device_config = lightning_device_config(gpu)
    if nf_model_name in ["TimeXer", "iTransformer"]:
        nf = NeuralForecast(
            models=[NFmodel(
                h=forecast_horizon,
                input_size=forecast_horizon * 2,
                n_series=1,
                max_steps=epochs,
                random_seed=current_seed,
                **device_config
            )],
            freq=freq
        )
    else:
        nf = NeuralForecast(
            models=[NFmodel(
                h=forecast_horizon,
                input_size=forecast_horizon * 2,
                max_steps=epochs,
                random_seed=current_seed,
                **device_config
            )],
            freq=freq
        )
    nf.fit(df=train_df, val_size=forecast_horizon)
    forecast_df = nf.predict()
    y_pred = forecast_df[nf_model_name].values
    y_true = test_df["y"].values
    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name,
        save_dir
    )
    return nf, mae, rmse

# Main pipeline for training all models
def train_all_models(
    dataset_name, dataset, start_dt, end_dt, save_dir, freq, forecast_horizon,
    sampling_rate, epochs, gpu, nf_model_name, NFmodel
):
    setup_model_logger(save_dir)  # Ensure logging is set up first
    metrics = []
    start_time = time.time()
    logging.info(f"...Start training for {epochs} epochs using {nf_model_name}...")
    if dataset_name == "belgium":
        # Train PV model
        logging.info("Training PV models")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt)
            _, pv_mae, pv_rmse = neural_forecast_model(
                pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, sampling_rate, epochs, gpu, nf_model_name, NFmodel
            )
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        # Train battery model
        logging.info("Training BESS models")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt)
            _, battery_mae, battery_rmse = neural_forecast_model(battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon,
                                                                 sampling_rate, epochs, gpu, nf_model_name, NFmodel)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})
    elif dataset_name == "germany":
        logging.info("Training Germany load model")
        germany_data = dataset.get_inputs_for_load(start_dt, end_dt)
        _, load_mae, load_rmse = neural_forecast_model(
            germany_data, "germany_load", save_dir, freq, forecast_horizon, sampling_rate, epochs, gpu, nf_model_name, NFmodel
        )
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        # Train london load model
        logging.info("Training London load model")
        london_data = dataset.get_inputs_for_load()
        _, load_mae, load_rmse = neural_forecast_model(
            london_data, "london_load", save_dir, freq, forecast_horizon, sampling_rate, epochs, gpu, nf_model_name, NFmodel
        )
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        # Train zonnedael customers
        logging.info("Training Zonnedael customer models")
        for customer_id in [8, 9, 43]:
            customer_data = dataset.get_inputs_for_zonnedael_consumption(customer_id)
            _, cust_mae, cust_rmse = neural_forecast_model(customer_data, f"zonnedael_customer_{customer_id}", save_dir, freq, forecast_horizon,
                                                           sampling_rate, epochs, gpu, nf_model_name, NFmodel)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info("...End training...")
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate, epochs, models, gpu, results_dir=PROJECT_ROOT / "results"):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    # Adjust frequency based on sampling rate
    freq_str = f"{int(15 * (100 / sampling_rate))}min"  # Convert to string format for frequency
    forecast_horizon = int(192 / (100 / sampling_rate))  # 2 days of 15-minute intervals
    for model_name, model_class in models.items():
        try:
            # Clear memory before training a new model
            gc.collect()
            if uses_gpu(gpu):
                torch.cuda.empty_cache()
            save_dir = results_dir / f"results_{dataset_name}/{model_name}/Sampling_{sampling_rate:.0f}/Epochs_{epochs}_{run_num}"
            train_all_models(
                dataset_name, dataset, start_dt, end_dt, save_dir, freq_str, forecast_horizon,
                sampling_rate, epochs, gpu, nf_model_name=model_name, NFmodel=model_class
            )
            # Clear memory after training a model
            gc.collect()
            if uses_gpu(gpu):
                torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Skipping model {model_name} due to error: {str(e)}", exc_info=True)
            continue

if __name__ == "__main__":
    model_classes = {
        "BiTCN": BiTCN,
        "DeepNPTS": DeepNPTS,
        "Informer": Informer,
        "NBEATS": NBEATS,
        "NHITS": NHITS,
        "NLinear": NLinear,
        "PatchTST": PatchTST,
        "TCN": TCN,
        "TiDE": TiDE,
        "TimesNet": TimesNet,
        "TimeXer": TimeXer,
        "iTransformer": iTransformer,
        "VanillaTransformer": VanillaTransformer
    }
    dataset_classes = dict(belgium=DatasetBelgiumNF, germany=DatasetGermanyNF, london=DatasetLondonNF, zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run NeuralForecast model benchmarks.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--models", nargs="+", choices=list(model_classes), default=list(model_classes), help="NeuralForecast models to evaluate.")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs.")
    parser.add_argument("--epochs", type=int, default=100, help="Training steps per model.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    selected_models = {name: model_classes[name] for name in args.models}
    for sampling_rate in args.sampling_rates:
        # Loop from run_num = 1 to 10
        for run_num in range(1, args.runs + 1):
            paper_forecasting_train(args.dataset, dataset, run_num, sampling_rate, args.epochs, selected_models, args.gpu, args.results_dir)
