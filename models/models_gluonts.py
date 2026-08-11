import argparse
import os
import sys
import warnings
import logging
import time
import gc
from pathlib import Path
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
import mxnet as mx
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.metrics import calculate_metrics, forecast_plot_and_csv, plot_model_metrics
from utils.dataset_config import DatasetBelgiumNF, DatasetGermanyNF, DatasetLondonNF, DatasetZonnedaelNF
from utils.device import GPU_HELP, uses_gpu
from gluonts.mx.model.seq2seq import MQRNNEstimator
from gluonts.mx.model.seq2seq import MQCNNEstimator
from gluonts.mx.model.deep_factor import DeepFactorEstimator
from gluonts.mx.model.wavenet import WaveNetEstimator
from gluonts.mx.model.tft import TemporalFusionTransformerEstimator
from gluonts.mx.model.deepar import DeepAREstimator


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

def to_gluonts_dataset(df, freq, start):
    return ListDataset([{"start": start, "target": df["y"].values}], freq=freq)

def train_gluonts_model(y_df, model_name, save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx):
    current_seed = int(time.time()) % (2**32 - 1)
    logging.info(f"Using random seed: {current_seed}")
    y_df = y_df.iloc[::int(100 / sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df
    start_idx = pd.Timestamp(train_df["ds"].iloc[0])
    train_ds = to_gluonts_dataset(train_df, freq, start_idx)
    test_ds = to_gluonts_dataset(test_df, freq, start_idx)
    logging.info(f"Training {model_name} model on {len(train_df)} samples...")
    estimator = model_class(
        freq=freq,
        prediction_length=forecast_horizon,
        trainer=Trainer(
            epochs=epochs,
            num_batches_per_epoch=1,
            ctx=ctx,
            hybridize=model_class not in {MQCNNEstimator, MQRNNEstimator}
        )
    )
    predictor = estimator.train(training_data=train_ds)
    forecast_it, _ = make_evaluation_predictions(dataset=test_ds, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    y_pred = forecasts[0].mean[:forecast_horizon]
    y_true = y_df["y"].values[-forecast_horizon:]
    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    forecast_plot_and_csv(
        pd.DataFrame({"datetime": y_df["ds"].values[-forecast_horizon:], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name,
        save_dir
    )
    return predictor, mae, rmse

def train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, model_name, model_class, sampling_rate, epochs, ctx):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    logging.info(f"...Start training using {model_name} with {sampling_rate}% sampling rate...")
    forecast_horizon = int(192 / (100 / sampling_rate))
    freq = f"{int(15 * (100 / sampling_rate))}min"
    if dataset_name == "belgium":
        logging.info("Training PV models")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt)
            _, pv_mae, pv_rmse = train_gluonts_model(pv_data, f"PV_house_{house}", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        logging.info("Training BESS models")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt)
            _, b_mae, b_rmse = train_gluonts_model(battery_data, f"BESS_house_{house}", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx)
            metrics.append({"model": f"bess_house_{house}", "MAE": b_mae, "RMSE": b_rmse})
    elif dataset_name == "germany":
        logging.info("Training Germany load model")
        germany_data = dataset.get_inputs_for_load(start_dt, end_dt)
        _, load_mae, load_rmse = train_gluonts_model(germany_data, "germany_load", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        logging.info("Training London load model")
        london_data = dataset.get_inputs_for_load()
        _, load_mae, load_rmse = train_gluonts_model(london_data, "london_load", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        logging.info("Training Zonnedael customer models")
        for customer_id in [8, 9, 43]:
            cust_data = dataset.get_inputs_for_zonnedael_consumption(customer_id)
            _, c_mae, c_rmse = train_gluonts_model(
                cust_data, f"zonnedael_customer_{customer_id}", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx
            )
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": c_mae, "RMSE": c_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info("...End training...")
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

def paper_forecasting_train(dataset_name, dataset, sampling_rates, models, runs, epochs, gpu, ctx, results_dir=PROJECT_ROOT / "results"):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    for sampling_rate in sampling_rates:
        for model_name, model_class in models.items():
            # Run 10 times
            for run_num in range(1, runs + 1):
                try:
                    gc.collect()
                    if uses_gpu(gpu):
                        mx.nd.waitall()
                    save_dir = results_dir / f"results_{dataset_name}/{model_name}/Sampling_{sampling_rate:.0f}/Epochs_{epochs}_{run_num}"
                    train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, model_name, model_class, sampling_rate, epochs, ctx)
                    gc.collect()
                    if uses_gpu(gpu):
                        mx.nd.waitall()
                except Exception as e:
                    logging.error(f"Skipping model {model_name} run {run_num} at sampling {sampling_rate}% due to error: {str(e)}", exc_info=True)
                    continue

if __name__ == "__main__":
    model_classes = {
        "DeepAR": DeepAREstimator,
        "DeepFactor": DeepFactorEstimator,
        "MQCNN": MQCNNEstimator,
        "MQRNN": MQRNNEstimator,
        "TemporalFusionTransformer": TemporalFusionTransformerEstimator,  # TFT
        "WaveNet": WaveNetEstimator
    }
    dataset_classes = dict(belgium=DatasetBelgiumNF, germany=DatasetGermanyNF, london=DatasetLondonNF, zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run GluonTS model benchmarks.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--models", nargs="+", choices=list(model_classes), default=list(model_classes), help="GluonTS models to evaluate.")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    selected_models = {name: model_classes[name] for name in args.models}
    ctx = mx.gpu(args.gpu) if uses_gpu(args.gpu) else mx.cpu()
    paper_forecasting_train(args.dataset, dataset, args.sampling_rates, selected_models, args.runs, args.epochs, args.gpu, ctx, args.results_dir)
