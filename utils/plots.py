import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def calculate_metrics(preds, actuals):
    preds = np.asarray(preds)
    actuals = np.asarray(actuals)
    
    mae = np.mean(np.abs(preds - actuals))
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    
    return mae, rmse

def process_forecast_directory(directory):
    """Loop through CSV files and compute metrics."""
    metrics = []
    for filename in os.listdir(directory):
        if filename.endswith("_forecast_vs_actual.csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            if {"Actual", "Forecast"}.issubset(df.columns):
                mae, rmse = calculate_metrics(df["Actual"], df["Forecast"])
                model_name = filename.replace("_forecast_vs_actual.csv", "")
                metrics.append({
                    "model": model_name,
                    "MAE": mae,
                    "RMSE": rmse
                })
                logging.info(f"Processed {filename}")
            else:
                logging.warning(f"Missing 'y_true' or 'y_pred' in {filename}")
    return metrics

def plot_model_metrics(metrics, save_dir: str):
    """Save metrics CSV and generate PDF bar plots."""
    os.makedirs(save_dir, exist_ok=True)

    # Convert to DataFrame
    if isinstance(metrics, str):
        metrics_df = pd.read_csv(metrics)
    else:
        metrics_df = pd.DataFrame(metrics)
        csv_path = os.path.join(save_dir, "model_metrics_summary.csv")
        metrics_df.to_csv(csv_path, index=False)
        logging.info(f"Saved model metrics to {csv_path}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Metrics Summary", fontsize=16)
    metrics_to_plot = ["MAE", "RMSE"]

    for ax, metric in zip(axes.flatten(), metrics_to_plot):
        sns.barplot(
            data=metrics_df,
            x="model",
            y=metric,
            hue="model",
            dodge=False,
            palette="Set2",
            ax=ax
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True)
        if ax.get_legend():
            ax.get_legend().remove()

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "model_metrics_summary.pdf")
    plt.savefig(pdf_path)
    logging.info(f"Saved model metrics plot to {pdf_path}")

if __name__ == "__main__":
    directory = "results/results_belgium/AutoARIMA/Sampling_100/Run_1"
    metrics = process_forecast_directory(directory)
    plot_model_metrics(metrics, save_dir=directory)
