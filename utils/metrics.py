import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import logging
from sklearn.preprocessing import MinMaxScaler

# Calculate MAE and RMSE metrics
def calculate_metrics(preds, actuals):
    preds = np.asarray(preds)
    actuals = np.asarray(actuals)
    
    mae = np.mean(np.abs(preds - actuals))
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    
    return mae, rmse

# Save trained model to pickle
def save_model(model, model_file):
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

def split_train_test(X, y, test_size):
    train_size = len(X) - test_size
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    return X_train, X_test, y_train, y_test

def setup_logger(save_dir):
    log_file = os.path.join(save_dir, "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

def min_max_normalize(X):
    scaler = MinMaxScaler()
    
    # If input is a Series or 1D array, reshape to 2D
    if isinstance(X, pd.Series):
        X_values = X.values.reshape(-1, 1)
        X_scaled = scaler.fit_transform(X_values)
        return pd.Series(X_scaled.flatten(), index=X.index)
    elif len(X.shape) == 1:
        X_scaled = scaler.fit_transform(X.reshape(-1, 1))
        return X_scaled.flatten()
    else:
        return pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

# Save forecast vs actual CSV and PDF plot
def forecast_plot_and_csv(df, model_name, save_dir):
    csv_path = os.path.join(save_dir, f"{model_name}_forecast_vs_actual.csv")
    pdf_path = os.path.join(save_dir, f"{model_name}_forecast_vs_actual.pdf")
    df.to_csv(csv_path)
    df.plot(figsize=(12, 5), title=f"Forecast vs Actual - {model_name}")
    plt.ylabel("Normalized Value")
    plt.xlabel("Datetime")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()

def plot_model_metrics(metrics, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # If metrics is a string, treat it as a CSV path
    if isinstance(metrics, str):
        metrics_df = pd.read_csv(metrics)
    else:
        # Assume it's a list of dicts and save to CSV
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
 