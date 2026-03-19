from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f_size = 12

# Dataset Selection Toggle
selected_dataset = "germany"  # Options: "belgium" or "germany" or "london" or "zonnedael"

legend_name_map = {
    "KNNRegression": "KNN Reg.",
    "NaiveDrift": "Naive Drift",
    "NaiveMovingAverage": "Naive MA",
    "MQCNN": "MQ-CNN",
    "TemporalFusionTransformer": "TFT",
    "MQRNN": "MQ-RNN",
    "VanillaTransformer": "Vanilla Trans.",
    "TimerXL": "Timer-XL",
}

def _one_sided_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def dm_test_pvalue(loss_x: pd.Series, loss_y: pd.Series) -> float:
    """
    One-sided DM test p-value for H1: model_x has lower expected loss than model_y.
    """
    aligned = pd.concat([loss_x, loss_y], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return np.nan

    # Loss differential series d_t = L_x,t - L_y,t.
    d_t = aligned.iloc[:, 0].to_numpy() - aligned.iloc[:, 1].to_numpy()
    mean_d = float(np.mean(d_t))
    centered_d = d_t - mean_d
    var_d = float(np.mean(centered_d**2))

    if var_d <= 0.0:
        if mean_d < 0.0:
            return 0.0
        if mean_d > 0.0:
            return 1.0
        return 0.5

    dm_stat = mean_d / math.sqrt(var_d / len(d_t))
    return float(np.clip(_one_sided_normal_cdf(dm_stat), 0.0, 1.0))


def find_model_dirs(dataset_dir: Path) -> List[Path]:
    return sorted([path for path in dataset_dir.iterdir() if path.is_dir()])


def _find_target_file(dir_path: Path, target_filename: str) -> Optional[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return None

    for candidate in dir_path.iterdir():
        if candidate.is_file() and candidate.suffix.lower() == ".csv" and candidate.name.lower() == target_filename.lower():
            return candidate

    return None


def _iter_sampling_subdirs(sampling_dir: Path) -> List[Path]:
    if not sampling_dir.exists() or not sampling_dir.is_dir():
        return []

    return sorted([path for path in sampling_dir.iterdir() if path.is_dir()])


def _collect_target_files(model_dir: Path, target_filename: str) -> List[Path]:
    sampling_dir = model_dir / "Sampling_100"
    if not sampling_dir.exists():
        return []

    direct = _find_target_file(sampling_dir, target_filename)
    if direct is not None:
        return [direct]

    run1 = _find_target_file(sampling_dir / "Run_1", target_filename)
    if run1 is not None:
        return [run1]

    target_files: List[Path] = []
    for child in _iter_sampling_subdirs(sampling_dir):
        target_file = _find_target_file(child, target_filename)
        if target_file is not None:
            target_files.append(target_file)

    return target_files


def discover_target_filenames(dataset_dir: Path) -> List[str]:
    targets: Set[str] = set()
    for model_dir in find_model_dirs(dataset_dir):
        sampling_dir = model_dir / "Sampling_100"
        if not sampling_dir.exists() or not sampling_dir.is_dir():
            continue

        search_dirs = [sampling_dir]
        search_dirs.extend(_iter_sampling_subdirs(sampling_dir))

        for search_dir in search_dirs:
            for candidate in search_dir.iterdir():
                if not candidate.is_file() or candidate.suffix.lower() != ".csv":
                    continue
                name_lower = candidate.name.lower()
                if name_lower.endswith("_forecast_vs_actual.csv"):
                    targets.add(name_lower)

    return sorted(targets)


def _read_single_load_file(load_csv: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(load_csv)
    except Exception:
        return None

    normalized_cols = {col.lower().strip(): col for col in df.columns}
    datetime_col = normalized_cols.get("datetime")
    actual_col = normalized_cols.get("actual")
    forecast_col = normalized_cols.get("forecast")

    if actual_col is None or forecast_col is None:
        return None

    if datetime_col is not None:
        dt = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)
    else:
        dt = pd.to_datetime(df.index, errors="coerce", utc=True)

    actual = pd.to_numeric(df[actual_col], errors="coerce")
    forecast = pd.to_numeric(df[forecast_col], errors="coerce")
    valid = (~dt.isna()) & (~actual.isna()) & (~forecast.isna())
    if not valid.any():
        return None

    parsed = pd.DataFrame(
        {
            "datetime": pd.DatetimeIndex(dt[valid]),
            "actual": actual[valid].to_numpy(),
            "forecast": forecast[valid].to_numpy(),
        }
    )
    return parsed.sort_values("datetime")


def read_target_mae_losses(model_dir: Path, target_filename: str) -> Tuple[Optional[pd.Series], Optional[Path]]:
    target_files = _collect_target_files(model_dir, target_filename)
    if not target_files:
        return None, None

    run_frames: List[pd.DataFrame] = []
    for target_file in target_files:
        parsed = _read_single_load_file(target_file)
        if parsed is not None:
            run_frames.append(parsed)

    if not run_frames:
        return None, target_files[0]

    if len(run_frames) == 1:
        merged = run_frames[0].copy()
    else:
        aligned_runs: List[pd.DataFrame] = []
        for idx, frame in enumerate(run_frames):
            aligned = frame.set_index("datetime")[["actual", "forecast"]].rename(
                columns={"actual": f"actual_{idx}", "forecast": f"forecast_{idx}"}
            )
            aligned_runs.append(aligned)

        merged_runs = pd.concat(aligned_runs, axis=1, join="inner").dropna()
        if merged_runs.empty:
            return None, target_files[0]

        actual_cols = [col for col in merged_runs.columns if col.startswith("actual_")]
        forecast_cols = [col for col in merged_runs.columns if col.startswith("forecast_")]
        merged = pd.DataFrame(
            {
                "datetime": merged_runs.index,
                "actual": merged_runs[actual_cols].mean(axis=1).to_numpy(),
                "forecast": merged_runs[forecast_cols].mean(axis=1).to_numpy(),
            }
        )

    err = merged["actual"].to_numpy() - merged["forecast"].to_numpy()
    index = pd.DatetimeIndex(merged["datetime"])

    mae_losses = pd.Series(np.abs(err), index=index).sort_index()
    return mae_losses, target_files[0]


def build_dm_matrix(losses: Dict[str, Optional[pd.Series]], model_names: Iterable[str]) -> np.ndarray:
    names = list(model_names)
    n_models = len(names)
    matrix = np.full((n_models, n_models), np.nan, dtype=float)

    for i, model_x in enumerate(names):
        for j, model_y in enumerate(names):
            if i == j:
                matrix[i, j] = 1.0
                continue

            loss_x = losses.get(model_x)
            loss_y = losses.get(model_y)
            if loss_x is None or loss_y is None:
                continue

            matrix[i, j] = dm_test_pvalue(loss_x, loss_y)

    return matrix

def plot_dm_heatmap(
    model_names: List[str],
    dm_matrix: np.ndarray,
    output_pdf: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.04])
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_over("yellow")
    cmap.set_bad("white")
    im = ax.imshow(dm_matrix, cmap=cmap, vmin=0.0, vmax=0.1, aspect="auto")

    ticks = np.arange(len(model_names))
    display_names = [legend_name_map.get(name, name) for name in model_names]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=f_size)
    ax.set_yticklabels(display_names, fontsize=f_size)

    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label("p-value", rotation=90, labelpad=0, fontsize=f_size)
    cbar.ax.tick_params(labelsize=f_size)
    fig.savefig(output_pdf, format="pdf", dpi=300)
    plt.close(fig)


def run_dm_for_dataset(project_root: Path, dataset_name: str) -> List[Path]:
    dataset_dir = project_root / "results" / f"results_{dataset_name}"
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset results folder not found: {dataset_dir}")

    output_pdfs: List[Path] = []

    model_dirs = find_model_dirs(dataset_dir)
    if not model_dirs:
        print(f"Warning: no model folders found in {dataset_dir}")
        return output_pdfs

    target_filenames = discover_target_filenames(dataset_dir)
    if not target_filenames:
        print(f"Warning: no *_forecast_vs_actual.csv files found in {dataset_dir}")
        return output_pdfs

    model_names = [model_dir.name for model_dir in model_dirs]

    for target_filename in target_filenames:
        losses: Dict[str, Optional[pd.Series]] = {}
        missing_loss_files: List[str] = []

        for model_dir in model_dirs:
            model_name = model_dir.name
            mae_series, _ = read_target_mae_losses(model_dir, target_filename)
            losses[model_name] = mae_series
            if mae_series is None:
                missing_loss_files.append(model_name)

        if missing_loss_files:
            print(
                f"Warning ({dataset_dir.name} | {target_filename}): could not read forecast series for models: "
                + ", ".join(sorted(missing_loss_files))
            )

        dm_matrix = build_dm_matrix(losses, model_names)

        target_name = target_filename.replace("_forecast_vs_actual.csv", "")
        output_pdf = project_root / "results" / "DM_test" / dataset_name / f"DM_test_{target_name}.pdf"
        plot_dm_heatmap(model_names, dm_matrix, output_pdf)
        output_pdfs.append(output_pdf)

    return output_pdfs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot a DM test heatmap (MAE loss) for model comparisons.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[selected_dataset],
        help="Dataset suffixes used in results/results_<dataset_name> folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    for dataset_name in args.datasets:
        output_pdfs = run_dm_for_dataset(project_root, dataset_name)
        for output_pdf in output_pdfs:
            print(f"Saved DM test heatmap to: {output_pdf}")


if __name__ == "__main__":
    main()
