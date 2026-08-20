# A Survey and Benchmark for Household Electricity Forecasting: From Statistical to Foundation Models

This is the official repository for our work on benchmarking forecasting models on **household energy consumption and generation** tasks. We evaluate **30 representative time-series forecasting models** spanning the full spectrum of forecasting approaches, from classical statistical and machine learning models to modern deep learning architectures and **time series foundation models (TSFMs)**. All models are evaluated within a **unified benchmarking framework** to ensure fair comparison. The study focuses on three key residential energy forecasting tasks:

- **Electricity load forecasting**
- **Solar photovoltaic (PV) generation forecasting**
- **Battery energy storage system (BESS) operation forecasting**

The benchmarking pipeline applies a standardized preprocessing workflow including **missing value interpolation, outlier detection, normalization, and temporal resampling**. Models are evaluated using their **default configurations** to assess their **out-of-the-box performance** without extensive hyperparameter tuning. The main objectives of this work are therefore to:

- Provide a **broad comparison of forecasting models** of different architectures
- Establish a **consistent benchmarking framework**
- Evaluate model performance across **multiple real-world energy forecasting tasks**
- Investigate the **zero-shot forecasting capabilities of TSFMs**

![Survey method overview](figures/survey_method.png)

## Quick Start

### Create environment and install dependencies

Separate python environments are provided for the model libraries to avoid dependency conflicts. Each model script has a corresponding environment file in the [`requirements`](requirements) directory.

For example, create and activate the *Chronos* environment with:

```bash
conda env create --file requirements/env_chronos.yml
conda activate chronos
```

To create another model environment, replace `env_chronos.yml` with its corresponding `env_<model>.yml` file. The environment name is defined by the `name` field in each file. Every model keeps its settings as defaults and provides command line interface (CLI) help for example:

```bash
python models/model_naivedrift.py --help
```

Use `python <script> --help` for all available options.

The common options are `--dataset`, `--sampling_rate`, `--runs`, `--gpu`, and `--results_dir`. Use `--gpu 0` for GPU 0, `--gpu 1` for GPU 1, or any other value (the default is `-1`) for CPU. Multi-model scripts also provide `--models`; trainable models provide `--epochs` where applicable. Foundation-model scripts expose their checkpoint, batch size, and service settings through their own CLI options.

## Datasets

Datasets are loaded via [utils/dataset_config.py](utils/dataset_config.py). File locations used by the loaders:

- Belgium PV and battery datasets: [local files](data/belgium_dataset); [original data](https://github.com/EVERGi/real_validation_saferl_treec_paper/tree/main/data/houses)
- Germany WPUQ (SFH19): [local file](data/germany_wpuq_dataset/SFH19_2023_2024_15min_3_month.csv); [original data](https://springernature.figshare.com/articles/dataset/Metadata_record_for_Dataset_on_electrical_single-family_house_and_heat_pump_load_profiles_in_Germany/17206271)
- London smart meter dataset: [local file](data/london_dataset/LCL_london_consumption_2013.csv); [original data](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
- Zonnedael dataset: [local file](data/zonnedael_dataset/liander_zonnedael_2013_original.csv); [original data](https://www.liander.nl/over-ons/open-data#verbruiksdata-slimme-meter)

## Forecasting Models

Each model lists its primary library and a GitHub repository. Scripts are linked for direct execution.

![Forecast models timeline](figures/forcast_models_timeline.png)

### 1. Statistical and Machine Learning

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| AutoARIMA | pmdarima | https://github.com/alkaline-ml/pmdarima | [models/models_statsml.py](models/models_statsml.py) |
| KNN Regression | scikit-learn | https://github.com/scikit-learn/scikit-learn | [models/models_statsml.py](models/models_statsml.py) |
| LightGBM | LightGBM | https://github.com/microsoft/LightGBM | [models/models_statsml.py](models/models_statsml.py) |
| Naive Drift | custom (numpy, pandas) | https://github.com/numpy/numpy | [models/model_naivedrift.py](models/model_naivedrift.py) |
| Naive Moving Average | custom (numpy, pandas) | https://github.com/pandas-dev/pandas | [models/models_statsml.py](models/models_statsml.py) |

Examples:

*AutoARIMA, KNN Regression, LightGBM, and Naive Moving Average*

```bash
python models/models_statsml.py --dataset london --sampling_rates 100 --models KNNRegression LightGBM ARIMA NaiveMovingAverage --runs 1 --results_dir results
```

*Naive Drift*

```bash
python models/model_naivedrift.py --dataset belgium --sampling_rates 100 --runs 1 --results_dir results
```

### 2. MLP-based Models

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| DeepNPTS | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| N-BEATS | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| NHITS | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| NLinear | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| TiDE | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |

Example:

*DeepNPTS, N-BEATS, NHITS, NLinear, and TiDE*

```bash
python models/models_neuralforecast.py --gpu 0 --dataset belgium --sampling_rates 100 --models DeepNPTS NBEATS NHITS NLinear TiDE --runs 10 --epochs 100 --results_dir results
```

### 3. Recurrent Networks

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| DeepAR | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |
| DeepFactor | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |
| MQ-RNN | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |
| Mamba | mamba-ssm | https://github.com/state-spaces/mamba | [models/model_mamba.py](models/model_mamba.py) |
| Temporal Fusion Transformer (TFT) | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |

Examples:

*DeepAR, DeepFactor, MQ-RNN, and Temporal Fusion Transformer*

```bash
python models/models_gluonts.py --gpu 0 --dataset belgium --sampling_rates 100 --models DeepAR DeepFactor MQRNN TemporalFusionTransformer --runs 10 --epochs 100 --results_dir results
```

*Mamba*

```bash
python models/model_mamba.py --gpu 0 --dataset belgium --sampling_rates 100 --runs 10 --epochs 100 --results_dir results
```

### 4. Convolutional Networks

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| TCN | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| BiTCN | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| TimesNet | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| WaveNet | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |
| MQ-CNN | GluonTS | https://github.com/awslabs/gluonts | [models/models_gluonts.py](models/models_gluonts.py) |

Examples:

*TCN, BiTCN, and TimesNet*

```bash
python models/models_neuralforecast.py --gpu 0 --dataset belgium --sampling_rates 100 --models TCN BiTCN TimesNet --runs 10 --epochs 100 --results_dir results
```

*WaveNet and MQ-CNN*

```bash
python models/models_gluonts.py --gpu 0 --dataset belgium --sampling_rates 100 --models WaveNet MQCNN --runs 10 --epochs 100 --results_dir results
```

### 5. Transformer-based Models

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| Informer | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| PatchTST | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| iTransformer | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| Vanilla Transformer | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |
| TimeXer | NeuralForecast | https://github.com/Nixtla/neuralforecast | [models/models_neuralforecast.py](models/models_neuralforecast.py) |

Example:

*Informer, PatchTST, iTransformer, Vanilla Transformer, and TimeXer*

```bash
python models/models_neuralforecast.py --gpu 0 --dataset belgium --sampling_rates 100 --models Informer PatchTST iTransformer VanillaTransformer TimeXer --runs 10 --epochs 100 --results_dir results
```

### 6. Time Series Foundation Models

| Model | Library | Repository | Script |
|------|---------|------------|--------|
| TimeGPT | nixtla | https://github.com/Nixtla/nixtla | [models/fmodel_timegpt.py](models/fmodel_timegpt.py) |
| TimesFM | timesfm | https://github.com/google-research/timesfm | [models/fmodel_timesfm.py](models/fmodel_timesfm.py) |
| MOIRAI | uni2ts | https://github.com/SalesforceAIResearch/uni2ts | [models/fmodel_moirai.py](models/fmodel_moirai.py) |
| Chronos | chronos-forecasting | https://github.com/amazon-science/chronos-forecasting | [models/fmodel_chronos.py](models/fmodel_chronos.py) |
| Timer-XL | transformers | https://github.com/thuml/Timer-XL | [models/fmodel_timerxl.py](models/fmodel_timerxl.py) |

Examples:

*TimeGPT*

```bash
python models/fmodel_timegpt.py --dataset belgium --sampling_rates 100 --runs 1 --api_key nixak-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX --model timegpt-1-long-horizon --confidence_level 95 --finetune_steps 0 --finetune_depth 1 --results_dir results
```

*TimesFM*

```bash
python models/fmodel_timesfm.py --gpu 0 --dataset belgium --sampling_rates 100 --runs 1 --batch_size 32 --model_id google/timesfm-1.0-200m-pytorch --results_dir results
```

*MOIRAI*

```bash
python models/fmodel_moirai.py --gpu 0 --dataset belgium --sampling_rates 100 --runs 1 --model moirai --size small --patch_size auto --batch_size 32 --results_dir results
```

*Chronos*

```bash
python models/fmodel_chronos.py --gpu 0 --dataset belgium --sampling_rates 100 --runs 1 --model_id amazon/chronos-bolt-small --results_dir results
```

*Timer-XL*

```bash
python models/fmodel_timerxl.py --gpu 0 --dataset belgium --sampling_rates 100 --runs 1 --model_id thuml/timer-base-84m --save_folder TimerXL --results_dir results
```

## Notes

- Some models can run on the CPU while others require GPU for practical runtimes.
- TimeGPT requires a valid API key supplied with `--api_key`.
- Chronos, TimesFM, MOIRAI, and Timer-XL download weights on first run.

## Results

All model outputs are saved in the `results_dir` folder with per-run plots and metrics summaries. The plotting and metrics utilities are in [utils/metrics.py](utils/metrics.py) and [utils/plots.py](utils/plots.py). Please refer to our paper for the results and references to all the models evaluated.
