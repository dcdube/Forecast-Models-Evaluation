# Datasets

This directory contains the household energy time series used by the forecasting benchmark. The dataset loaders are implemented in [`utils/dataset_config.py`](../utils/dataset_config.py).

| Dataset | Forecasting target | Series used by the benchmark | Time span | Frequency | Unit |
|---|---|---:|---|---:|---|
| Belgium | PV generation and BESS power | 4 houses | 1 January–31 March 2024 | 15 minutes | kW |
| Germany WPUQ | Household electricity demand | SFH19 | 1 January–31 March 2024 | 15 minutes | kW |
| London LCL | Household electricity demand | MAC000033 | 1 January–31 December 2013 | 30 minutes | kWh per half-hour |
| Zonnedael | Household electricity demand | Customers 8, 9, and 43 | 1 January–30 June 2013 | 15 minutes | Wh per 15 minutes |

## Belgium

Original data can be found [here](https://github.com/EVERGi/real_validation_saferl_treec_paper/tree/main/data/houses).

[`belgium_dataset`](belgium_dataset) contains PV generation and battery power measurements for four houses. Each house has separate `solar.csv` and `battery.csv` files with a UTC `datetime` column and one target column:

- `SolarPv_0 (kW)` for PV generation;
- `Battery_0 (kW)` for BESS power.

Two measurement windows are included for every house: `2024-01-01_0000_2024-04-01_0000` and `2024-04-08_1500_2024-06-17_1500`. The benchmark scripts uses the first window, which contains 8,736 expected 15-minute timestamps. Some battery files contain missing timestamps or values; the loaders remove battery outliers using an absolute z-score threshold of 3 and linearly interpolate missing values.

## Germany WPUQ

Original data can be found [here](https://springernature.figshare.com/articles/dataset/Metadata_record_for_Dataset_on_electrical_single-family_house_and_heat_pump_load_profiles_in_Germany/17206271).

[`germany_wpuq_dataset`](germany_wpuq_dataset) contains the electricity demand of the WPUQ single-family house SFH19. The benchmark loads [`SFH19_2023_2024_15min_3_month.csv`](germany_wpuq_dataset/SFH19_2023_2024_15min_3_month.csv), which has 8,736 continuous UTC observations from January through March 2024:

- `datetime`: UTC timestamp;
- `Consumer_0_electric (kW)`: household electricity demand in kW.

The directory also retains [`SFH19_2018_2019_15min_original.csv`](germany_wpuq_dataset/SFH19_2018_2019_15min_original.csv), a longer 57,066-row series covering 16 May 2018 through 31 December 2019. 

## London LCL

Original data can be found [here](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households).

[`london_dataset`](london_dataset) contains half-hourly smart-meter consumption for the anonymized household `MAC000033` on the standard (`Std`) tariff during 2013. The columns are:

- `LCLid`: anonymized household identifier;
- `stdorToU`: tariff group;
- `DateTime`: local timestamp;
- `KWH/hh (per half hour)`: electricity consumed during the half-hour interval.

The benchmark uses [`LCL_london_consumption_2013.csv`](london_dataset/LCL_london_consumption_2013.csv), which is a clean version containing 17,507 observations. [`LCL_london_consumption_2013_original.csv`](london_dataset/LCL_london_consumption_2013_original.csv) is retained as the original variant and includes duplicated timestamps.

## Zonnedael

Original data can be found [here](https://www.liander.nl/over-ons/open-data#verbruiksdata-slimme-meter).

The Zonnedael data were collected from smart meters at approximately 80 addresses in the Netherlands. Participants consented to their smart meters being read for research. The broader collection contains electricity measurements at 15-minute intervals from 1 May 2012 through 7 March 2014 and gas measurements at hourly intervals from 1 May 2012 through 7 January 2014. Electricity is expressed in Wh per 15 minutes and gas in m³ per hour. Timestamps use local Dutch time, including daylight-saving time transitions.

Missing or unrealistic readings were corrected and may therefore be represented by average values rather than actual consumption. Series that were incomplete or contained unrealistic values were removed. In the original collection, the anonymized customer identifier can be used with the separate `Klanttypering` (customer classification) file to associate a connection with its household type, dwelling type, and dwelling age; that classification file is not included in this repository.

The bundled [`liander_zonnedael_2013_original.csv`](zonnedael_dataset/liander_zonnedael_2013_original.csv) is an electricity-only subset covering 1 January through 30 June 2013. It contains 17,376 complete 15-minute timestamps and 56 anonymized household columns named `Klant <number>` (`Klant` means customer in Dutch). The benchmark evaluates `Klant 8`, `Klant 9`, and `Klant 43`.

## Loading and preprocessing

All targets are min–max normalized independently when loaded. The statistical and machine-learning loaders also derive time-of-day features, while the NeuralForecast-compatible loaders return the standard `unique_id`, `ds`, and `y` columns. Model scripts may further subsample the series according to `--sampling_rates`; the CSV files in this directory remain at their native resolution.
