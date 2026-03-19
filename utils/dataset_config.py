import pandas as pd
import numpy as np
from models.metrics import min_max_normalize
from scipy import stats


class DatasetBelgiumNF:
    # Load PV data for specified range and reformat for NeuralForecast
    def get_inputs_for_pv(self, house, start_dt, end_dt):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        pv_file = f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/solar.csv"

        df = pd.read_csv(pv_file, parse_dates=["datetime"])

        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.rename(columns={"datetime": "ds", "SolarPv_0 (kW)": "y"})
        df["y"] = min_max_normalize(df[["y"]])

        df["unique_id"] = f"series_pv_house_{house}"
        return df[["unique_id", "ds", "y"]]

    def get_inputs_for_battery(self, house, start_dt, end_dt, freq="15min"):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        battery_file = f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/battery.csv"

        df = pd.read_csv(battery_file, parse_dates=["datetime"])

        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.rename(columns={"datetime": "ds", "Battery_0 (kW)": "y"})

        full_range = pd.date_range(start=start_dt, end=end_dt - pd.Timedelta(freq), freq=freq)
        df = df.set_index("ds").reindex(full_range)
        df.index.name = "ds"

        if df["y"].notna().sum() > 0:
            z_scores = np.abs(stats.zscore(df["y"], nan_policy="omit"))
            df.loc[z_scores >= 3, "y"] = np.nan

        df["y"] = df["y"].interpolate(method="linear", limit_direction="both")
        df["y"] = min_max_normalize(df[["y"]])

        df["unique_id"] = f"series_battery_house_{house}"
        df = df.reset_index()

        return df[["unique_id", "ds", "y"]]


class DatasetBelgium1D:
    # Load PV data for specified range and reformat for LightGBM
    def get_pv_data(self, house, start_dt, end_dt):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        pv_file = f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/solar.csv"

        df = pd.read_csv(pv_file, parse_dates=["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.sort_values("datetime").set_index("datetime")

        df["y"] = min_max_normalize(df[["SolarPv_0 (kW)"]])

        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["dayofweek"] = df.index.dayofweek
        df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        X = df[["quarter_hour"]]
        y = df["y"]
        return X, y

    # Load battery data for specified range and reformat for LightGBM
    def get_battery_data(self, house, start_dt, end_dt):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        battery_file = f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/battery.csv"

        df = pd.read_csv(battery_file, parse_dates=["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.sort_values("datetime").set_index("datetime")

        df["y"] = df["Battery_0 (kW)"]
        z_scores = np.abs(stats.zscore(df["y"], nan_policy="omit"))
        df.loc[z_scores >= 3, "y"] = np.nan

        df["y"] = df["y"].interpolate(method="linear", limit_direction="both")
        df["y"] = min_max_normalize(df[["y"]])

        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["dayofweek"] = df.index.dayofweek
        df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        X = df[["quarter_hour"]]
        y = df["y"]
        return X, y


class DatasetGermanyNF:
    # Load electric consumption data for specified range and reformat for NeuralForecast
    def get_inputs_for_load(self, start_dt, end_dt):
        load_file = "data/germany_wpuq_dataset/SFH19_2023_2024_15min_3_month.csv"
        load_df = pd.read_csv(load_file)
        load_df["datetime"] = pd.to_datetime(load_df["datetime"])

        df = load_df[
            (load_df["datetime"] >= start_dt) & (load_df["datetime"] < end_dt)
        ][["datetime", "Consumer_0_electric (kW)"]]

        df = df.rename(columns={"datetime": "ds", "Consumer_0_electric (kW)": "y"})
        df["y"] = min_max_normalize(df[["y"]])

        df["unique_id"] = "series_load"
        return df[["unique_id", "ds", "y"]]


class DatasetGermany1D:
    # Load electric consumption data for specified range and reformat for LightGBM
    def get_load_data(self, start_dt, end_dt):
        load_file = "data/germany_wpuq_dataset/SFH19_2023_2024_15min_3_month.csv"
        df = pd.read_csv(load_file)
        df["datetime"] = pd.to_datetime(df["datetime"])

        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.sort_values("datetime").set_index("datetime")

        df["y"] = min_max_normalize(df[["Consumer_0_electric (kW)"]])

        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["dayofweek"] = df.index.dayofweek
        df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        X = df[["quarter_hour"]]
        y = df["y"]
        return X, y


class DatasetLondonNF:
    # Load electric consumption data for specified range and reformat for NeuralForecast
    def get_inputs_for_load(self):
        london_consumption_file = "data/london_dataset/LCL_london_consumption_2013.csv"
        df = pd.read_csv(london_consumption_file)
        df["datetime"] = pd.to_datetime(df["DateTime"])

        df = df.rename(columns={"datetime": "ds", "KWH/hh (per half hour)": "y"})
        df["y"] = min_max_normalize(df[["y"]])

        df["unique_id"] = "series_load"
        return df[["unique_id", "ds", "y"]]


class DatasetLondon1D:
    # Load electric consumption data for specified range and reformat for LightGBM
    def get_load_data(self):
        london_consumption_file = "data/london_dataset/LCL_london_consumption_2013.csv"
        df = pd.read_csv(london_consumption_file)
        df["datetime"] = pd.to_datetime(df["DateTime"])
        df = df.sort_values("datetime").set_index("datetime")

        df["y"] = min_max_normalize(df[["KWH/hh (per half hour)"]])

        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["dayofweek"] = df.index.dayofweek
        df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        X = df[["quarter_hour"]]
        y = df["y"]
        return X, y


class DatasetZonnedaelNF:
    # Load electric consumption data for specified range and reformat for NeuralForecast
    def get_inputs_for_zonnedael_consumption(self, customer_number: int):
        zonnedael_consumption_file = "data/zonnedael_dataset/liander_zonnedael_2013_original.csv"
        df = pd.read_csv(zonnedael_consumption_file)

        df["datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
        df = df.dropna(subset=["datetime"])

        client_no = f"Klant {customer_number}"
        df = df.rename(columns={"datetime": "ds", client_no: "y"})
        df["y"] = min_max_normalize(df["y"])
        df["unique_id"] = f"series_load_{customer_number}"

        return df[["unique_id", "ds", "y"]]


class DatasetZonnedael1D:
    def get_inputs_for_zonnedael_consumption(self, customer_number: int):
        zonnedael_consumption_file = "data/zonnedael_dataset/liander_zonnedael_2013_original.csv"
        df = pd.read_csv(zonnedael_consumption_file)
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            format="%d-%m-%Y %H:%M",
            errors="coerce",
        )
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").set_index("datetime")

        customer_column = f"Klant {customer_number}"

        df["y"] = min_max_normalize(df[[customer_column]])

        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["dayofweek"] = df.index.dayofweek
        df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        X = df[["quarter_hour", "is_weekend"]]
        y = df["y"]
        return X, y
