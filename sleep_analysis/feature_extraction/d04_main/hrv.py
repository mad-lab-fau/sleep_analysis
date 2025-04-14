from pathlib import Path

import numpy as np
import pandas as pd
from hrvanalysis import (
    get_time_domain_features,
    get_frequency_domain_features,
    get_poincare_plot_features,
    get_csi_cvi_features,
    get_geometrical_features,
)

from sleep_analysis.datasets.d04_main_dataset_control import D04MainStudy
from sleep_analysis.datasets.helper import build_base_path_processed


def get_hrv_features(r_peak_df: pd.DataFrame):
    r_peak_df.index = pd.to_datetime(r_peak_df.index)
    r_peak_df = r_peak_df.dropna()[["R_Peak_Idx", "RR_Interval"]]

    hrv_features_30 = get_hrv_features_windows(r_peak_df, window=30)
    hrv_features_150 = get_hrv_features_windows(r_peak_df, window=150)
    hrv_features_210 = get_hrv_features_windows(r_peak_df, window=210)
    hrv_features_270 = get_hrv_features_windows(r_peak_df, window=270)

    hrv_features_30 = hrv_features_30.add_prefix("30_hrv_")
    hrv_features_150 = hrv_features_150.add_prefix("150_hrv_")
    hrv_features_210 = hrv_features_210.add_prefix("210_hrv_")
    hrv_features_270 = hrv_features_270.add_prefix("270_hrv_")

    hrv_features_df = pd.concat([hrv_features_30, hrv_features_150, hrv_features_210, hrv_features_270], axis=1)
    return hrv_features_df


def get_hrv_features_windows(r_peak_df: pd.DataFrame, window: int):
    windows = _overlapping_windows(r_peak_df, window)

    # movement_features = pd.read_csv("/Users/danielkrauss/code/Empkins/sleep-analysis/experiments/Feature_extraction/movement_features_04.csv", index_col=0, parse_dates=True)
    movement_features = pd.read_csv(
        processed_folder.joinpath("movement_features_" + str(subj_id) + ".csv"), index_col=0, parse_dates=True
    )
    index = movement_features.index

    epochs = len(index)  # Number of epochs

    columns = [
        "mean_nni",
        "sdnn",
        "sdsd",
        "nni_50",
        "pnni_50",
        "nni_20",
        "pnni_20",
        "rmssd",
        "median_nni",
        "range_nni",
        "cvsd",
        "cvnni",
        "mean_hr",
        "max_hr",
        "min_hr",
        "std_hr",
        "lf",
        "hf",
        "lf_hf_ratio",
        "lfnu",
        "hfnu",
        "total_power",
        "vlf",
        "sd1",
        "sd2",
        "ratio_sd2_sd1",
        "csi",
        "cvi",
        "Modified_csi",
        "triangular_index",
        "tinn",
    ]

    # Initialize the DataFrame with zeros
    feature_data = np.empty((epochs, len(columns)))
    feature_data.fill(np.nan)
    df_features = pd.DataFrame(feature_data, columns=columns, index=index)

    keys = windows.keys()

    for key in keys:
        # display(windows[key])

        RR_values = windows[key]["RR_Interval"].values * 1000
        all_hr_features = {}
        try:
            all_hr_features.update(get_time_domain_features(RR_values))
            all_hr_features.update(get_frequency_domain_features(RR_values))
            all_hr_features.update(get_poincare_plot_features(RR_values))
            all_hr_features.update(get_csi_cvi_features(RR_values))
            all_hr_features.update(get_geometrical_features(RR_values))

            all_hr_features = pd.DataFrame(all_hr_features, index=[key])

            # display(all_hr_features.columns)
            df_features.loc[key] = all_hr_features.loc[key]
        except:
            continue

    df_features = df_features.drop("tinn", axis=1)
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.interpolate(limit_direction="both")
    df_features = df_features.fillna(0.0)

    return df_features


def _overlapping_windows(df, window_seconds):

    # Calculate overlap of window size -30s to get a features for every 30 seconds
    overlap_seconds = window_seconds - 30
    window_size = pd.Timedelta(seconds=window_seconds)
    step_size = pd.Timedelta(seconds=(window_seconds - overlap_seconds))

    start_time = df.index.min()
    end_time = df.index.max()

    windows_dict = {}

    # Generate windows with overlap
    while start_time + window_size <= end_time:
        # Floor the start_time to the nearest 30-second mark
        floored_start_time = start_time.floor("30S")

        # Only proceed if the key (floored_start_time) doesn't exist in the dictionary
        if floored_start_time not in windows_dict:
            window = df.loc[start_time : start_time + window_size]
            if not window.empty:
                # Store the window in the dictionary with the floored_start_time as the key
                windows_dict[floored_start_time] = window

        # Move to the next window based on the step size
        start_time += step_size

    return windows_dict


if __name__ == "__main__":
    dataset = D04MainStudy()

    for subj in dataset:
        subj_id = subj.index["subj_id"][0]
        processed_folder = Path(build_base_path_processed().joinpath("Vp_" + subj_id))

        if not processed_folder.joinpath("r_peaks" + str(subj_id) + ".csv").exists():
            print("Heart sound data not found for subject", subj_id, flush=True)
            continue

        # if file in folder exists, skip
        # if processed_folder.joinpath("hrv_features_" + str(subj_id) + ".csv").exists():
        #    print("Features already extracted for subject", subj_id, flush=True)
        #    continue

        r_peaks = pd.read_csv(
            processed_folder.joinpath("r_peaks" + str(subj_id) + ".csv"), index_col=0, parse_dates=True
        )
        # r_peaks = pd.read_csv("/Users/danielkrauss/code/Empkins/sleep-analysis/experiments/Feature_extraction/r_peaks04.csv", index_col=0, parse_dates=True)

        hrv_features = get_hrv_features(r_peaks)

        hrv_features.to_csv(processed_folder.joinpath("hrv_features_" + str(subj_id) + ".csv"))
