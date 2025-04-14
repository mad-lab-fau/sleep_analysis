from pathlib import Path

import pandas as pd

from sleep_analysis.datasets.d04_main_dataset_control import D04MainStudy
from sleep_analysis.datasets.d04_main_dataset_pd import D04PDStudy
from sleep_analysis.datasets.helper import build_base_path_processed
from sleep_analysis.preprocessing.d04_main.respiration_radar import preprocess_respiration
from sleep_analysis.preprocessing.d04_main.utils import load_and_sync_radar_data


def extract_respiration_radar_per_subj(subj, fs_radar=1953.125):

    synced_radar = load_and_sync_radar_data(subj)

    if synced_radar is None:
        return None

    # Extract unique radar names dynamically
    radar_names = synced_radar.columns.get_level_values(0).unique()

    # Create dictionary of sub-dataframes
    radar_dict = {radar: synced_radar[radar] for radar in radar_names}

    resp_peaks_df = preprocess_respiration(radar_dict, fs_radar=fs_radar)

    return resp_peaks_df


# main method:
if __name__ == "__main__":
    dataset_name = "d04_control"

    if dataset_name == "d04_control":
        dataset = D04MainStudy()
    elif dataset_name == "d04_pd":
        dataset = D04PDStudy()
    else:
        raise ValueError("Unknown dataset_name")

    fs_radar = 1953.125
    for subj in dataset:
        subj_id = subj.index["subj_id"][0]
        print("Processing subject", subj_id, flush=True)

        if str(subj_id) == "09":
            print("Skip subject 09", flush=True)
            continue

        processed_folder = Path(build_base_path_processed(dataset_name).joinpath("Vp_" + subj_id))

        # check if folder exists
        if not processed_folder.exists():
            print("create folder ...", flush=True)
            processed_folder.mkdir(parents=True)

        # if file in folder exists, skip
        if processed_folder.joinpath("resp_peaks_df" + str(subj_id) + ".csv").exists():
            continue

        resp_df = extract_respiration_radar_per_subj(subj, fs_radar=fs_radar)

        if resp_df is None:
            continue

        resp_df.to_csv(processed_folder.joinpath("resp_df" + str(subj_id) + ".csv"))

        # Filter rows where Peaks is 1
        df_peaks = resp_df[resp_df["Peaks"] == 1]

        # Calculate peak-to-peak intervals
        df_peaks["peak_to_peak_interval"] = df_peaks.index.to_series().diff().dt.total_seconds()

        # Drop the first row as it will have a NaN interval
        df_peaks = df_peaks.dropna()

        df_peaks.to_csv(processed_folder.joinpath("resp_peaks_df" + str(subj_id) + ".csv"))
        movement_features.to_csv(processed_folder.joinpath("movement_features_" + str(subj_id) + ".csv"))
