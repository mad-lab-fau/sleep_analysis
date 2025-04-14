from sleep_analysis.datasets.d04_main_dataset_control import D04MainStudy
import numpy as np
import pandas as pd
import os
from empkins_io.sync import SyncedDataset

from biopsykit.signals.ecg import EcgProcessor
from empkins_micro.emrad.radar import get_rpeaks, get_peak_probabilities
from biopsykit.utils.exceptions import EcgProcessingError


### General Settings


# General Radar Settings
fs_radar = 1953.125
window_size = 30

# General PSG settings
clean_method = "biosppy"
peak_method = "neurokit"


def cut_signals(hr_ecg, hr_radar):
    start = max(hr_ecg.index[0], hr_radar.index[0])
    end = min(hr_ecg.index[-1], hr_radar.index[-1])

    return hr_ecg[start:end], hr_radar[start:end]


def process_radar(synced_radar):
    print("-------------------------------------------------")
    print("Processing participant " + subj.index["subj_id"][0])
    print("-------------------------------------------------")

    print("Radar 1")
    lstm_output_1 = get_peak_probabilities(
        synced_radar["rad1_aligned_resampled_"][["I", "Q"]], fs_radar=fs_radar, window_size=window_size
    )
    print("Radar 2")
    lstm_output_2 = get_peak_probabilities(
        synced_radar["rad2_aligned_resampled_"][["I", "Q"]], fs_radar=fs_radar, window_size=window_size
    )
    print("Radar 3")
    lstm_output_3 = get_peak_probabilities(
        synced_radar["rad3_aligned_resampled_"][["I", "Q"]], fs_radar=fs_radar, window_size=window_size
    )
    print("Radar 4")
    lstm_output_4 = get_peak_probabilities(
        synced_radar["rad4_aligned_resampled_"][["I", "Q"]], fs_radar=fs_radar, window_size=window_size
    )

    return {
        "lstm_output_1": lstm_output_1,
        "lstm_output_2": lstm_output_2,
        "lstm_output_3": lstm_output_3,
        "lstm_output_4": lstm_output_4,
    }


def get_MAE_results(probability_dict, hr_ecg_10s, threshold_list):
    MAE_results = {}
    for threshold in threshold_list:
        try:
            r_peaks_radar, lstm_probability = get_rpeaks(
                probability_dict, fs_radar=fs_radar, outlier_correction=True, threshold=threshold
            )
        except EcgProcessingError:
            MAE_results[threshold] = np.nan
            continue

        hr_radar = pd.DataFrame({"Heart_Rate": 60 / r_peaks_radar["RR_Interval"]})
        hr_radar.index = hr_radar.index.floor("10s")
        hr_radar_10s = hr_radar.groupby("date (Europe/Berlin)").mean()
        hr_radar_10s = hr_radar_10s.interpolate().rolling(20, center=True, min_periods=1).mean()

        hr_ecg_10s, hr_radar_10s = cut_signals(hr_ecg_10s, hr_radar_10s)

        MAE = abs(hr_ecg_10s - hr_radar_10s).mean()
        MAE_results[threshold] = MAE

    return MAE_results, lstm_probability


def get_hr_ecg(subj):
    ecg_data = subj.ecg_data.data_as_df(index="local_datetime")[["ECG II"]]
    ecg_data = ecg_data.rename(columns={"ECG II": "ecg"})
    ep = EcgProcessor(ecg_data, 256)
    ep.ecg_process(outlier_correction=None, clean_method=clean_method, peak_mathod=peak_method)
    hr_ecg = ep.heart_rate["Data"]
    hr_ecg.index = hr_ecg.index.floor("10s")
    hr_ecg_10s = hr_ecg.groupby("date (Europe/Berlin)").mean()
    hr_ecg_10s = hr_ecg_10s.rolling(20, center=True, min_periods=1).mean()

    return hr_ecg_10s


if __name__ == "__main__":

    dataset = D04MainStudy()

    threshold_list = [
        0.050,
        0.075,
        0.1,
        0.125,
        0.150,
        0.175,
        0.200,
        0.225,
        0.250,
        0.255,
        0.260,
        0.265,
        0.270,
        0.275,
        0.280,
        0.285,
        0.290,
        0.295,
        0.300,
        0.325,
        0.350,
        0.375,
        0.400,
    ]
    id_list = [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "10",
        "11",
        "12",
        "14",
        "16",
        "18",
        "19",
        "20",
        "21",
        "22",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "31",
        "32",
        "36",
        "37",
        "38",
        "41",
        "42",
        "43",
        "44",
    ]

    for subj in dataset:

        if str(subj.index["subj_id"][0]) not in id_list:
            continue

        file_path = "MAE_gridsearch_subj" + str(window_size) + "_" + str(subj.index["subj_id"][0]) + ".csv"

        # Check if the file exists
        file_exists = os.path.isfile(file_path)
        if file_exists:
            print("File for subj " + str(subj.index["subj_id"][0]) + " already existis ... skip!")
            continue

        radar_data = subj.radar_data.data_as_df(index="local_datetime", add_sync_out=True)
        synced_radar = subj.sync_radar(radar_data)

        probability_dict = process_radar(synced_radar)

        hr_ecg_10s = get_hr_ecg(subj)

        MAE_dict, lstm_probabiliy = get_MAE_results(probability_dict, hr_ecg_10s, threshold_list)

        print("MAE of subj " + str(subj.index["subj_id"][0]))
        print(MAE_dict)

        pd.DataFrame(MAE_dict).to_csv(file_path)
