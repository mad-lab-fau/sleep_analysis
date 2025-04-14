from pathlib import Path

import numpy as np
import pandas as pd
from biopsykit.utils.array_handling import sliding_window
import neurokit2 as nk

from sleep_analysis.datasets.d04_main_dataset_control import D04MainStudy
from sleep_analysis.datasets.helper import build_base_path_processed


def get_resp_features(df_resp_signal: pd.DataFrame, fs_radar: int = 1953.125):

    resp_features_150 = get_resp_features_windows(df_resp_signal, window_size=150, fs_radar=fs_radar)
    resp_features_210 = get_resp_features_windows(df_resp_signal, window_size=210, fs_radar=fs_radar)
    resp_features_270 = get_resp_features_windows(df_resp_signal, window_size=270, fs_radar=fs_radar)

    resp_features_150 = resp_features_150.add_prefix("150_")
    resp_features_210 = resp_features_210.add_prefix("210_")
    resp_features_270 = resp_features_270.add_prefix("270_")

    df_resp_features = pd.concat([resp_features_150, resp_features_210, resp_features_270], axis=1)
    return df_resp_features


def _check_pad_length(resp_signal, window_size: int, pad_length, fs_radar: int = 1953.125):

    rolling_window_shape = np.pad(
        sliding_window(resp_signal["Peaks"], window_size * fs_radar, overlap_samples=(window_size - 30) * fs_radar),
        pad_length,
        mode="constant",
        constant_values=(0,),
    ).shape

    datetime_shape = resp_signal.index.floor("30s").drop_duplicates().shape

    if datetime_shape[0] - rolling_window_shape[0] == 1:
        if window_size == 150:
            return (2, 3), (0, 0)
        elif window_size == 210:
            return (3, 4), (0, 0)
        elif window_size == 270:
            return (4, 5), (0, 0)
    elif datetime_shape[0] - rolling_window_shape[0] == 2:
        if window_size == 150:
            return (3, 3), (0, 0)
        elif window_size == 210:
            return (4, 4), (0, 0)
        elif window_size == 270:
            return (5, 5), (0, 0)
    if datetime_shape[0] - rolling_window_shape[0] == -1:
        if window_size == 150:
            return (2, 1), (0, 0)
        elif window_size == 210:
            return (3, 2), (0, 0)
        elif window_size == 270:
            return (4, 3), (0, 0)
    else:
        return pad_length


def get_resp_features_windows(df_resp_signal: pd.DataFrame, window_size: int, fs_radar: int = 1953.125):

    pad_length = _get_pad_length(window_size)

    pad_length = _check_pad_length(df_resp_signal, window_size, pad_length, fs_radar=fs_radar)

    peaks_rolling = np.nan_to_num(
        np.pad(
            sliding_window(
                df_resp_signal["Peaks"], window_size * fs_radar, overlap_samples=(window_size - 30) * fs_radar
            ),
            pad_length,
            mode="constant",
            constant_values=(0,),
        ),
        nan=0.0,
    )

    epochs = len(peaks_rolling)  # Number of epochs
    columns = [
        "RRV_RMSSD",
        "RRV_MeanBB",
        "RRV_SDBB",
        "RRV_SDSD",
        "RRV_CVBB",
        "RRV_CVSD",
        "RRV_MedianBB",
        "RRV_MadBB",
        "RRV_MCVBB",
        "RRV_VLF",
        "RRV_LF",
        "RRV_HF",
        "RRV_LFHF",
        "RRV_LFn",
        "RRV_HFn",
        "RRV_SD1",
        "RRV_SD2",
        "RRV_SD2SD1",
        "RRV_ApEn",
        "RRV_SampEn",
    ]

    # Initialize the DataFrame with zeros
    feature_data = np.empty((epochs, len(columns)))
    feature_data.fill(np.nan)

    df_features = pd.DataFrame(feature_data, columns=columns)

    for df_peaks_slice, idx in zip(peaks_rolling, np.arange(len(peaks_rolling))):

        nonzero = np.nonzero(df_peaks_slice)[0]
        peaks_dict = {"RSP_Troughs": list(nonzero)}

        time_intervals = np.diff(nonzero) / fs_radar  # Time in seconds between peaks
        respiration_rate_bpm = 60 / time_intervals  # Convert to breaths per minute

        # Initialize array for respiration rate with same shape as initial respiration signal
        # respiration_rate_array = np.zeros_like(df_peaks_slice)

        # We assign the calculated respiration rates between the indices of detected peaks
        # for i in range(len(nonzero) - 1):
        #    if i == 0:
        #        respiration_rate_array[:nonzero[i]] = respiration_rate_bpm[i]
        #    respiration_rate_array[nonzero[i]:nonzero[i + 1]] = respiration_rate_bpm[i]

        respiration_rate_array = nk.rsp_rate(df_peaks_slice, peaks_dict, sampling_rate=fs_radar)

        # To avoid features being calculated for unrealistic respiration rates, we skip the current epoch if any value is below 5.0
        if any(value < 5.0 for value in respiration_rate_bpm):
            continue

        try:
            rrv = nk.rsp_rrv(respiration_rate_array, peaks_dict, sampling_rate=fs_radar, show=False)
            df_features.loc[idx] = rrv.loc[0]
        except:
            continue

    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.interpolate(limit_direction="both")
    df_features = df_features.fillna(0.0)

    df_features = df_features.set_index(resp_signal.index.floor("30s").drop_duplicates(), drop=True)

    return df_features


def _get_pad_length(windowsize):
    if windowsize == 150:
        return (2, 2), (0, 0)
    elif windowsize == 210:
        return (3, 3), (0, 0)
    elif windowsize == 270:
        return (4, 4), (0, 0)


if __name__ == "__main__":
    dataset = D04MainStudy()

    for subj in dataset:
        subj_id = subj.index["subj_id"][0]
        processed_folder = Path(build_base_path_processed().joinpath("Vp_" + subj_id))

        if not processed_folder.joinpath("resp_df" + str(subj_id) + ".csv").exists():
            print("Respiration data not found for subject", subj_id, flush=True)
            continue

        # if file in folder exists, skip
        if processed_folder.joinpath("resp_features_" + str(subj_id) + ".csv").exists():
            print("Features already extracted for subject", subj_id, flush=True)
            continue

        resp_signal = pd.read_csv(
            processed_folder.joinpath("resp_df" + str(subj_id) + ".csv"), index_col=0, parse_dates=True
        )
        # resp_signal = pd.read_csv("/Users/danielkrauss/code/Empkins/sleep-analysis/experiments/Feature_extraction/resp_df04.csv", index_col=0, parse_dates=True)
        resp_features = get_resp_features(resp_signal)

        resp_features.to_csv(processed_folder.joinpath("resp_features_" + str(subj_id) + ".csv"))
