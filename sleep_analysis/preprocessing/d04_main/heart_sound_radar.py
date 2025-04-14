import pandas as pd
from empkins_micro.emrad.radar import get_peak_probabilities, get_rpeaks


def preprocess_heart_sound_df(radar_df: pd.DataFrame, window_size=300, fs_radar=1953.125) -> pd.DataFrame:

    peak_probability = get_peak_probabilities(radar_df[["I", "Q"]], fs_radar=fs_radar, window_size=window_size)

    return peak_probability


def preprocess_heart_sound(radar_dict: dict, threshold: float = 0.18, window_size=300, fs_radar=1953.125) -> tuple:

    for key, radar_df in radar_dict.items():
        print("get peak probabilities for radar datastream: " + str(key), flush=True)

        radar_dict[key] = preprocess_heart_sound_df(radar_df, window_size=window_size, fs_radar=fs_radar)

    r_peaks_radar, lstm_probability = get_rpeaks(
        radar_dict, fs_radar=fs_radar, outlier_correction=True, threshold=threshold
    )

    return r_peaks_radar, lstm_probability
