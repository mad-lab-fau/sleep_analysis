import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, hilbert, filtfilt


def preprocess_respiration_df(radar_df: pd.DataFrame, fs_radar) -> pd.DataFrame:

    nyquist = 0.5 * fs_radar
    lowpass_cutoff = 0.5 / nyquist

    radar_values = radar_df["Q"].values

    b_lp, a_lp = butter(2, lowpass_cutoff, btype="low")
    smoothed_phase = filtfilt(b_lp, a_lp, radar_values)

    envelope = np.abs(hilbert(smoothed_phase))
    peaks, _ = find_peaks(envelope, distance=2.4 * fs_radar)

    envelope = pd.DataFrame(envelope, index=radar_df.index)

    envelope["Peaks"] = 0
    envelope.iloc[peaks, envelope.columns.get_loc("Peaks")] = 1

    return envelope


def preprocess_respiration(radar_dict: dict, fs_radar) -> tuple:

    # Calculate movement for each radar datastream
    for key, radar_df in radar_dict.items():
        radar_dict[key] = preprocess_respiration_df(radar_df, fs_radar=fs_radar)

    # only use radar datastream 3 as no meaningful aggregation implemented

    respiration_df = radar_dict["rad3_aligned_resampled_"]

    return respiration_df
