import numpy as np
import pandas as pd
from hrvanalysis import interpolate_nan_values, remove_ectopic_beats, remove_outliers


def process_rpoint(ecg_df):
    # remove the noise data points if two peaks overlapped or not wear
    ecg_df = ecg_df[ecg_df["TPoint"] > 0]
    # Define RR intervals by taking the difference between each one of the measurements in seconds (*1k to get milliseconds)
    rr_intervals = pd.DataFrame(ecg_df["seconds"].diff() * 1000)
    rr_intervals = rr_intervals.rename(columns={"seconds": "RR Intervals"})

    rr_intervals["RR Intervals"].fillna(rr_intervals["RR Intervals"].mean(), inplace=True)  # fill mean for first sample

    # apply HRV-Analysis package
    # filter any hear rate over 60000/300 = 200, 60000/2000 = 30
    clean_rri = rr_intervals["RR Intervals"].values
    clean_rri = remove_outliers(rr_intervals=clean_rri, low_rri=300, high_rri=2000, verbose=False)
    clean_rri = interpolate_nan_values(rr_intervals=clean_rri, interpolation_method="linear")
    clean_rri = remove_ectopic_beats(rr_intervals=clean_rri, method="malik", verbose=False)
    clean_rri = interpolate_nan_values(rr_intervals=clean_rri)

    rr_intervals["RR Intervals"] = clean_rri
    # calculate the Heart Rate

    hr_df = pd.DataFrame(np.round((60000.0 / rr_intervals["RR Intervals"]), 0))
    hr_df = hr_df.rename(columns={"RR Intervals": "HR"})

    rr_intervals["RR Intervals"].fillna(
        rr_intervals["RR Intervals"].mean(), inplace=True
    )  # eventually fill mean for first samples
    ecg_df = pd.concat([ecg_df, hr_df], axis=1)
    ecg_df = pd.concat([ecg_df, rr_intervals], axis=1)

    # filter RRI
    t1 = ecg_df.epoch.value_counts().reset_index().rename({"index": "epoch_idx", "epoch": "count"}, axis=1)
    invalid_idx = set(t1[t1["count"] < 10]["epoch_idx"].values)
    del t1
    ecg_df = ecg_df[~ecg_df["epoch"].isin(list(invalid_idx))]
    # get intersect epochs

    return ecg_df
