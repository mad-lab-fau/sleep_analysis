import json
import re
from pathlib import Path

import biopsykit as bp
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import tqdm
from biopsykit.utils.array_handling import sliding_window

from sleep_analysis.feature_extraction.mesa_datasst.utils import check_processed
from sleep_analysis.preprocessing.utils import extract_edf_channel


def extract_rrv_features(overwrite=False):
    """
    Extracts RRV features from the MESA dataset.
    Calculates 62 different RRV features in sliding windows of 5 min (10 epochs), 7 min (14 epochs) and 9 min (18 epochs) with overlap of 30s (1 epoch).

    :param overwrite: If overwrite = True, the features are calculated and overwritten. If set to false, all features that are calculated are skipped.
    """

    with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
        path_dict = json.load(f)
        edf_path = Path(path_dict["mesa_path"]).joinpath("polysomnography/edfs")
        processed_mesa_path = Path(path_dict["processed_mesa_path"])

    path_list = list(Path(edf_path).glob("*.edf"))
    mesa_id = re.findall("(\d{4})", str(path_list))

    with tqdm.tqdm(total=len(mesa_id)) as progress_bar:
        for subj in mesa_id:
            if not overwrite:  # check if file already exists
                if check_processed(Path(path_dict["processed_mesa_path"]).joinpath("respiration_features_raw"), subj):
                    progress_bar.update(1)  # update progress
                    continue

            # read in .edf file and process to respiration dataframe
            # tmin and tmax are variables to check edf file streams (crops to a smaller part)
            resp_df, epochs = extract_edf_channel(edf_path, subj_id=int(subj), channel="Thor")
            resp_df, epochs = process_resp(resp_df, epochs)
            features = extract_rrv_features_helper(resp_df)
            features.to_csv(
                Path(path_dict["processed_mesa_path"]).joinpath(
                    "respiration_features_raw/respiration" + str(subj) + ".csv"
                )
            )
            progress_bar.update(1)  # update progress

            print("Features extraction of RRV of subj: " + subj + " finished!")


def extract_rrv_features_helper(resp_arr, nan_pad=1.0, sampling_rate=32):
    """
    Calculate features for sliding widows of 5 min (10 epochs), 7 min (14 epochs) and 9 min (18 epochs) with overlap of 30s (1 epoch)
    according to Fonseca et al., 2015
    zero-padding at beginning and end because of sliding window
    Feature DataFrame with prefix 150_, 210_ and 270_ for features of 5 min, 7 min and 9-min window size

    :param resp_arr: respiration datastream
    :param nan_pad: value to pad NaN values with
    """

    # resp_arr_30s = sliding_window(resp_arr, 30*32, overlap_samples=0)

    # mode=mean to prevent first epochs to be zero --> no breathing extracted --> exception
    resp_arr_150s = np.nan_to_num(
        np.pad(
            sliding_window(resp_arr, 150 * sampling_rate, overlap_samples=120 * sampling_rate),
            ((2, 2), (0, 0)),
            mode="mean",
        ),
        nan=nan_pad,
    )
    resp_arr_210s = np.nan_to_num(
        np.pad(
            sliding_window(resp_arr, 210 * sampling_rate, overlap_samples=180 * sampling_rate),
            ((3, 3), (0, 0)),
            mode="mean",
        ),
        nan=nan_pad,
    )
    resp_arr_270s = np.nan_to_num(
        np.pad(
            sliding_window(resp_arr, 270 * sampling_rate, overlap_samples=240 * sampling_rate),
            ((4, 4), (0, 0)),
            mode="mean",
        ),
        nan=nan_pad,
    )

    feature_list = []
    for resp_150 in resp_arr_150s:
        try:
            peaks = extract_peaks(resp_150, sampling_rate)
            features_150 = calc_rrv_features(resp_150, peaks, sampling_rate)
            feature_list.append(features_150[0])
        except ValueError:
            print("handle Value-Error")
            feature_list.append(dict.fromkeys(features_150[0], 0))
            continue
        except IndexError:
            print("handle Index-Error")
            feature_list.append(dict.fromkeys(features_150[0], 0))
            continue

    features = pd.DataFrame(feature_list).add_prefix("150_")

    feature_list = []
    for resp_210 in resp_arr_210s:
        try:
            peaks = extract_peaks(resp_210, sampling_rate)
            features_210 = calc_rrv_features(resp_210, peaks, sampling_rate)
            feature_list.append(features_210[0])

        except ValueError:
            print("handle Value-Error")
            feature_list.append(dict.fromkeys(features_210[0], 0))
            continue
        except IndexError:
            print("handle Index-Error")
            feature_list.append(dict.fromkeys(features_210[0], 0))
            continue

    features = pd.concat([features, pd.DataFrame(feature_list).add_prefix("210_")], axis=1)

    feature_list = []
    for resp_270 in resp_arr_270s:
        try:
            peaks = extract_peaks(resp_270, sampling_rate)
            features_270 = calc_rrv_features(resp_270, peaks, sampling_rate)
            feature_list.append(features_270[0])

        except ValueError:
            print("handle Value-Error")
            feature_list.append(dict.fromkeys(features_270[0], 0))
            continue
        except IndexError:
            print("handle Index-Error")
            feature_list.append(dict.fromkeys(features_270[0], 0))
            continue

    features = pd.concat([features, pd.DataFrame(feature_list).add_prefix("270_")], axis=1)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    features.fillna(0.0, inplace=True)

    time_axis = resp_arr.index.round("30s").drop_duplicates()[0 : features.shape[0]]
    features.index = time_axis
    features["epoch"] = np.arange(1, features.shape[0] + 1)

    if features.shape[1] == 70:
        print("length 70")
        features = features[features.columns.drop(list(features.filter(regex="210_RRV_DFA")))]

    return features


def calc_rrv_features(rsp_rate, peaks_dict, sampling_rate: int):
    """
    RRV (Respiratory Rate Variability) Features extracted via python library neurokit
    :param: rsp_rate: extracted rsp_rate from edf file
            peaks_dict: extracted peaks from rsp_signal
            sampling_rate: sampling rate of signal - commonly 32 Hz
    :return: Features compressed in a dict
    """
    rrv = nk.rsp_rrv(rsp_rate, peaks_dict, sampling_rate=sampling_rate, show=False)

    return rrv.to_dict("records")


def process_resp(resp_df, epochs):
    time_index = resp_df.index

    sampling_rate_in = 256
    sampling_rate_out = 32
    resp_arr = _downsample_resp(resp_df, sampling_rate_in=sampling_rate_in, sampling_rate_out=sampling_rate_out)
    epochs = epochs[:: int(sampling_rate_in / sampling_rate_out)]
    time_index = time_index[:: int(sampling_rate_in / sampling_rate_out)]

    resp_df = pd.DataFrame(resp_arr, index=time_index)

    # peaks = extract_peaks(resp_arr,32)
    # _plot_features(resp_arr,peaks,32)
    # _plot_rsp_rate(resp_arr,peaks,32)

    return resp_df, epochs


def _downsample_resp(resp_df, sampling_rate_in: int, sampling_rate_out: int):
    cleaned = nk.rsp_clean(resp_df, sampling_rate=sampling_rate_in, method="biosppy")

    return bp.utils.array_handling.downsample(np.asarray(cleaned), sampling_rate_in, sampling_rate_out)


def extract_peaks(resp_df, sampling_rate: int):
    # Extract peaks
    df, peaks_dict = nk.rsp_peaks(resp_df, sampling_rate=sampling_rate, method="biosppy")
    info = nk.rsp_fixpeaks(peaks_dict)
    # formatted = nk.signal_formatpeaks(info, desired_length=len(resp_df), peak_indices=info["RSP_Peaks"])

    # candidate_peaks = nk.events_plot(peaks_dict["RSP_Peaks"], resp_df)
    # fixed_peaks = nk.events_plot(info["RSP_Peaks"], resp_df)

    return info


def _extract_rsp_rate(resp_df, peaks_dict, sampling_rate: int):
    # Extract rate
    rsp_rate = nk.rsp_rate(resp_df, peaks_dict, sampling_rate=sampling_rate)

    return rsp_rate


def _plot_rsp_rate(rsp_rate, peaks, sampling_rate: int):
    # Visualize
    nk.signal_plot(rsp_rate, sampling_rate=sampling_rate)
    plt.rcParams["figure.figsize"] = 15, 5  # Bigger images

    candidate_peaks = nk.events_plot(peaks["RSP_Peaks"], rsp_rate)
    plt.xlabel("Samples (32 Hz)")
    plt.title("Breath detection from breathing cycle acquired by respiratory band")
    plt.show()


def _plot_features(rsp_rate, peaks_dict, sampling_rate: int):
    rrv = nk.rsp_rrv(rsp_rate, peaks_dict, sampling_rate=sampling_rate, show=True)
    plt.show()
