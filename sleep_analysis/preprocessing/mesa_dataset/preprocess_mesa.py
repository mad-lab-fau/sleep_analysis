"""Main file for preprocessing the MESA sleep dataset."""
import json
from pathlib import Path

import mesa_data_importer as importer
import numpy as np
import pandas as pd
import tqdm

from sleep_analysis.preprocessing.mesa_dataset.actigraphy import process_actigraphy
from sleep_analysis.preprocessing.mesa_dataset.ecg import process_rpoint
from sleep_analysis.preprocessing.mesa_dataset.ground_truth import sleep_stage_convert_binary
from sleep_analysis.preprocessing.mesa_dataset.respiration import check_resp_features
from sleep_analysis.preprocessing.mesa_dataset.utils import (
    align_datastreams,
    check_mesa_data_availability,
    clean_data_to_csv,
    match_exclusion_criteria,
)

with open(Path(__file__).parents[3].joinpath("study_data.json")) as f:
    path_dict = json.load(f)
    mesa_path = Path(path_dict["mesa_path"])
    processed_mesa_path = Path(path_dict["processed_mesa_path"])


def preprocess_mesa():
    """Clean and preprocess data of MESA Sleep dataset.

    Iterate through all mesa_id with full data availability and call clean_data_helper
    """
    overlap = pd.read_csv(mesa_path.joinpath("overlap/mesa-actigraphy-psg-overlap.csv"))

    dataset_info = pd.read_csv(mesa_path.joinpath("datasets/mesa-sleep-dataset-0.6.0.csv")).set_index("mesaid")

    mesa_id_set = check_mesa_data_availability(mesa_path, processed_mesa_path)

    with tqdm.tqdm(total=len(mesa_id_set)) as progress_bar:
        for subj in mesa_id_set:

            if match_exclusion_criteria(dataset_info, subj):
                progress_bar.update(1)
                continue

            df_actigraph = importer.load_single_actigraphy(mesa_path, int(subj))
            df_r_point = importer.load_single_r_point(mesa_path, int(subj))
            df_psg = importer.load_single_psg(mesa_path, int(subj))
            df_resp_features = importer.load_single_resp_features(processed_mesa_path, int(subj))
            edr_features = importer.load_single_edr_feature(processed_mesa_path, int(subj))

            _clean_data_helper(df_actigraph, df_r_point, df_psg, df_resp_features, edr_features, overlap, int(subj))
            progress_bar.update(1)


def _clean_data_helper(df_actigraph, df_r_point, df_sleep_xml, df_resp_features, df_edr_features, overlap, mesa_id):
    """Clean data of MESA Sleep dataset - Preprocess and align datastreams."""
    # process resp_features and edr_features
    df_resp_features = check_resp_features(df_resp_features)
    df_edr_features = check_resp_features(df_edr_features)
    df_edr_features.columns = [col.replace("RRV", "EDR") for col in df_edr_features.columns]

    # preprocess HR data
    df_hr = process_rpoint(df_r_point)

    # convert PSG data from sleep stage to sleep/wake
    df_psg = sleep_stage_convert_binary(df_sleep_xml)

    # process and crop actigraphy to correct length
    df_actigraph = process_actigraphy(df_actigraph, df_psg, overlap, mesa_id)

    # Align Actigraphy, PSG, HR and Respiration
    df_actigraphy, df_psg, df_hr, df_resp_features, df_edr_features = align_datastreams(
        df_actigraph, df_psg, df_hr, df_resp_features, df_edr_features
    )

    # Convert HR to epoch-wise values
    hr_per_epoch = df_hr.groupby("epoch").transform("mean")
    sleep_stages = hr_per_epoch.drop_duplicates()[["stage"]].reset_index(drop=True)

    # 0 = wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM
    sleep_stages["5stage"] = sleep_stages["stage"].map({0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: np.nan, 9: np.nan})
    # 0 = wake, 1 = Light sleep, 2 = Deep sleep, 3 = REM sleep
    sleep_stages["4stage"] = sleep_stages["stage"].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: np.nan, 9: np.nan})
    # 0 = wake, 1 = NREM, 2 = REM
    sleep_stages["3stage"] = sleep_stages["stage"].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: np.nan, 9: np.nan})

    # ensure that no value of sleep stage df is nan
    try:
        assert sleep_stages.isnull().values.any() is False
    except AssertionError:
        print("Nan value in sleep stage df of subject {}".format(mesa_id))

    hr_per_epoch = hr_per_epoch.drop_duplicates()[["HR"]].reset_index(drop=True)

    # Combine actigraphy, PSG, HR, respiration and Ground truth in one DataFrame
    datastreams_combined = pd.concat(
        [df_actigraphy.reset_index(drop=True), df_psg, hr_per_epoch, sleep_stages, df_resp_features, df_edr_features],
        axis=1,
    )

    # Remove remaining NaN values
    # save the epochs from NaN values from the activity + resp data stream to delete them afterward also in the ECG data
    nan_epochs = list(datastreams_combined.loc[pd.isna(datastreams_combined["activity"]), :]["line"])
    nan_epochs += list(datastreams_combined.loc[pd.isna(datastreams_combined["5stage"]), :]["line"])
    datastreams_combined = datastreams_combined.dropna()
    # Remove NaN values from activity + resp data stream in ECG data
    df_hr = df_hr[~df_hr["epoch"].isin(nan_epochs)].reset_index(drop=True)

    df_resp_features = datastreams_combined.filter(regex="RRV").dropna()
    df_edr_features = datastreams_combined.filter(regex="EDR").dropna()

    datastreams_combined = datastreams_combined.drop(datastreams_combined.filter(regex="RRV"), axis=1)
    datastreams_combined = datastreams_combined.drop(datastreams_combined.filter(regex="EDR"), axis=1)

    hr_per_epoch = datastreams_combined["HR"]

    # ensure same length of datastreams
    assert (
        datastreams_combined.shape[0] == df_resp_features.shape[0] == hr_per_epoch.shape[0] == df_edr_features.shape[0]
    )

    # ensure that total sleep time is longer than 2h
    if (
        datastreams_combined.empty is False
        and datastreams_combined["sleep"][datastreams_combined["sleep"] == 1].shape[0]
        > 120  # only save if sleeptime is >2h
    ):

        clean_data_to_csv(datastreams_combined, df_hr, df_resp_features, df_edr_features, mesa_id)

        # print(str(mesa_id) + " cleaning successful")
    else:
        print(str(mesa_id) + " cleaning failed due to short sleep-time that is detected.")
