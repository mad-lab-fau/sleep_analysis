import json
import re
from pathlib import Path

import pandas as pd
import tqdm

with open(Path(__file__).parents[3].joinpath("study_data.json")) as f:
    path_dict = json.load(f)
    mesa_path = Path(path_dict["mesa_path"])
    processed_mesa_path = Path(path_dict["processed_mesa_path"])


def check_mesa_data_availability(mesa_path, processed_mesa_path):
    """
    Check data completeness - This function iterates through all actigraphy, HR and .edf files as well as the "overlap file" which indicates the overlap of the single signals
    :param mesa_path: path to mesa dataset - saved in json file in root folder
    :param processed_mesa_path: path to processed mesa file - saved in json file in root folder
    """

    overlap = pd.read_csv(mesa_path.joinpath("overlap/mesa-actigraphy-psg-overlap.csv"))

    actigraphy_path = mesa_path.joinpath("actigraphy")
    psg_path = mesa_path.joinpath("polysomnography/annotations-events-nsrr")
    r_point_path = mesa_path.joinpath("polysomnography/annotations-rpoints")
    path_resp = Path(processed_mesa_path.joinpath("respiration_features_raw"))
    edr_path = Path(processed_mesa_path.joinpath("edr_respiration_features_raw"))

    path_list_actigraphy = list(Path(actigraphy_path).glob("*.csv"))
    path_list_psg = list(Path(psg_path).glob("*.xml"))
    path_list_r_point_path = list(Path(r_point_path).glob("*.csv"))
    path_list_resp = list(Path(path_resp).glob("*.csv"))
    path_list_edr = list(Path(edr_path).glob("*.csv"))

    mesa_id_actigraphy = set(re.findall("(\d{4})", str(path_list_actigraphy)))
    mesa_id_psg = set(re.findall("(\d{4})", str(path_list_psg)))
    mesa_id_r_point = set(re.findall("(\d{4})", str(path_list_r_point_path)))
    mesa_id_resp = set(re.findall("(\d{4})", str(path_list_resp)))
    mesa_id_edr = set(re.findall("(\d{4})", str(path_list_edr)))
    mesa_id_overlap = set(overlap["mesaid"].apply(str).apply(lambda x: x.zfill(4)).tolist())

    # set.intersection(set1, set2 ... etc) # Method to find intersection between two or more sets
    return mesa_id_actigraphy.intersection(mesa_id_psg, mesa_id_r_point, mesa_id_resp, mesa_id_overlap, mesa_id_edr)


def match_exclusion_criteria(info, subj):
    """
    Exclude every subejct with bad quality gradings.
    Scale:
    2 - Poor
    3 - Fair
    4 - Good
    5 - Very Good
    6 - Excellent
    7 - Outstanding
    Criteria: PSG needs to have at least score 4, Actigraphy at least Score 3
    """
    subj_info = info.loc[[int(subj)]]

    # check overall quality
    if subj_info["overall5"].iloc[0] <= 3:
        print("Subj - " + subj + " - PSG quality insufficient <= 3")
        return True

    # check if epochs are not scored realistic
    if subj_info["slewake5"].iloc[0] == True:
        print("Subj - " + subj + " - PSG scoring quality insufficient")
        return True

    # check actigraphy quality
    if subj_info["overallqual5"].iloc[0] <= 2:
        print("Subj - " + subj + " - Actigraphy quality insufficient")
        return True

    return False


def align_datastreams(df_actigraphy, df_psg, df_hr, df_resp_features, df_edr_features):

    epoch_hr_set = set(df_hr["epoch"].values)
    epoch_actigraphy_set = set(df_actigraphy["line"])
    epoch_set_resp = set(df_resp_features["epoch"])
    epoch_set_edr = set(df_edr_features["epoch"])
    epoch_psg_set = set(df_psg.index + 1)

    # only keep intersect dataset
    intersect_epochs = epoch_psg_set.intersection(epoch_actigraphy_set, epoch_hr_set, epoch_set_resp, epoch_set_edr)
    df_actigraphy = df_actigraphy[df_actigraphy["line"].isin(intersect_epochs)].reset_index(drop=True)
    df_hr = df_hr[df_hr["epoch"].isin(intersect_epochs)].reset_index(drop=True)
    df_resp_features = df_resp_features[df_resp_features["epoch"].isin(intersect_epochs)].reset_index(drop=True)
    df_edr_features = df_edr_features[df_edr_features["epoch"].isin(intersect_epochs)].reset_index(drop=True)
    df_psg = df_psg[(df_psg.index + 1).isin(intersect_epochs)].reset_index(drop=True)

    # check see if their epochs are equal
    assert (
        df_actigraphy.shape[0]
        == len(df_hr["epoch"].unique())
        == df_resp_features.shape[0]
        == df_psg.shape[0]
        == df_edr_features.shape[0]
    )

    return df_actigraphy, df_psg, df_hr, df_resp_features, df_edr_features


def clean_data_to_csv(datastreams_combined, df_hr, df_resp_features, df_edr_features, mesa_id):
    # save datastreams to csv
    datastreams_combined.to_csv(
        processed_mesa_path.joinpath("actigraph_data_clean/actigraph_data_clean" + "{:04d}".format(mesa_id) + ".csv"),
        index=False,
    )
    df_hr.to_csv(
        processed_mesa_path.joinpath("ecg_data_clean/ecg_data_clean" + "{:04d}".format(mesa_id) + ".csv"), index=False
    )
    df_resp_features.to_csv(
        processed_mesa_path.joinpath(
            "respiration_features_clean/respiration_features" + "{:04d}".format(mesa_id) + ".csv"
        ),
        index=False,
    )

    df_edr_features.to_csv(
        processed_mesa_path.joinpath("edr_features_clean/edr_features" + "{:04d}".format(mesa_id) + ".csv"),
        index=False,
    )


def check_dataset_validity(dataset):
    """
    Check if the number of epochs for feature set and ground truth is the same
    Error is printed if this is not the case
    """
    with tqdm.tqdm(total=len(dataset)) as progress_bar:
        i = 0
        for subj in dataset:
            try:
                assert subj.ground_truth.dropna().shape[0] == subj.features.dropna().shape[0]
                assert subj.features.shape[1] == 460
            except AssertionError:
                print("Shape of feature/ground-truth of subject " + subj.index["mesa_id"][0] + " invalid!")
                print(str(subj.index["mesa_id"][0] + " - columns: ") + str(subj.features.shape[1]))
                i += 1
                progress_bar.update(1)
                continue
            progress_bar.update(1)
