from pathlib import Path

import pandas as pd

from sleep_analysis.datasets.d04_main_dataset_control import D04MainStudy
from sleep_analysis.datasets.helper import build_base_path_processed


def combine_feature_df(processed_folder: Path, subj_id: int):

    movement_features = pd.read_csv(
        processed_folder.joinpath("movement_features_" + str(subj_id) + ".csv"), index_col=0, parse_dates=True
    )
    hrv_features = pd.read_csv(
        processed_folder.joinpath("hrv_features_" + str(subj_id) + ".csv"), index_col=0, parse_dates=True
    )
    resp_features = pd.read_csv(
        processed_folder.joinpath("resp_features_" + str(subj_id) + ".csv"), index_col=0, parse_dates=True
    )

    # movement_features = pd.read_csv("/Users/danielkrauss/code/Empkins/sleep-analysis/experiments/Feature_extraction/movement_features_04.csv", index_col=0, parse_dates=True)
    # hrv_features = pd.read_csv("/Users/danielkrauss/code/Empkins/sleep-analysis/experiments/Feature_extraction/hrv_features_04.csv", index_col=0, parse_dates=True)
    # resp_features = pd.read_csv("/Users/danielkrauss/code/Empkins/sleep-analysis/experiments/Feature_extraction/resp_features_04.csv", index_col=0, parse_dates=True)

    assert (
        len(movement_features) == len(hrv_features) == len(resp_features)
    ), "Length of feature DataFrames do not match"

    return pd.concat([movement_features, hrv_features, resp_features], axis=1)


def _align_groundtruth(features, ground_truth):

    ground_truth = ground_truth[ground_truth["Sleep Phase"] != "A"]
    ground_truth_clean = ground_truth[ground_truth["Sleep Phase"] != "Artefakt"]

    # First, ensure both DataFrames have the same timezone for proper merging
    ground_truth_clean.index = ground_truth_clean.index.tz_convert("Europe/Berlin")
    features.index = features.index.tz_convert("Europe/Berlin")

    # Merge the two DataFrames on the index (timestamps) to retain only overlapping epochs
    merged_df = ground_truth_clean.merge(features, left_index=True, right_index=True, how="inner")

    # Create feature DataFrame and ground truth DataFrame
    df_ground_truth = merged_df[["Sleep Phase"]]
    df_features = merged_df.drop(columns=["Sleep Phase"])

    # Now, df_ground_truth contains the sleep phases and df_features_only contains only the features.

    assert len(df_features) == len(
        df_ground_truth
    ), "Length of feature DataFrame and ground truth DataFrame do not match"

    return df_features, df_ground_truth


def _map_sleep_phases_to_num(df):
    # Mapping dictionary
    stage_map_5 = {"A": 0, "Wach": 0, "N1": 1, "N2": 2, "N3": 3, "Rem": 4}  # remove!!

    stage_map_3 = {"A": 0, "Wach": 0, "N1": 1, "N2": 1, "N3": 1, "Rem": 2}  # remove!!

    stage_map_2 = {"A": 0, "Wach": 0, "N1": 1, "N2": 1, "N3": 1, "Rem": 1}  # remove!!

    # Replace the values using the map
    df["5stage"] = df["Sleep Phase"].map(stage_map_5)
    df["3stage"] = df["Sleep Phase"].map(stage_map_3)
    df["2stage"] = df["Sleep Phase"].map(stage_map_2)

    return df


if __name__ == "__main__":
    dataset = D04MainStudy()

    for subj in dataset:
        subj_id = subj.index["subj_id"][0]
        processed_folder = Path(build_base_path_processed().joinpath("Vp_" + subj_id))

        # if file in folder does not exist, skip
        # if not processed_folder.joinpath("movement_features_" + str(subj_id) + ".csv").exists():
        #    print("No Movement features extracted for subject", subj_id, flush=True)
        #    continue

        # if file in folder does not exist, skip
        # if not processed_folder.joinpath("hrv_features_" + str(subj_id) + ".csv").exists():
        #    print("No HRV features extracted for subject", subj_id, flush=True)
        #    continue

        # if file in folder does not exist, skip
        # if not processed_folder.joinpath("resp_features_" + str(subj_id) + ".csv").exists():
        #    print("No Resp features already extracted for subject", subj_id, flush=True)
        #    continue

        try:
            features = combine_feature_df(processed_folder, subj_id)
            ground_truth = subj.psg_labels

        except FileNotFoundError:
            print("At least one set of features or the labels were not found for subject", subj_id, flush=True)
            continue

        df_features, ground_truth = _align_groundtruth(features, ground_truth)

        ground_truth = _map_sleep_phases_to_num(ground_truth)

        df_features.to_csv(processed_folder.joinpath("features_" + str(subj_id) + ".csv"))
        ground_truth.to_csv(processed_folder.joinpath("ground_truth_" + str(subj_id) + ".csv"))
