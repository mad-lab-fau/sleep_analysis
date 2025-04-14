from pathlib import Path

import pandas as pd

from sleep_analysis.datasets.d04_main_dataset_control import D04MainStudy
from sleep_analysis.datasets.helper import build_base_path_processed
from sleep_analysis.feature_extraction.mesa_datasst.actigraphy import calc_actigraph_features


def get_movement_features(df: pd.DataFrame):

    return calc_actigraph_features(df["movement"])


if __name__ == "__main__":
    dataset = D04MainStudy()

    for subj in dataset:
        subj_id = subj.index["subj_id"][0]
        processed_folder = Path(build_base_path_processed().joinpath("Vp_" + subj_id))

        if not processed_folder.joinpath("movement" + str(subj_id) + ".csv").exists():
            print("Movement data not found for subject", subj_id, flush=True)
            continue

        movement_df = pd.read_csv(
            processed_folder.joinpath("movement" + str(subj_id) + ".csv"), index_col=0, parse_dates=True
        )

        movement_features = get_movement_features(movement_df)

        movement_features.set_index(movement_df.index, drop=True, inplace=True)

        movement_features.to_csv(processed_folder.joinpath("movement_features_" + str(subj_id) + ".csv"))
