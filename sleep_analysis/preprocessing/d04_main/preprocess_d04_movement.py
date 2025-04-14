from sleep_analysis.datasets.d04_main_dataset_control import D04MainStudy
from sleep_analysis.datasets.d04_main_dataset_pd import D04PDStudy
from sleep_analysis.datasets.helper import build_base_path_processed
from sleep_analysis.preprocessing.d04_main.movement_radar import preprocess_movement
from sleep_analysis.preprocessing.d04_main.utils import load_and_sync_radar_data
from pathlib import Path


def extract_movement_radar_per_subj(subj):

    synced_radar = load_and_sync_radar_data(subj)

    if synced_radar is None:
        return None

    # Extract unique radar names dynamically
    radar_names = synced_radar.columns.get_level_values(0).unique()

    # Create dictionary of sub-dataframes
    radar_dict = {radar: synced_radar[radar] for radar in radar_names}

    movement = preprocess_movement(radar_dict)

    return movement


if __name__ == "__main__":
    # "d04_control" or "d04_pd_study"
    dataset_name = "d04_pd_study"

    if dataset_name == "d04_control":
        dataset = D04MainStudy()
    elif dataset_name == "d04_pd":
        dataset = D04PDStudy()
    else:
        raise ValueError("Unknown dataset_name")

    for subj in dataset:
        subj_id = subj.index["subj_id"][0]
        processed_folder = Path(build_base_path_processed(dataset_name).joinpath("Vp_" + subj_id))

        # check if folder exists
        if not processed_folder.exists():
            print("create folder ...", flush=True)
            processed_folder.mkdir(parents=True)

        # if file in folder exists, skip
        if processed_folder.joinpath("movement" + str(subj_id) + ".csv").exists():
            continue

        movement_df = extract_movement_radar_per_subj(subj)

        if movement_df is None:
            continue

        print(movement_df.head())
        movement_df.to_csv(processed_folder.joinpath("movement" + str(subj_id) + ".csv"))
