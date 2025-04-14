from sleep_analysis.datasets.d04_main_dataset_control import D04MainStudy
from sleep_analysis.datasets.d04_main_dataset_pd import D04PDStudy
from sleep_analysis.datasets.helper import build_base_path_processed
from sleep_analysis.preprocessing.d04_main.heart_sound_radar import preprocess_heart_sound
from sleep_analysis.preprocessing.d04_main.utils import load_and_sync_radar_data
from pathlib import Path

# General parameters
fs_radar = 1953.125


def extract_heart_sound_radar_per_subj(subj, fs_radar=1953.125):

    synced_radar = load_and_sync_radar_data(subj)
    if synced_radar is None:
        return None, None

    # Extract unique radar names dynamically
    radar_names = synced_radar.columns.get_level_values(0).unique()

    # Create dictionary of sub-dataframes
    radar_dict = {radar: synced_radar[radar] for radar in radar_names}

    r_peaks_df, lstm_probability = preprocess_heart_sound(radar_dict, window_size=300, fs_radar=fs_radar)

    return r_peaks_df, lstm_probability


# main method:
if __name__ == "__main__":
    dataset_name = "d04_control"

    if dataset_name == "d04_control":
        dataset = D04MainStudy()
    elif dataset_name == "d04_pd":
        dataset = D04PDStudy()
    else:
        raise ValueError("Unknown dataset_name")

    for subj in dataset:
        subj_id = subj.index["subj_id"][0]
        print("Processing subject", subj_id, flush=True)

        if str(subj_id) == "09":
            print("Skip subject 09", flush=True)
            continue

        processed_folder = Path(build_base_path_processed(dataset_name).joinpath("Vp_" + subj_id))

        # check if folder exists
        if not processed_folder.exists():
            print("create folder ...", flush=True)
            processed_folder.mkdir(parents=True)

        # if file in folder exists, skip
        if processed_folder.joinpath("r_peaks" + str(subj_id) + ".csv").exists():
            continue

        r_peak_df, lstm_probability = extract_heart_sound_radar_per_subj(subj, fs_radar=fs_radar)

        if r_peak_df is None:
            print("No processable data for subject", subj_id, flush=True)
            continue

        r_peak_df.to_csv(processed_folder.joinpath("r_peaks" + str(subj_id) + ".csv"))
        lstm_probability.to_csv(processed_folder.joinpath("lstm_probability" + str(subj_id) + ".csv"))
