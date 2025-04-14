import json
import re
from datetime import datetime, timedelta
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Optional, Sequence

import empkins_io.sensors.psg.psg_channels as psg_ch
import pandas as pd
import pytz
from biopsykit.io.psg import PSGDataset
from empkins_io.sync import SyncedDataset
from tpcp import Dataset
import platform

from sleep_analysis.datasets.helper import (
    _build_base_path,
    _load_radar_data,
    _exclude_data,
    _load_exclusion_list,
    build_base_path_processed_local,
    build_base_path_processed,
)

_cached_load_radar_data = lru_cache(maxsize=4)(_load_radar_data)


class D04MainStudy(Dataset):
    use_cache: bool
    exclusion_criteria: list
    classification: str
    retrain: bool

    def __init__(
            self,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            use_cache: Optional[bool] = True,
            exclusion_criteria: Optional[list] = None,
            classification: Optional[str] = "5stage",
            retrain: Optional[bool] = False,
    ):
        self.exclusion_criteria = exclusion_criteria
        self.use_cache = use_cache
        self.classification = classification
        self.retrain = retrain
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        # load path dependent on OS:
        # TODO

        path = _build_base_path(dataset_name="empkins_mainstudy_control")

        path_list = list(Path(path).glob("Vp_*"))
        subjects = sorted(re.findall("(\d{2})", str(path_list)))

        exclusion_list = _load_exclusion_list(path)
        subjects = [subj for subj in subjects if subj not in exclusion_list]

        return pd.DataFrame(subjects, columns=["subj_id"])

    @property
    def full_psg_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="somno", psg_channel_group="FullPSG")
        path = _build_base_path(dataset_name="empkins_mainstudy_control")
        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def ecg_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="somno", psg_channel_group="ECG")
        path = _build_base_path(dataset_name="empkins_mainstudy_control")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def eeg_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="somno", psg_channel_group="EEG")
        path = _build_base_path(dataset_name="empkins_mainstudy_control")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def emg_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="somno", psg_channel_group="EMG")
        path = _build_base_path(dataset_name="empkins_mainstudy_control")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def eog_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="somno", psg_channel_group="EOG")
        path = _build_base_path(dataset_name="empkins_mainstudy_control")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def sync_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="somno", psg_channel_group="Sync")
        path = _build_base_path(dataset_name="empkins_mainstudy_control")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def activity_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="somno", psg_channel_group="Activity")
        path = _build_base_path(dataset_name="empkins_mainstudy_control")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def position_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="somno", psg_channel_group="Position")
        path = _build_base_path(dataset_name="empkins_mainstudy_control")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def respiration_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="somno", psg_channel_group="Resp")
        with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
            path_dict = json.load(f)
            path = Path(path_dict["empkins_mainstudy_control"])

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @cached_property
    def radar_data(self):
        path = _build_base_path(dataset_name="empkins_mainstudy_control")
        path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/radar")

        if self.use_cache:
            return _cached_load_radar_data(path)
        else:
            if self.is_single(["subj_id"]):
                return _load_radar_data(path)

            raise ValueError(
                "Data can only be accessed when there is only a single recording of a single participant in the subset"
            )

    @property
    def psg_labels(self):
        """
        Returns the labels of the PSG data as a pandas DataFrame
        This is only possible if there is only a single recording of a single participant in the subset
        The time index is always localized to Europe/Berlin - This might need to be changed in the future!
        """
        if self.is_single(["subj_id"]):
            path = _build_base_path(dataset_name="empkins_mainstudy_control")

            file_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/labels/Schlafprofil.txt")
            data = []

            with open(file_path, "r") as file:
                _ = file.readline()
                second_line = file.readline()

                start_date = datetime.strptime(second_line.split(" ")[2].strip(), "%d.%m.%Y")

                for _ in range(5):
                    next(file)

                previous_time = None
                current_date = start_date

                for line in file:
                    line = line.strip()
                    if line:
                        parts = line.split(";")

                        # Extract the time part
                        current_time = datetime.strptime(parts[0].strip(), "%H:%M:%S,%f").time()

                        # If this is the first timestamp or if the current time is earlier than the previous one, increment the date
                        if previous_time and current_time < previous_time:
                            current_date += timedelta(days=1)

                        # Combine current date and time to form the full timestamp
                        timestamp = datetime.combine(current_date, current_time)
                        timestamp = pytz.timezone("Europe/Berlin").localize(timestamp)

                        sleep_phase = parts[1].strip()
                        data.append([timestamp, sleep_phase])

                        previous_time = current_time  # Update previous_time for the next iteration

            df = pd.DataFrame(data, columns=["Timestamp", "Sleep Phase"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df.set_index("Timestamp", inplace=True)

            df = _exclude_data(df, self.exclusion_criteria, vp_id=self.index["subj_id"][0])

            return df

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def feature_table(self):
        """
        Returns the feature table of Movement, HRV and RRV data as a pandas DataFrame
        This is only possible if there is only a single recording of a single participant in the subset
        """
        if self.is_single(["subj_id"]):

            # Check the operating system
            if platform.system() == "Linux":
                path = build_base_path_processed()
            elif platform.system() == "Darwin":  # Darwin is the system name for macOS
                path = build_base_path_processed_local()
            else:
                raise EnvironmentError("Unsupported operating system")
            file_path = path.joinpath(
                "Vp_" + self.index["subj_id"][0] + "/features_" + self.index["subj_id"][0] + ".csv"
            )
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def ground_truth(self):
        """
        Returns the ground truth data as a pandas DataFrame
        """

        # Check the operating system
        if platform.system() == "Linux":
            path = build_base_path_processed()
        elif platform.system() == "Darwin":  # Darwin is the system name for macOS
            path = build_base_path_processed_local()
        else:
            raise EnvironmentError("Unsupported operating system")

        file_path = path.joinpath(
            "Vp_" + self.index["subj_id"][0] + "/ground_truth_" + self.index["subj_id"][0] + ".csv"
        )
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        return data

    @property
    def classification_labels(self):

        with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
            path_dict = json.load(f)
            path = Path(path_dict["results_per_subj"])

        if self.classification == "3stage":
            if self.retrain:
                path = path.joinpath("3stage", "retrain")
            else:
                path = path.joinpath("3stage", "radar_only")
        if self.classification == "5stage":
            if self.retrain:
                path = path.joinpath("5stage", "retrain")
            else:
                path = path.joinpath("5stage", "radar_only")
        classification_labels = pd.read_csv(path.joinpath(self.index["subj_id"][0] + ".csv"), index_col=0)
        classification_labels.index = self.ground_truth.index

        return classification_labels

    def sync_radar(self, radar_data: pd.DataFrame):
        """
        Synchronize the radar data from four radar streams with each other.
        """

        # radar data sampling rate
        fs_radar = 1953.125

        if self.is_single(["subj_id"]):

            # check if radar data is available
            if radar_data is None:
                raise ValueError("No radar data available for this participant")

            # check if radar data is a DataFrame containing radar signals named "rad_number"
            if not isinstance(radar_data, pd.DataFrame):
                raise ValueError("Radar data must be a DataFrame")
            # check if at least one of the radar signals is in the DataFrame
            if not any([col in radar_data.columns for col in ["rad1", "rad2", "rad3", "rad4"]]):
                raise ValueError("Radar data must contain at least one of the columns 'rad1', 'rad2', 'rad3', 'rad4'")

            radar_data = radar_data.dropna()

            # Extract unique radar names dynamically
            radar_names = radar_data.columns.get_level_values(0).unique()

            # Create dictionary of sub-dataframes
            radar_dict = {radar: radar_data[radar] for radar in radar_names}

            print("Prepare SyncedDataset")

            synced_dataset = SyncedDataset(sync_type="m-sequence")

            # add all radar DataFrames to SyncedDataset

            for radar_name, radar_df in radar_dict.items():
                synced_dataset.add_dataset(
                    radar_name, data=radar_df, sync_channel_name="Sync_Out", sampling_rate=fs_radar
                )

            # Sync beginning of m-sequence
            print("Sync beginning of m-sequence")
            synced_dataset.align_and_cut_m_sequence(
                primary="rad1",
                reset_time_axis=True,
                cut_to_shortest=True,
                sync_params={"sync_region_samples": (0, 100000)},
            )

            # Find shift at the end of the m-sequence
            print("Find shift at the end of the m-sequence")
            dict_shift = synced_dataset._find_shift(
                primary="rad1_aligned_", sync_params={"sync_region_samples": (-100000, -1)}
            )

            # Resample sample-wise to get equal length
            print("Resample sample-wise to get equal length")
            synced_dataset.resample_sample_wise(primary="rad1_aligned_", dict_sample_shift=dict_shift)

            return pd.concat(synced_dataset.datasets_resampled, axis=1)
