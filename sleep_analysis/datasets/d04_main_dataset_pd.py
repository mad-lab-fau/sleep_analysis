import re
from functools import cached_property, lru_cache
from pathlib import Path

import pandas as pd
from empkins_io.sync import SyncedDataset
from tpcp import Dataset

from sleep_analysis.datasets.helper import _load_radar_data, _build_base_path, _load_exclusion_list
from typing import Optional, Sequence
import empkins_io.sensors.psg.psg_channels as psg_ch
from biopsykit.io.psg import PSGDataset

_cached_load_radar_data = lru_cache(maxsize=4)(_load_radar_data)


class D04PDStudy(Dataset):
    use_cache: bool
    exclusion_criteria: list

    def __init__(
        self,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        use_cache: Optional[bool] = True,
        exclusion_criteria: Optional[list] = None,
    ):
        self.exclusion_criteria = exclusion_criteria
        self.use_cache = use_cache
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)


    def create_index(self):
        path = _build_base_path(dataset_name="empkins_mainstudy_pd")

        path_list = list(Path(path).glob("Vp_*"))
        subjects = sorted(re.findall("(\d{2})", str(path_list)))

        exclusion_list = _load_exclusion_list(path)
        subjects = [subj for subj in subjects if subj not in exclusion_list]

        return pd.DataFrame(subjects, columns=["subj_id"])


    @property
    def full_psg_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="pd_sleep_lab", psg_channel_group="FullPSG")
        path = _build_base_path(dataset_name="empkins_mainstudy_pd")
        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def ecg_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="pd_sleep_lab", psg_channel_group="ECG")
        path = _build_base_path(dataset_name="empkins_mainstudy_pd")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def eeg_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="pd_sleep_lab", psg_channel_group="EEG")
        path = _build_base_path(dataset_name="empkins_mainstudy_pd")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def emg_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="pd_sleep_lab", psg_channel_group="EMG")
        path = _build_base_path(dataset_name="empkins_mainstudy_pd")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def eog_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="pd_sleep_lab", psg_channel_group="EOG")
        path = _build_base_path(dataset_name="empkins_mainstudy_pd")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def position_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="pd_sleep_lab", psg_channel_group="Position")
        path = _build_base_path(dataset_name="empkins_mainstudy_pd")

        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @property
    def respiration_data(self):
        channels = psg_ch.get_psg_channels_by_group(system="pd_sleep_lab", psg_channel_group="Resp")
        path = _build_base_path(dataset_name="empkins_mainstudy_pd")


        if self.is_single(["subj_id"]):
            edf_path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/psg")
            data = PSGDataset.from_edf_file(edf_path, datastreams=channels)

            return data

        raise ValueError(
            "Data can only be accessed when there is only a single recording of a single participant in the subset"
        )

    @cached_property
    def radar_data(self):
        path = _build_base_path(dataset_name="empkins_mainstudy_pd")
        path = path.joinpath("Vp_" + self.index["subj_id"][0] + "/radar")
        if self.use_cache:
            return _cached_load_radar_data(path)
        else:
            if self.is_single(["subj_id"]):
                return _load_radar_data(path)

            raise ValueError(
                "Data can only be accessed when there is only a single recording of a single participant in the subset"
            )


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
                radar_data = radar_data.data_as_df(index = 'local_datetime', add_sync_out=True)
                if not isinstance(radar_data, pd.DataFrame):
                    raise ValueError("Radar data must be a DataFrame")
            # check if at least one of the radar signals is in the DataFrame
            if not any([col in radar_data.columns for col in ["rad1", "rad2", "rad3", "rad4"]]):
                raise ValueError("Radar data must contain at least one of the columns 'rad1', 'rad2', 'rad3', 'rad4'")

            radar_data = radar_data.dropna()

            # Extract unique radar names dynamically
            radar_names = radar_data.columns.get_level_values(0).unique()

            print("Found radar names:", radar_names)

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
                primary=radar_names[0],
                reset_time_axis=True,
                cut_to_shortest=True,
                sync_params={"sync_region_samples": (0, 100000)},
            )

            # Find shift at the end of the m-sequence
            print("Find shift at the end of the m-sequence")
            dict_shift = synced_dataset._find_shift(
                primary=radar_names[0] + "_aligned_", sync_params={"sync_region_samples": (-100000, -1)}
            )

            # Resample sample-wise to get equal length
            print("Resample sample-wise to get equal length")
            synced_dataset.resample_sample_wise(primary=radar_names[0] + "_aligned_", dict_sample_shift=dict_shift)

            return pd.concat(synced_dataset.datasets_resampled, axis=1)
