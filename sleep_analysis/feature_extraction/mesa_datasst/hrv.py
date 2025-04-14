import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from hrvanalysis import (
    get_csi_cvi_features,
    get_frequency_domain_features,
    get_geometrical_features,
    get_poincare_plot_features,
    get_time_domain_features,
)

from sleep_analysis.feature_extraction.mesa_datasst.utils import check_processed

with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
    path_dict = json.load(f)
    mesa_path = Path(path_dict["mesa_path"])
    processed_mesa_path = Path(path_dict["processed_mesa_path"])


def extract_hrv_features(overwrite=True):
    """
    Extract HRV features from MESA dataset
    Calculates 30 different HRV features from time-domain, frequency-domain and non-linear domain.

    :param overwrite: If overwrite = True, the features are calculated and overwritten. If set to false, all features that are calculated are skipped.
    """

    path_list = list(processed_mesa_path.joinpath("actigraph_data_clean").glob("*.csv"))
    mesa_id = re.findall("(\d{4})", str(path_list))
    with tqdm.tqdm(total=len(mesa_id)) as progress_bar:
        for subj in mesa_id:
            if not overwrite:  # check if file already exists
                if check_processed(processed_mesa_path.joinpath("hrv_features"), subj):
                    progress_bar.update(1)
                    continue

            df_hr = pd.read_csv(processed_mesa_path.joinpath("ecg_data_clean/ecg_data_clean" + subj + ".csv"))

            hr_features = calc_hrv_features(df_hr)

            del hr_features["_hrv_epoch"]

            hr_features.to_csv(processed_mesa_path.joinpath("hrv_features/hrv_features" + subj + ".csv"), index=False)

            print("Features extraction of HRV of subj: " + subj + " finished!")
        progress_bar.update(1)  # update progress


def calc_hrv_features(df_hr: pd.DataFrame):
    """
    Compute HRV features from time-domain, frequency-domain and non-linear domain. This is done via the python package hrvanalysis. This function returns a total of 30 HRV features.
    :param df_hr: pd.DataFrame that contains RR-intervals
    :returns: pd.DataFrame that contains all HRV features
    """
    hr_epoch_set = set(df_hr["epoch"].values)

    all_hr_features = {}
    for i, hr_epoch_idx in enumerate(list(hr_epoch_set)):
        tmp_hr_df = df_hr[df_hr["epoch"] == hr_epoch_idx]
        if tmp_hr_df.size > 3:
            rr_epoch = tmp_hr_df["RR Intervals"].values

            all_hr_features[hr_epoch_idx] = {}
            all_hr_features[hr_epoch_idx].update(get_time_domain_features(rr_epoch))
            all_hr_features[hr_epoch_idx].update(get_frequency_domain_features(rr_epoch))
            all_hr_features[hr_epoch_idx].update(get_poincare_plot_features(rr_epoch))
            all_hr_features[hr_epoch_idx].update(get_csi_cvi_features(rr_epoch))
            all_hr_features[hr_epoch_idx].update(get_geometrical_features(rr_epoch))
            all_hr_features[hr_epoch_idx].update({"epoch": hr_epoch_idx})
    all_hr_features = pd.DataFrame(all_hr_features).T
    del all_hr_features["tinn"]

    # eventually needed for covering the issue of some samples with nan inf or -inf
    with pd.option_context("mode.use_inf_as_na", True):
        all_hr_features.fillna(0.0)
        all_hr_features.replace([np.inf, -np.inf], 0.0, inplace=True)

    all_hr_features.columns = ["_hrv_" + str(col) for col in all_hr_features.columns]

    return all_hr_features.reset_index(drop=True)
