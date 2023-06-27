import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

with open(Path(__file__).parents[2].joinpath("study_data.json")) as f:
    path_dict = json.load(f)
    mesa_path = Path(path_dict["mesa_path"])
    processed_mesa_path = Path(path_dict["processed_mesa_path"])


def extract_actigraph_features(overwrite=True):
    """
    Extract actigraph features from MESA dataset
    Calculates 370 time-series features with different sized rolling windows. These features are centered or non-centered.
    If the window size is fixed at 20, this fuction returns a total of 370 features.

    :param overwrite: If overwrite = True, the features are calculated and overwritten. If set to false, all features that are calculated are skipped.
    """

    path_list = list(processed_mesa_path.joinpath("actigraph_data_clean").glob("*.csv"))
    mesa_id = re.findall("(\d{4})", str(path_list))
    for subj in mesa_id:
        if not overwrite:  # check if file already exists
            exists = os.path.isfile(
                processed_mesa_path.joinpath("actigraph_features/actigraph_features" + subj + ".csv")
            )
            if exists:
                print("skip mesa_id because it already exists: " + subj)
                continue

        series_actigraph = pd.read_csv(
            processed_mesa_path.joinpath("actigraph_data_clean/actigraph_data_clean" + subj + ".csv")
        )["activity"]

        actigraphy_features = calc_actigraph_features(series_actigraph)
        actigraphy_features.to_csv(
            processed_mesa_path.joinpath("actigraph_features/actigraph_features" + subj + ".csv"), index=False
        )

        print("Feature extraction of actigraphy of subj: " + subj + " finished!")


def calc_actigraph_features(series_actigraph: pd.Series, windows_size: int = 20):
    """
    Compute time-series features of Actigraph data as input with different sized rolling windows. These features are centered or non-centered. If the window size is fixed at 20, this fuction returns a total of 370 features.
    :param series_actigraph: pd.Series that contains actigraph data
    :param windows_size: maximum size of rolling windows. All window-sizes from 1 to window_size will be computed: default = 20
    :returns: pd.DataFrame containing the extracted features
    """
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    df_features = pd.DataFrame()
    for win_size in np.arange(1, windows_size):
        df_features["_acc_mean_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).mean().fillna(0.0)
        )
        df_features["_acc_mean_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=True, min_periods=1).mean().fillna(0.0)
        )
        df_features["_acc_median_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).median().fillna(0.0)
        )
        df_features["_acc_median_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=True, min_periods=1).median().fillna(0.0)
        )

        df_features["_acc_std_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).std().fillna(0.0)
        )
        df_features["_acc_std_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=True, min_periods=1).std().fillna(0.0)
        )

        df_features["_acc_max_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).max().fillna(0.0)
        )
        df_features["_acc_max_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=True, min_periods=1).max().fillna(0.0)
        )

        df_features["_acc_min_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).min().fillna(0.0)
        )
        df_features["_acc_min_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=True, min_periods=1).min().fillna(0.0)
        )

        df_features["_acc_var_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=False, min_periods=1).var().fillna(0.0)
        )
        df_features["_acc_var_centered_%d" % win_size] = (
            series_actigraph.rolling(window=win_size, center=True, min_periods=1).var().fillna(0.0)
        )

        df_features["_acc_nat_%d" % win_size] = (
            ((series_actigraph >= 50) & (series_actigraph < 100))
            .rolling(window=win_size, center=False, min_periods=1)
            .sum()
            .fillna(0.0)
        )
        df_features["_acc_nat_centered_%d" % win_size] = (
            ((series_actigraph >= 50) & (series_actigraph < 100))
            .rolling(window=win_size, center=True, min_periods=1)
            .sum()
            .fillna(0.0)
        )

        df_features["_acc_anyact_%d" % win_size] = (
            (series_actigraph > 0).rolling(window=win_size, center=False, min_periods=1).sum().fillna(0.0)
        )
        df_features["_acc_anyact_centered_%d" % win_size] = (
            (series_actigraph > 0).rolling(window=win_size, center=True, min_periods=1).sum().fillna(0.0)
        )

        if win_size > 3:
            df_features["_acc_skew_%d" % win_size] = (
                series_actigraph.rolling(window=win_size, center=False, min_periods=1).skew().fillna(0.0)
            )
            df_features["_acc_skew_centered_%d" % win_size] = (
                series_actigraph.rolling(window=win_size, center=True, min_periods=1).skew().fillna(0.0)
            )
            #
            df_features["_acc_kurt_%d" % win_size] = (
                series_actigraph.rolling(window=win_size, center=False, min_periods=1).kurt().fillna(0.0)
            )
            df_features["_acc_kurt_centered_%d" % win_size] = (
                series_actigraph.rolling(window=win_size, center=True, min_periods=1).kurt().fillna(0.0)
            )
    df_features["_acc_Act"] = series_actigraph.fillna(0.0)
    df_features["_acc_LocAct"] = (series_actigraph + 1).apply(np.log).fillna(0.0)

    return df_features.reset_index(drop=True)
