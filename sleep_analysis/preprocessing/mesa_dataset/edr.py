import json
import random
import re
from pathlib import Path

import mesa_data_importer as mesa
import numpy as np
import pandas as pd
import tqdm
from sleep_analysis.preprocessing.mesa_dataset.edr.extraction_feature import (
    ExtractionCharlton,
)
from biopsykit.signals.ecg import EcgProcessor

from sleep_analysis.feature_extraction.rrv import extract_rrv_features_helper, process_resp
from sleep_analysis.feature_extraction.utils import check_processed
from sleep_analysis.preprocessing.utils import extract_edf_channel

with open(Path(__file__).parents[3].joinpath("study_data.json")) as f:
    path_dict = json.load(f)
    edf_path = Path(path_dict["mesa_path"]).joinpath("polysomnography/edfs")
    processed_mesa_path = Path(path_dict["processed_mesa_path"])


def extract_edr_features(overwrite=False):
    """
    Extract EDR signal from ECG signal
    overwrite: if True, overwrite existing files

    """
    path_list = list(Path(edf_path).glob("*.edf"))
    mesa_id = re.findall("(\d{4})", str(path_list))

    with tqdm.tqdm(total=len(mesa_id)) as progress_bar:
        for subj in mesa_id:

            if not overwrite:  # check if file already exists
                if check_processed(
                    Path(path_dict["processed_mesa_path"]).joinpath("edr_respiration_features_raw"), subj
                ):
                    progress_bar.update(1)  # update progress
                    continue

            raw_ecg, epochs = extract_edf_channel(edf_path, subj_id=int(subj), channel="EKG")

            edr_signal = _extract_edr(raw_ecg, sampling_rate=256)

            resp_df, epochs = process_resp(edr_signal.respiratory_signal, epochs)
            features = extract_rrv_features_helper(resp_df, nan_pad=0.0, sampling_rate=32)

            features.to_csv(
                Path(path_dict["processed_mesa_path"]).joinpath(
                    "edr_respiration_features_raw/edr_respiration" + str(subj) + ".csv"
                )
            )

            progress_bar.update(1)  # update progress


def _preprocess_ecg(raw_ecg: np.array):
    """
    Preprocess ECG signal using default method from biopsykit.
    raw_ecg: raw ECG signal
    return: preprocessed ECG signal
    """
    raw_ecg.rename({0: "ecg"}, axis=1, inplace=True)

    # preprocess ECG signal using default method from biopsykit
    ecg_processer = EcgProcessor(raw_ecg, sampling_rate=256)
    ecg_processer.ecg_process()

    # return cleaned ECG signal
    clean_ecg = ecg_processer.ecg_result["Data"][["ECG_Clean"]].rename(columns={"ECG_Clean": "ecg"})
    return clean_ecg


def _extract_edr(hr_signal: pd.DataFrame, sampling_rate: int):
    """
    Extract EDR signal from HR signal using Charlton's method:
    Mean amplitude of troughs and proceeding peaks (Charlton et al. 2016a)

    hr_signal: HR signal for EDR extraction
    sampling_rate: sampling rate of the HR signal
    return: extracted EDR signal
    """

    # Perform EDR extraction
    EDR_extraction = ExtractionCharlton()
    edr_signal = EDR_extraction.extract(hr_signal, sampling_rate=sampling_rate)

    return edr_signal
