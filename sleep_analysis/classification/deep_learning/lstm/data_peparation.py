import biopsykit as bp
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from tpcp import Dataset

from sleep_analysis.feature_extraction.d04_main.combine_clean import _map_sleep_phases_to_num


def create_tensor(x_normalized, y_mat):
    # convert into tensor
    x_tensor_final = Variable(torch.Tensor(x_normalized))
    y_tensor = Variable(torch.Tensor(y_mat))

    # reshape it to (number of windows, window_length, number of features)
    y_tensor = torch.reshape(y_tensor, (y_tensor.shape[0], 1))

    return x_tensor_final, y_tensor


def test_to_list(subj, x_list, y_list, x_test_subj, y_test_subj):
    if subj.__class__.__name__ == "RealWorldIMUSet" or subj.__class__.__name__ == "RealWorldData":  # Deprecated
        x_list.append([x_test_subj, subj.index["subj_id"][0] + "_" + subj.index["night"][0]])
        y_list.append([y_test_subj, subj.index["subj_id"][0] + "_" + subj.index["night"][0]])
    elif subj.__class__.__name__ == "D04MainStudy":
        x_list.append([x_test_subj, subj.index["subj_id"][0]])
        y_list.append([y_test_subj, subj.index["subj_id"][0]])
    else:
        x_list.append([x_test_subj, subj.index["subj_id"][0]])
        y_list.append([y_test_subj, subj.index["subj_id"][0]])
    return x_list, y_list


def batchify(x_mat):
    return np.array_split(x_mat, 10)


class DataPreparation:
    """
    Data Preparation class:
    Returns sequential data for the corresponding input modality and a combination of those
    Supported modalities:
    *ACT:       Actigraphy
    *HRV:       Heart Rate Variability
    *RRV:       Respiration
    :param seq_len: Sequence length; Sequence that gets fed into LSTM
    :param overlap: Overlap of Sequences: Highly impacts runtime
    """

    def __init__(self, seq_len, overlap):
        self.seq_len = seq_len
        self.overlap = overlap

    def get_sequence_data(self, features: pd.DataFrame, ground_truth: pd.DataFrame, overlap, padding=False):
        """
        Create sequential data using biopsykit
        :param features: features that should be sequential
        :param ground_truth: ground truth to crop it to same length (sliding window cuts window/s from beginning and end)
        """

        if overlap is None:
            self.overlap = None

        feature_arr = np.asarray(features)
        ground_truth_arr = np.asarray(ground_truth)

        if padding:
            npad = ((int(self.seq_len / 2), int(self.seq_len / 2)), (0, 0))
            feature_arr = np.pad(feature_arr, npad, mode="mean")
            y_mat = ground_truth_arr
            x_mat = bp.utils.array_handling.sliding_window(
                feature_arr.squeeze(), overlap_percent=self.overlap, window_samples=self.seq_len
            )

        if not padding:
            y_mat = bp.utils.array_handling.sliding_window(
                ground_truth_arr.squeeze(), overlap_percent=self.overlap, window_samples=self.seq_len
            )[:, int(self.seq_len - 1)][:-1]
            x_mat = bp.utils.array_handling.sliding_window(
                feature_arr.squeeze(), overlap_percent=self.overlap, window_samples=self.seq_len
            )[:-1]

        return x_mat, y_mat

    def scale_data(self, x_mat, scaler: StandardScaler):
        """
        Normalize all datasets with the scaling obtained from training set
        If train set: initialize new scaler - If val/test: take scaler from train_set obtained from parameters
        :param x_mat: sequential data of x_tensor
        :param scaler: StandardScaler if val/test set or None and new initialisation if train set
        """
        if scaler is None:
            scaler = StandardScaler()
            for data in batchify(x_mat):
                scaler.partial_fit(data.reshape(-1, data.shape[-1]))
            x_normalized = scaler.transform(x_mat.reshape(-1, x_mat.shape[-1])).reshape(x_mat.shape)
        else:
            x_normalized = scaler.transform(x_mat.reshape(-1, x_mat.shape[-1])).reshape(x_mat.shape)

        return x_normalized, scaler

    def get_data(
        self,
        dataset: Dataset,
        modality: list,
        scaler: StandardScaler = None,
        overlap=None,
        classification_type="binary",
        padding=False,
    ):
        """
        Returns and processes the data and brings them into the correct way to feed into the deep learning network
        :dataset: Dataset of tpcp class
        :modality: list of modalities to extract
        :scaler: Scaler to scale the respective data
        :overlap: overlap that is used in the sliding window method
        """
        x_dict = {}
        y_dict = {}

        for subj in dataset:
            features = pd.DataFrame()
            all_features = subj.feature_table

            if "ACT" in modality:
                movement_features = all_features.filter(regex="_acc")[
                    [
                        "_acc_mean_1",
                    ]
                ]
                features = pd.concat([features, movement_features], axis=1)
            if "HRV" in modality:
                if dataset.__class__.__name__ == "D04MainStudy":
                    hrv_features = all_features.filter(regex="_hrv")[
                     [
                        "30_hrv_median_nni",
                        "30_hrv_ratio_sd2_sd1",
                        "150_hrv_median_nni",
                        "150_hrv_vlf",
                        "150_hrv_lf",
                        "150_hrv_hf",
                        "150_hrv_lf_hf_ratio",
                        "150_hrv_total_power",
                    ]
                ]
                elif dataset.__class__.__name__ == "MesaDataset":
                    hrv_features = all_features.filter(regex="_hrv")[
                        [
                        "_hrv_median_nni",
                        "_hrv_ratio_sd2_sd1",
                        "_hrv_median_nni",
                        "_hrv_vlf",
                        "_hrv_lf",
                        "_hrv_hf",
                        "_hrv_lf_hf_ratio",
                        "_hrv_total_power",
                    ]
                ]
                else:
                    raise AttributeError("Dataset not known")

                features = pd.concat([features, hrv_features], axis=1)
            if "RRV" in modality:
                rrv_features = all_features.filter(regex="RRV")[
                    [
                        "150_RRV_MedianBB",
                        "150_RRV_LF",
                        "270_RRV_MCVBB",
                        "150_RRV_CVBB",
                    ]
                ]
                # "150_RRV_MedianBB",
                # "150_RRV_RMSSD",
                # "150_RRV_SampEn",
                # "150_RRV_LFHF",

                features = pd.concat([features, rrv_features], axis=1)

            if "EDR" in modality:
                edr_features = subj.features.filter(regex="EDR")[
                    [
                        "150_EDR_MeanBB",
                        "150_EDR_LF",
                        "150_EDR_HF",
                        "150_EDR_LFHF",
                    ]
                ]
                features = pd.concat([features, edr_features], axis=1)

            if classification_type == "binary":
                ground_truth = subj.ground_truth["sleep"]
            else:
                ground_truth = subj.ground_truth
                ground_truth = ground_truth[classification_type]

            x_mat, y_mat = self.get_sequence_data(features, ground_truth, overlap=overlap, padding=padding)

            x_dict[subj.index["subj_id"][0]] = x_mat
            y_dict[subj.index["subj_id"][0]] = y_mat

        x_mat = np.concatenate([x_dict[x] for x in x_dict])
        y_mat = np.concatenate([y_dict[y] for y in y_dict])

        x_normalized, scaler = self.scale_data(x_mat, scaler)

        x_tensor_final, y_tensor = create_tensor(x_normalized, y_mat)

        return x_tensor_final, y_tensor, scaler

    def get_final_tensors(self, modality, train: Dataset, val: Dataset, test: Dataset, classification_type="binary"):
        """
        Return final sequential tensors for each input modality
        This is the function that gets called in class LSTM_Optuna
        :param modality: input modality of datastream
        :param train: Training set
        :param val: Validation set
        :param test: Test set
        """

        mod_set = {"HRV", "ACT", "RRV", "EDR"}
        if not all({mod}.issubset(mod_set) for mod in modality):
            raise AttributeError("modality MUST be list of either HRV, ACT, RRV, EDR")

        x_train, y_train, scaler = self.get_data(
            train,
            scaler=None,
            overlap=self.overlap,
            modality=modality,
            classification_type=classification_type,
            padding=True,
        )
        x_val, y_val, scaler = self.get_data(
            val,
            scaler=scaler,
            overlap=self.overlap,
            modality=modality,
            classification_type=classification_type,
            padding=True,
        )
        x_test = []
        y_test = []
        for subj in test:
            x_test_subj, y_test_subj, sc = self.get_data(
                subj,
                scaler=scaler,
                overlap=None,
                modality=modality,
                classification_type=classification_type,
                padding=True,
            )
            x_test, y_test = test_to_list(subj, x_test, y_test, x_test_subj, y_test_subj)
        return x_train, y_train, x_val, y_val, x_test, y_test
