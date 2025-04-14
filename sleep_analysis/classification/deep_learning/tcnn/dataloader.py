import torch

from sleep_analysis.classification.deep_learning.lstm.data_peparation import DataPreparation, test_to_list


class Dataloader:
    def __init__(self, seq_len, dataset_name="MESA_Sleep", overlap=0.95, classification_type="binary"):
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.overlap = overlap
        self.classification_type = classification_type

    def get_final_tensors(self, modality, train, val, test):
        # check if modality is a list of either HRV, ACT, RRV or EDR
        mod_set = {"HRV", "ACT", "RRV", "EDR"}
        if not all({mod}.issubset(mod_set) for mod in modality):
            raise AttributeError("modality MUST be list of either HRV, ACT, RRV, EDR")

        data_processor = DataPreparation(seq_len=self.seq_len, overlap=self.overlap)

        # get train data
        x_train, y_train, scaler = data_processor.get_data(
            train,
            modality=modality,
            scaler=None,
            overlap=None,
            classification_type=self.classification_type,
            padding=True,
        )

        # get validation data
        x_val, y_val, scaler = data_processor.get_data(
            val,
            modality=modality,
            scaler=scaler,
            overlap=None,
            classification_type=self.classification_type,
            padding=True,
        )

        # get test data
        x_test = []
        y_test = []
        for subj in test:
            x_test_subj, y_test_subj, sc = data_processor.get_data(
                subj,
                modality=modality,
                scaler=scaler,
                overlap=None,
                classification_type=self.classification_type,
                padding=True,
            )

            # reshape data to 3D if only ACT is used
            if modality == ["ACT"]:
                x_test_subj = torch.reshape(x_test_subj, (x_test_subj.shape[0], x_test_subj.shape[1], 1))

            x_test_subj = x_test_subj.reshape((x_test_subj.shape[0], x_test_subj.shape[2], x_test_subj.shape[1]))
            x_test, y_test = test_to_list(subj, x_test, y_test, x_test_subj, y_test_subj)

        if modality == ["ACT"]:
            x_val = torch.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
            x_train = torch.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # reshape data to TCN input format
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[2], x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[2], x_val.shape[1]))
        print("Size of train tensors: " + str(x_train.shape))
        print("Size of val tensors: " + str(x_val.shape))

        return x_train, y_train, x_val, y_val, x_test, y_test
