import pickle
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler

from sleep_analysis.classification.deep_learning.lstm.data_peparation import DataPreparation
from sleep_analysis.classification.deep_learning.lstm.LSTM import LSTM
from sleep_analysis.classification.deep_learning.utils import get_num_input
from sleep_analysis.classification.ml_algorithms.ml_pipeline_helper import _get_sleep_stage_labels
from sleep_analysis.classification.utils.utils import get_db_path
from sleep_analysis.datasets.mesadataset import MesaDataset


class LSTM_Optuna:
    """
    Optuna class for LSTM to do a randomized hyperparameter search
    """

    def __init__(self, modality, dataset_name, seed=1, classification_type="binary"):
        self.seed = seed
        self.modality = modality
        self.dataset_name = dataset_name
        self.seq_len = 101
        self.classification_type = classification_type

        # find number of input files dependent on modality and data source
        self.num_inputs = get_num_input(modality)

        # fix seeds
        torch.manual_seed(seed=42)
        torch.cuda.manual_seed(seed=42)
        torch.cuda.manual_seed_all(seed=42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def optimize(self, dataset: MesaDataset, **optimize_params):
        """Apply optuna optimization on the input parameters of the pipeline."""

        # Split dataset into train, val and test set: (80/20 splits)
        train_set, test_set = dataset.get_random_split(dataset=dataset)
        train_set, val_set = dataset.get_random_split(dataset=train_set)

        # load data in the respective format using custom dataloader
        print("load data...")
        data_loader = DataPreparation(seq_len=self.seq_len, overlap=0.95)
        x_train, y_train, x_val, y_val, x_test, y_test = data_loader.get_final_tensors(
            self.modality, train_set, val_set, test_set, self.classification_type
        )

        print("Shape of train tensors:" + str(x_train.shape) + " , " + str(y_train.shape))
        print("Shape of val tensors:" + str(x_val.shape) + " , " + str(y_val.shape))

        def objective(trial):
            paras_to_be_searched = {
                "seq_len": trial.suggest_categorical("seq_len", [21, 51, 101]),
                "hidden_size": trial.suggest_int("hidden_size", 4, 600, 4),
                "num_layers": trial.suggest_int("num_layers", 1, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.005),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
            }

            # Initialize model with the parameters to be searched
            model = LSTM(
                num_epochs=80,
                input_size=self.num_inputs,
                hidden_size=paras_to_be_searched["hidden_size"],
                num_layers=paras_to_be_searched["num_layers"],
                learning_rate=paras_to_be_searched["learning_rate"],
                seq_len=paras_to_be_searched["seq_len"],
                batch_size=paras_to_be_searched["batch_size"],
                modality=self.modality,
                dataset_name=self.dataset_name,
                classification_type=self.classification_type,
            )
            # train model
            print("train...")
            max_val_performance = model.train(x_train, y_train, x_val, y_val)

            # Return only the mean of all performance-scores
            return max_val_performance

        # Create and run an optuna study + save trial information
        db_path = get_db_path()
        study = optuna.create_study(
            direction="maximize",
            study_name="lstm_" + "_".join(self.modality),
            sampler=TPESampler(seed=self.seed),
            storage="sqlite:////"
            + db_path
            + "/lstm_"
            + "_".join(self.modality)
            + "_"
            + self.classification_type
            + ".db",
            load_if_exists=True,
        )

        study.optimize(objective, n_trials=1, show_progress_bar=True)
        self.best_parameters = study.best_params
        print(study.best_params)
        model = LSTM(
            num_epochs=80,
            input_size=self.num_inputs,
            hidden_size=self.best_parameters["hidden_size"],
            num_layers=self.best_parameters["num_layers"],
            learning_rate=self.best_parameters["learning_rate"],
            seq_len=self.best_parameters["seq_len"],
            batch_size=self.best_parameters["batch_size"],
            modality=self.modality,
            dataset_name=self.dataset_name,
            classification_type=self.classification_type,
        )

        # retrain best model with the best parameters found in optuna optimization
        model.train(x_train, y_train, x_val, y_val)

        # Apply trained model on test set
        subject_results, score_mean, pred_dict = model.test(x_test, y_test)

        # save results subject-wise as .csv file
        subject_results.index.name = "metric"
        subject_results.to_csv(
            Path(__file__)
            .parents[4]
            .joinpath(
                "exports/results_per_algorithm/LSTM/LSTM_"
                + self.dataset_name
                + "_"
                + "_".join(self.modality)
                + "_"
                + self.classification_type
                + ".csv"
            )
        )

        # compute overall confusion matrix and save it as .csv file
        sleep_stage_labels, conf_matrix = _get_sleep_stage_labels(self.classification_type)
        for subj in subject_results.keys():
            conf_matrix += subject_results[subj]["confusion_matrix"].get_value()

        conf_matrix = pd.DataFrame(conf_matrix, index=sleep_stage_labels, columns=sleep_stage_labels)

        conf_matrix.to_csv(
            Path(__file__)
            .parents[4]
            .joinpath(
                "exports/results_per_algorithm/LSTM/"
                "confusion_matrix_LSTM"
                + "_"
                + self.dataset_name
                + "_"
                + "_".join(self.modality)
                + "_"
                + self.classification_type
                + ".csv"
            ),
            index=True,
        )

        # save predictions as .pickle file
        with open(
            Path(__file__)
            .parents[4]
            .joinpath(
                "exports/results_per_algorithm/LSTM/predictions/predictions"
                + "_"
                + self.dataset_name
                + "_"
                + "_".join(self.modality)
                + "_"
                + self.classification_type
                + ".pickle"
            ),
            "wb",
        ) as file:
            pickle.dump(pred_dict, file)

        return self
