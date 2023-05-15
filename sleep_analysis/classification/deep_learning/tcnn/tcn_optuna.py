from pathlib import Path

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from sleep_analysis.classification.deep_learning.tcnn.dataloader import Dataloader
from sleep_analysis.classification.deep_learning.tcnn.tcnn_main import TcnMain
from sleep_analysis.classification.deep_learning.utils import get_num_classes, get_num_input
from sleep_analysis.classification.ml_algorithms.ml_pipeline_helper import _get_sleep_stage_labels
from sleep_analysis.classification.utils.utils import get_db_path
from sleep_analysis.datasets.mesadataset import MesaDataset


class TCNOptuna:
    def __init__(self, modality, dataset_name, seq_len, seed=1, classification_type="binary"):
        self.res = {}
        self.seed = seed
        self.modality = modality
        self.best_parameters = None
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.classification_type = classification_type
        self.output_size = get_num_classes(self.classification_type)

        self.num_inputs = get_num_input(modality)

    def optimize(self, dataset: MesaDataset, **optimize_params):
        """Apply optuna optimization on the input parameters of the pipeline."""

        # Split dataset into train, val and test set: (80/20 splits)
        train_set, test_set = dataset.get_random_split(dataset=dataset, train_size=0.8)
        train_set, val_set = dataset.get_random_split(dataset=train_set, train_size=0.8)

        # load data in the respective format using custom dataloader
        data_loader = Dataloader(
            seq_len=self.seq_len,
            dataset_name=self.dataset_name,
            overlap=0.95,
            classification_type=self.classification_type,
        )
        x_train, y_train, x_val, y_val, x_test, y_test = data_loader.get_final_tensors(
            self.modality, train_set, val_set, test_set
        )

        def objective(trial):
            # Define parameters to be optimized
            paras_to_be_searched = {
                "num_chanels": trial.suggest_int("num_chanels", 2, 6),
                "n_hid": trial.suggest_int("n_hid", 8, 512, 32),
                "kernel_size": trial.suggest_int("kernel_size", 2, 5),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
                "learning_rate": trial.suggest_float("learning_rate", 0.0001, 1),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            }

            # Create model with the respective parameters
            model = TcnMain(
                num_inputs=self.num_inputs,
                output_size=self.output_size,
                num_chanels=[paras_to_be_searched["n_hid"]] * (paras_to_be_searched["num_chanels"] - 1),
                kernel_size=paras_to_be_searched["kernel_size"],
                dropout=paras_to_be_searched["dropout"],
                learning_rate=paras_to_be_searched["learning_rate"],
                batch_size=paras_to_be_searched["batch_size"],
                modality=self.modality,
                dataset_name=self.dataset_name,
                classification_type=self.classification_type,
            )

            # train model
            max_val_acc = model.train(x_train, y_train, x_val, y_val, num_epochs=80)

            # Return only the mean of all performance-scores
            return max_val_acc

        # Create and run an optuna study + save trial information
        db_path = get_db_path()
        study = optuna.create_study(
            direction="maximize",
            study_name="tcn_" + "_".join(self.modality),
            sampler=TPESampler(seed=self.seed),
            storage="sqlite:////"
            + db_path
            + "/tcn_"
            + "_".join(self.modality)
            + "_"
            + self.classification_type
            + ".db",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner,
        )

        study.optimize(objective, n_trials=1, show_progress_bar=True)

        # Save best parameters
        self.best_parameters = study.best_params

        # Set the best params in a new cloned pipeline, refit it and save it
        model = TcnMain(
            num_inputs=self.num_inputs,
            output_size=self.output_size,
            num_chanels=[self.best_parameters["n_hid"]] * (self.best_parameters["num_chanels"] - 1),
            kernel_size=self.best_parameters["kernel_size"],
            dropout=self.best_parameters["dropout"],
            learning_rate=self.best_parameters["learning_rate"],
            batch_size=self.best_parameters["batch_size"],
            modality=self.modality,
            dataset_name=self.dataset_name,
            classification_type=self.classification_type,
        )

        print(study.best_params)

        x_train, y_train, x_val, y_val, x_test, y_test = data_loader.get_final_tensors(
            self.modality, train_set, val_set, test_set
        )

        # retrain best model with the best parameters found in optuna optimization
        model.train(x_train, y_train, x_val, y_val, num_epochs=80)

        # Apply trained model on test set
        subject_results, score_mean = model.test(x_test, y_test)

        # save results subject-wise as .csv file
        subject_results.index.name = "metric"
        subject_results.to_csv(
            Path(__file__)
            .parents[4]
            .joinpath(
                "exports/results_per_algorithm/TCN/TCN_"
                + "benchmark"
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
                "exports/results_per_algorithm/TCN/"
                "confusion_matrix_TCN"
                + "_"
                + "benchmark"
                + "_"
                + "_".join(self.modality)
                + "_"
                + self.classification_type
                + ".csv"
            ),
            index=True,
        )

        return self
