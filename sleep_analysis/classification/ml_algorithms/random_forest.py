import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from tpcp import OptimizableParameter, OptimizablePipeline, cf
from tpcp.optimize import Optimize
from tpcp.validate import cross_validate

from sleep_analysis.classification.utils.utils import get_db_path
from sleep_analysis.datasets.mesadataset import MesaDataset


class RandomForestPipeline(OptimizablePipeline):
    classifier: OptimizableParameter

    def __init__(
        self,
        modality,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=0.5,
        min_samples_leaf=1,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=None,
        ccp_alpha=0.0,
        classifier=cf(RandomForestClassifier(n_jobs=-1, verbose=0, random_state=42)),
        classification_type="binary",
    ):
        self.classifier = classifier
        self.modality = modality

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.ccp_alpha = ccp_alpha
        self.classification_type = classification_type

        self.epoch_length = 30
        self.algorithm = "rf"

    def self_optimize(self, dataset: MesaDataset, **kwargs):
        """
        Optimization of whole trainset
        :param dataset: Dataset instance representing the whole train set with its sleep data
        """
        # Concat whole dataset to one DataFrame
        features, ground_truth = dataset.get_concat_dataset(dataset, modality=self.modality)

        # Set classifier parameters from Optuna Optimization
        c = self._set_classifier_params(clone(self.classifier))

        # Train classifier
        if self.classification_type == "binary":
            self.classifier = c.fit(features, ground_truth["sleep"])
        else:
            self.classifier = c.fit(features, ground_truth[self.classification_type])
        return self

    def _set_classifier_params(self, classifier):
        params = {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "bootstrap": self.bootstrap,
            "ccp_alpha": self.ccp_alpha,
        }
        return classifier.set_params(**params)

    def run(self, datapoint: MesaDataset):
        """
        Subject-wise classification based on trained model
        :param datapoint: Dataset instance representing the sleep data of one participant
        """
        features = datapoint.get_features(datapoint, modality=self.modality)

        self.classification_ = self.classifier.predict(features)

        return self


from tpcp.optimize.optuna import CustomOptunaOptimize


class RandomForestOptuna(CustomOptunaOptimize):
    def __init__(self, pipeline, score_function, modality, classification_type: str = "binary", seed=1):
        self.pipeline = pipeline
        self.score_function = score_function
        self.res = {}
        self.seed = seed
        self.modality = modality
        self.classification_type = classification_type

    def optimize(self, dataset: MesaDataset, **optimize_params):
        """Apply optuna optimization on the input parameters of the pipeline."""

        def objective(trial):
            paras_to_be_searched = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 350, 10),
                "criterion": trial.suggest_categorical("criterion", ["gini"]),
                "max_depth": trial.suggest_int("max_depth", 5, 40, 1),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 70, 1),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, 1),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [None]),
                "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.0),
                "bootstrap": trial.suggest_categorical("bootstrap", [True]),
                "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.0),
            }

            # Clone the pipeline, so it will not change the original pipeline
            # Set the parameters that need to be searched in this cloned pipeline

            temp_pipeline = self.pipeline.clone().set_params(**paras_to_be_searched)

            # Run cross validation
            results = cross_validate(
                optimizable=Optimize(temp_pipeline),
                dataset=dataset,
                cv=5,
                scoring=self.score_function,
                return_optimizer=True,
                n_jobs=-1,
            )

            return np.mean(results["test_mcc"])

        # Create and run an optuna study + save trial information
        db_path = get_db_path()
        study = optuna.create_study(
            direction="maximize",
            study_name="RandomForestOptuna_" + "|".join(self.modality) + "_" + self.classification_type,
            sampler=TPESampler(seed=self.seed),
            storage="sqlite:////" + db_path + "/rf_" + "_".join(self.modality) + "_" + self.classification_type + ".db",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner,
        )

        study.optimize(objective, n_trials=1, show_progress_bar=True)

        best_parameters = study.best_params
        best_parameters["classifier"] = RandomForestClassifier(study.best_params)
        # Set the best params in a new cloned pipeline, refit it and save it
        self.optimized_pipeline_ = (
            Optimize(self.pipeline.clone().set_params(**best_parameters))
            .optimize(dataset, **(optimize_params or {}))
            .optimized_pipeline_
        )

        return self
