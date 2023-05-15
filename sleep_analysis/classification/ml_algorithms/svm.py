from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tpcp import OptimizableParameter, OptimizablePipeline, cf

from sleep_analysis.datasets.mesadataset import MesaDataset


class SVMPipeline(OptimizablePipeline):
    classifier: OptimizableParameter

    def __init__(
        self,
        modality,
        loss="hinge",
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        class_weight=None,
        warm_start=False,
        epsilon=0.1,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        max_iter=1000,
        early_stopping=False,
        k=20,
        classifier=cf(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("reduce_dim", SelectKBest()),
                    ("clf", SGDClassifier(shuffle=False, average=False, n_jobs=1, random_state=42)),
                ]
            )
        ),
        classification_type="binary",
    ):
        self.classifier = classifier
        self.modality = modality
        self.class_weight = class_weight
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.k = k
        self.classification_type = classification_type

        self.epoch_length = 30
        self.algorithm = "svm"

    def self_optimize(self, dataset: MesaDataset, **kwargs):
        """
        Optimization of whole trainset
        :param dataset: Dataset instance representing the whole train set with its sleep data
        """
        # Concat whole dataset to one DataFrame
        features, ground_truth = dataset.get_concat_dataset(dataset, modality=self.modality)

        # set pipeline parameters from GridSearchCV
        c = self._set_classifier_params(clone(self.classifier))

        # train the model
        if self.classification_type == "binary":
            self.classifier = c.fit(features, ground_truth["sleep"])
        else:
            self.classifier = c.fit(features, ground_truth[self.classification_type])

        return self

    def _set_classifier_params(self, classifier):
        params = {
            "clf__class_weight": self.class_weight,
            "clf__loss": self.loss,
            "clf__penalty": self.penalty,
            "clf__alpha": self.alpha,
            "clf__l1_ratio": self.l1_ratio,
            "clf__fit_intercept": self.fit_intercept,
            "clf__warm_start": self.warm_start,
            "clf__epsilon": self.epsilon,
            "clf__learning_rate": self.learning_rate,
            "clf__eta0": self.eta0,
            "clf__power_t": self.power_t,
            "clf__max_iter": self.max_iter,
            "clf__early_stopping": self.early_stopping,
            "reduce_dim__k": self.k,
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
