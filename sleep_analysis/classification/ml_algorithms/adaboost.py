from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from tpcp import OptimizableParameter, OptimizablePipeline, cf

from sleep_analysis.datasets.mesadataset import MesaDataset


class AdaBoostPipeline(OptimizablePipeline):
    classifier: OptimizableParameter

    def __init__(
        self,
        modality,
        n_estimators=50,
        learning_rate=1,
        algorithm="SAMME.R",
        classifier=cf(AdaBoostClassifier(random_state=1)),
        classification_type="binary",
    ):
        self.modality = modality
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.classifier = classifier
        self.epoch_length = 30
        self.classification_type = classification_type

    def self_optimize(self, dataset: MesaDataset, **kwargs):
        """
        Optimization of whole trainset
        :param dataset: Dataset instance representing the whole train set with its sleep data
        """
        # Concat whole dataset to one DataFrame
        features, ground_truth = dataset.get_concat_dataset(dataset, modality=self.modality)

        # set classifier parameters from GridSearchCV
        c = self._set_classifier_params(clone(self.classifier))

        # train classifier
        if self.classification_type == "binary":
            self.classifier = c.fit(features, ground_truth["sleep"])
        else:
            self.classifier = c.fit(features, ground_truth[self.classification_type])

        return self

    def _set_classifier_params(self, classifier):
        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "algorithm": self.algorithm,
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
