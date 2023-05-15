from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tpcp import OptimizableParameter, OptimizablePipeline, cf

from sleep_analysis.datasets.mesadataset import MesaDataset


class MLPPipeline(OptimizablePipeline):
    classifier: OptimizableParameter

    def __init__(
        self,
        modality,
        hidden_layer_sizes=(100,),
        activation="identity",
        solver="lbfgs",
        alpha=0.0001,
        learning_rate="constant",
        classifier=cf(
            Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(early_stopping=True, random_state=1))])
        ),
        classification_type="binary",
    ):
        self.classifier = classifier
        self.epoch_length = 30
        self.modality = modality
        self.algorithm = "mlp"

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
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
            "clf__hidden_layer_sizes": self.hidden_layer_sizes,
            "clf__activation": self.activation,
            "clf__solver": self.solver,
            "clf__alpha": self.alpha,
            "clf__learning_rate": self.learning_rate,
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
