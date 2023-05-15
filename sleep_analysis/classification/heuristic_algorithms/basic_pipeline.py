from biopsykit.sleep.sleep_wake_detection.sleep_wake_detection import SleepWakeDetection
from tpcp import Dataset, Pipeline

from sleep_analysis.datasets.mesadataset import MesaDataset


class BasicPipeline(Pipeline):
    rescore_data: bool
    algorithm: str

    def __init__(
        self,
        rescore_data: bool = True,
        algorithm: str = "sadeh",
        dataset_name="dataset_name",
        classification_type="binary",
    ):
        self.rescore_data = rescore_data
        self.algorithm = algorithm
        self.epoch_length = 30
        self.dataset_name = dataset_name
        self.classification_type = classification_type

    def run(self, datapoint: MesaDataset):
        """
        Subject-wise classification based on heuristic algorithm
        :param datapoint: Dataset instance representing the sleep data of one participant
        """
        if self.algorithm == "sadeh":
            # The input data for the sadeh algorithm has to be sampled in 60s windows, while our dataset is based on 30s windows.
            # As the actigraphy counts are cumulative, we sum up two consecutive samples
            self.epoch_length = 60
            actigraph_data = (
                datapoint.actigraph_data[["activity"]][::2].reset_index()[["activity"]]
                + datapoint.actigraph_data[["activity"]][1::2].reset_index()[["activity"]]
            ).fillna(0.0)
        else:
            self.epoch_length = 30
            actigraph_data = datapoint.actigraph_data

        # create instance of algorithm which is implemented in biopsykit
        algo = SleepWakeDetection(algorithm_type=self.algorithm)

        # classification using heuristic algorithm from biopsykit
        self.classification_ = algo.predict(
            actigraph_data, rescore_data=self.rescore_data, epoch_length=self.epoch_length
        )

        return self
