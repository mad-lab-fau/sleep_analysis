from biopsykit.sleep.sleep_wake_detection.sleep_wake_detection import SleepWakeDetection
from tpcp import Pipeline

from sleep_analysis.datasets.mesadataset import MesaDataset


class ScalePipeline(Pipeline):
    def __init__(
        self,
        scale_value: float = 0.1,
        rescore_data: bool = True,
        algorithm: str = "scripps_clinic",
        dataset_name="dataset_name",
        classification_type="binary",
    ):
        self.scale_value = scale_value
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
        if self.algorithm == "webster" or self.algorithm == "cole_kripke":
            # webster and cole-kripe-algorithm are sampled in 60s windows, while our dataset is based on 30s windows.
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
        algo = SleepWakeDetection(algorithm_type=self.algorithm, scale_factor=self.scale_value)

        # classification using heuristic algorithm from biopsykit
        self.classification_ = algo.predict(
            actigraph_data, rescore_data=self.rescore_data, epoch_length=self.epoch_length
        )

        return self
