import json
from pathlib import Path

import numpy as np
import pandas as pd
from tpcp import Pipeline

from sleep_analysis.classification.utils.scoring import score
from sleep_analysis.datasets.mesadataset import MesaDataset


class BaselinePipeline(Pipeline):
    def __init__(self, baseline, dataset_name="benchmark", classification_type="binary"):
        self.epoch_length = 30
        self.dataset_name = dataset_name
        self.baseline = baseline
        self.classification_type = classification_type
        self.algorithm = "baseline"

    def run(self, datapoint: MesaDataset):
        if self.baseline == "always_sleep":
            self.classification_ = np.ones((len(datapoint.ground_truth)))
        elif self.baseline == "always_wake":
            self.classification_ = np.zeros((len(datapoint.ground_truth)))
        elif self.baseline == "ground_truth":
            self.classification_ = np.asarray(datapoint.ground_truth[["sleep"]])
        else:
            raise ValueError("baseline unknown")
        return self


dataset = MesaDataset()
train, test = dataset.get_random_split(dataset)

baseline_list = ["always_wake", "always_sleep", "ground_truth"]

baseline_dict = {}
for baseline_type in baseline_list:
    res_dict = {}

    pipe = BaselinePipeline(baseline=baseline_type)

    for subj in test:
        res_dict[subj.index["mesa_id"][0]] = score(pipe, subj)
    res = pd.DataFrame(res_dict).agg(["mean", "std"], axis=1)
    baseline_dict[baseline_type] = res
    print(baseline_type + " test - finished")

baseline = pd.concat(baseline_dict)

print(baseline)


baseline.to_csv(Path(__file__).parents[3].joinpath("exports/baseline/baseline_test_agg.csv"))


baseline_dict = {}
for baseline_type in baseline_list:
    res_dict = {}

    pipe = BaselinePipeline(baseline=baseline_type)

    for subj in train:
        res_dict[subj.index["mesa_id"][0]] = score(pipe, subj)
    res = pd.DataFrame(res_dict).agg(["mean", "std"], axis=1)
    baseline_dict[baseline_type] = res
    print(baseline_type + "train - finished")

baseline = pd.concat(baseline_dict)

print(baseline)


baseline.to_csv(Path(__file__).parents[3].joinpath("exports/baseline/baseline_train_agg.csv"))


baseline_dict = {}
baseline_list = ["ground_truth"]
for baseline_type in baseline_list:
    res_dict = {}

    pipe = BaselinePipeline(baseline=baseline_type)

    for subj in dataset:
        res_dict[subj.index["mesa_id"][0]] = score(pipe, subj)
    res = pd.DataFrame(res_dict)  # .agg(["mean", "std"], axis=1)
    baseline_dict[baseline_type] = res
    print(baseline_type + "full - finished")

baseline = pd.DataFrame(res_dict)

print(baseline)


baseline.to_csv(Path(__file__).parents[3].joinpath("exports/baseline/baseline_full.csv"))
