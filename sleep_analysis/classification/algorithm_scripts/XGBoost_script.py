import pickle
from pathlib import Path

import pandas as pd
from tpcp.optimize import Optimize

import sleep_analysis.classification.utils.scoring as sc
from sleep_analysis.classification.ml_algorithms.ml_pipeline_helper import _get_sleep_stage_labels
from sleep_analysis.classification.ml_algorithms.xgboost_classifier import XGBOptuna, XGBPipeline
from sleep_analysis.classification.utils.data_loading import load_dataset

"""
change list of modalities to select the data modality to train with - options:  ACT, HRV, RRV, EDR
change dataset_name to change between the different datasets - options: MESA_Sleep
"""

# modality MUST be a list from either ACT, HRV, RRV, or EDR
modality = ["ACT", "HRV"]
dataset_name = "MESA_Sleep"
# classification type: can be either binary, 3stage, 4stage or 5stage
classification = "binary"
small = False

print("Run with following parameters:")
print("modality " + " ,".join(modality))
print("classification: " + classification)
print("dataset_name: " + dataset_name)
print("small: " + str(small))

dataset, group_labels = load_dataset(dataset_name, small=small)

pipe = XGBPipeline(modality=modality, classification_type=classification)

train, test = dataset
xgb_optuna = XGBOptuna(
    pipeline=pipe, score_function=sc.score, seed=360, modality=modality, classification_type=classification
)
xgb_optuna = xgb_optuna.optimize(train)

optimized_pipeline = xgb_optuna.optimized_pipeline_

optimizable_pipeline = Optimize(optimized_pipeline)
optimizable_pipeline = optimizable_pipeline.optimize(train)

print("save pickle file...")
file = open(
    Path(__file__)
    .parents[3]
    .joinpath(
        "exports/pickle_pipelines/xgb_" + dataset_name + "_" + "_".join(modality) + "_" + classification + ".obj"
    ),
    "wb",
)
pickle.dump(optimizable_pipeline, file)

print("... done!")

final_results = {}

for subj in test:
    final_results[subj.index["mesa_id"][0]] = sc.score(optimizable_pipeline.optimized_pipeline_, subj)

sleep_stage_labels, conf_matrix = _get_sleep_stage_labels(classification)

for subj in final_results.keys():
    conf_matrix += final_results[subj]["confusion_matrix"].get_value()

final_results = pd.DataFrame(final_results)
final_results.index.name = "metric"

final_results.to_csv(
    Path(__file__)
    .parents[3]
    .joinpath(
        "exports/results_per_algorithm/xgb/xgb_"
        + dataset_name
        + "_"
        + "_".join(modality)
        + "_"
        + classification
        + ".csv"
    ),
    index=True,
)

conf_matrix = pd.DataFrame(conf_matrix, index=sleep_stage_labels, columns=sleep_stage_labels)
pd.DataFrame(conf_matrix).to_csv(
    Path(__file__)
    .parents[3]
    .joinpath(
        "exports/results_per_algorithm/"
        + "xgb"
        + "/confusion_matrix_"
        + "xgb"
        + "_"
        + dataset_name
        + "_"
        + "_".join(modality)
        + "_"
        + classification
        + ".csv"
    ),
    index=True,
)
