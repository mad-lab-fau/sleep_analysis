import pickle
from pathlib import Path

import pandas as pd
from tpcp.optimize import Optimize

import sleep_analysis.classification.utils.scoring as sc
from sleep_analysis.classification.ml_algorithms.ml_pipeline_helper import _get_sleep_stage_labels
from sleep_analysis.classification.ml_algorithms.xgboost_classifier import (
    OptunaSearch,
    XGBPipeline,
    get_study_params,
    create_search_space,
)
from sleep_analysis.classification.utils.data_loading import load_dataset


"""
change list of modalities to select the data modality to train with - options:  ACT, HRV, RRV, EDR
change dataset_name to change between the different datasets - options: MESA_Sleep
"""

# modality MUST be a list from either ACT, HRV, RRV, or EDR
modality = ["ACT", "HRV", "RRV"]
dataset_name = "Radar"
# classification type: can be either binary, 3stage, 4stage or 5stage
classification = "5stage"
small = False

print("Run with following parameters:")
print("modality " + " ,".join(modality))
print("classification: " + classification)
print("dataset_name: " + dataset_name)
print("small: " + str(small))

dataset, group_labels = load_dataset(dataset_name, small=small)

pipe = XGBPipeline(modality=modality, classification_type=classification)

train, test = dataset

opti = OptunaSearch(
    pipe,
    get_study_params,
    create_search_space=create_search_space,
    score_function=sc.score,
    n_trials=200,
    random_seed=42,
)

opti = opti.optimize(train)
print(
    f"The best performance was achieved with the parameters {opti.best_params_} and an MCC-score of {opti.best_score_}."
)


# xgb_optuna = XGBOptuna(
#    pipeline=pipe, score_function=sc.score, seed=360, modality=modality, classification_type=classification
# )
# xgb_optuna = xgb_optuna.optimize(train)

optimized_pipeline = opti.optimized_pipeline_

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
    final_results[subj.index["subj_id"][0]] = sc.score(optimizable_pipeline.optimized_pipeline_, subj)

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
