import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from tpcp.optimize import GridSearchCV
from tpcp.validate import cross_validate

import sleep_analysis.classification.utils.scoring as sc


def nested_cv_optimization(pipe, parameters, algorithm, dataset, dataset_name, modality, group_labels):
    """
    In this work, nested cv optimization is used for sleep/wake detection using the real-world sleep dataset containing ecg and imu data
    :input_param: pipe: Pipeline of type OptimizablePipeline of tpcp
    :input_param: parameters: Parameters for hyperparameter search via GridSearch
    :input_param: algorithm: Algorithm to optimize (just needed for name of .csv file)
    :input_param: dataset: Data to optimize with (Datasetclass of IMUDataset/ RealWorldDataset)
    :input_param: dataset_name: Which study to optimize (IMU/ Real-World Actigraphy/ Pretrained)
    :input_param: modality: Which input modalities? Actigraphy/HRV/Respiration/Combination of those
    input_param: group_labels: This parameter prevents inter-subject splitting in cross validation splits
    :returns: nothing - result get saved as csv at the end of the function
    """
    cv = GroupKFold(n_splits=5)

    gs = GridSearchCV(
        pipe,
        parameters,
        scoring=sc.score,
        cv=cv,
        return_optimized="kappa",
        return_train_score=False,
        n_jobs=-1,
        pre_dispatch="n_jobs/1.3",
        verbose=10,
    )
    results = cross_validate(
        gs,
        dataset=dataset,
        groups=group_labels,
        cv=cv,
        scoring=sc.score,
        return_optimizer=True,
        return_train_score=False,
        propagate_groups=True,
    )

    file = open(
        Path(__file__)
        .parents[1]
        .joinpath(
            "exports/results_per_algorithm/"
            + algorithm
            + "/"
            + algorithm
            + "_results_"
            + dataset_name
            + "_"
            + modality
            + ".pkl"
        ),
        "wb",
    )
    pickle.dump(results, file)


def hold_out_optimization(
    pipe,
    parameters,
    dataset,
    algorithm,
    dataset_name,
    modality,
    classification_type="binary",
    n_jobs=-1,
    pre_dispatch="n_jobs/1.7",
):
    cv = KFold(n_splits=5)
    train, test = dataset

    # optimization parameter for Grid Search
    optimize_param = "mcc"

    gs = GridSearchCV(
        pipe,
        parameters,
        scoring=sc.score,
        cv=cv,
        return_optimized=optimize_param,
        return_train_score=False,
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch,
        verbose=10,
    )
    gs = gs.optimize(train)

    # save results of Grid Search to .csv file
    pd.DataFrame(gs.cv_results_).to_csv(
        Path(__file__)
        .parents[3]
        .joinpath(
            "exports/results_per_algorithm/"
            + algorithm
            + "/"
            + algorithm
            + "_gridsearch_"
            + dataset_name
            + "_"
            + "_".join(modality)
            + "_"
            + classification_type
            + ".csv"
        ),
        index=True,
    )

    # save optimized pipeline to .obj file
    optimized_pipeline = gs.optimized_pipeline_
    file = open(
        Path(__file__)
        .parents[3]
        .joinpath(
            "exports/pickle_pipelines/"
            + algorithm
            + "_"
            + dataset_name
            + "_"
            + "_".join(modality)
            + "_"
            + classification_type
            + ".obj"
        ),
        "wb",
    )
    pickle.dump(optimized_pipeline, file)

    final_results = {}

    # apply pipeline to each subject in test set and get classification performance
    for subj in test:
        final_results[subj.index["mesa_id"][0]] = sc.score(optimized_pipeline, subj)

    # compute confusion matrix for all subjects
    sleep_stage_labels, conf_matrix = _get_sleep_stage_labels(classification_type)
    for subj in final_results.keys():
        conf_matrix += final_results[subj]["confusion_matrix"].get_value()

    # Save subject-wise results as .csv file
    final_results = pd.DataFrame(final_results)
    final_results.index.name = "metric"
    final_results.to_csv(
        Path(__file__)
        .parents[3]
        .joinpath(
            "exports/results_per_algorithm/"
            + algorithm
            + "/"
            + algorithm
            + "_"
            + dataset_name
            + "_"
            + "_".join(modality)
            + "_"
            + classification_type
            + ".csv"
        ),
        index=True,
    )

    # Save confusion matrix as .csv file
    conf_matrix = pd.DataFrame(conf_matrix, index=sleep_stage_labels, columns=sleep_stage_labels)
    pd.DataFrame(conf_matrix).to_csv(
        Path(__file__)
        .parents[3]
        .joinpath(
            "exports/results_per_algorithm/"
            + algorithm
            + "/confusion_matrix_"
            + algorithm
            + "_"
            + dataset_name
            + "_"
            + "_".join(modality)
            + "_"
            + classification_type
            + ".csv"
        ),
        index=True,
    )


def _get_sleep_stage_labels(classification_type):
    """Get sleep stage labels and empty confusion matrix dependent on classification type"""
    if classification_type == "binary":
        return ["wake", "sleep"], np.zeros((2, 2))
    elif classification_type == "3stage":
        return ["wake", "nrem", "rem"], np.zeros((3, 3))
    elif classification_type == "5stage":
        return ["wake", "n1", "n2", "n3", "rem"], np.zeros((5, 5))
    elif classification_type == "4stage":
        return ["wake", "light", "deep", "rem"], np.zeros((4, 4))
    else:
        raise ValueError("Classification type not supported")
