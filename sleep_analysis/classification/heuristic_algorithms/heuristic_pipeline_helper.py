import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, ParameterGrid
from tpcp import Pipeline
from tpcp.optimize import GridSearch
from tpcp.validate import cross_validate

import sleep_analysis.classification.utils.scoring as sc
from sleep_analysis.classification.heuristic_algorithms.scale_pipeline import ScalePipeline
from sleep_analysis.datasets.mesadataset import MesaDataset


def cv_optmization_group(
    pipe: [ScalePipeline, Pipeline],
    parameters: ParameterGrid,
    algorithm: str,
    dataset: MesaDataset,
    dataset_name: str,
    group_labels,
):
    # GroupKFold to prevent different nights of the same participant to be in train and test set
    cv = GroupKFold(n_splits=5)

    # GridSearch instance with pipeline, search-parameters, scoring function and optimization-metrics as input
    gs = GridSearch(pipe, parameters, scoring=sc.score, return_optimized="kappa", n_jobs=-1)

    # 5-fold cross-validation
    results = cross_validate(
        gs,
        dataset,
        scoring=sc.score,
        groups=group_labels,
        cv=cv,
        return_optimizer=True,
        return_train_score=False,
        verbose=10,
        n_jobs=-1,
    )

    file = open(
        Path.cwd().joinpath(
            "src/sleepwakebenchmarking/exports/results_per_algorithm/heuristic_algorithms/"
            + algorithm
            + dataset_name
            + ".pkl"
        ),
        "wb",
    )
    pickle.dump(results, file)


def cv_optimization(
    pipe: [ScalePipeline, Pipeline],
    parameters: ParameterGrid,
    dataset: tuple,
    algorithm: str,
    dataset_name: str,
    n_jobs=-1,
):
    train, test = dataset

    # KFold because every participant slept only one night in this study
    cv = KFold(n_splits=5)

    # GridSearch instance with pipeline, search-parameters, scoring function and optimization-metrics as input
    gs = GridSearch(pipe, parameters, scoring=sc.score, return_optimized="accuracy", n_jobs=1)

    # 5-fold cross-validation
    results = cross_validate(
        gs, train, scoring=sc.score, cv=cv, return_optimizer=True, return_train_score=True, verbose=10, n_jobs=n_jobs
    )

    # conversion of results to df
    result_df = pd.DataFrame(results)

    optimized_pipeline = result_df["optimizer"]
    optimi_pipeline = optimized_pipeline[0]

    # optimize test dataset
    final_results = {}
    for subj in test:
        final_results[subj.index["mesa_id"][0]] = sc.score(optimi_pipeline.optimized_pipeline_, subj)

    final_results = pd.DataFrame(final_results)
    final_results.index.name = "metric"

    final_results.to_csv(
        Path(__file__)
        .parents[1]
        .joinpath("exports/results_per_algorithm/heuristic_algorithms/" + algorithm + "_" + dataset_name + ".csv"),
        index=True,
    )
