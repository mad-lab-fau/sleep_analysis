import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from biopsykit.sleep.sleep_endpoints import compute_sleep_endpoints
from biopsykit.sleep.sleep_processing_pipeline.sleep_processing_pipeline import *
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from tpcp.validate import NoAgg

from sleep_analysis.datasets.mesadataset import MesaDataset

with open(Path(__file__).parents[3].joinpath("study_data.json")) as f:
    path_dict = json.load(f)
    path = Path(path_dict["mesa_path"])


def compute_bed_interval_from_datapoint(datapoint: MesaDataset):
    """
    Compute bed-interval - Aka the time spend in bed. This information is necessary to compute several sleep statistics such as Sleep Onselt Latency (SOL) or Sleep Efficiency (SE)
    :param datapoint: Sleep data of one participant
    returns: 2-item list of bed interval of one participant (as epochs)
    """

    # load meta data file from mesa dataset
    data_documentation = pd.read_csv(path.joinpath("datasets/mesa-sleep-dataset-0.5.0.csv"))

    # extract relevant columns:
    # mesaid = Unique identifier of participant
    # lighoff5 = Time participant went to bed
    # time_bed5 = Total time in minutes participant stayed in bed
    data = data_documentation[["mesaid", "stloutp5", "stlonp5", "time_bed5"]]
    data = data.loc[data["mesaid"] == int(datapoint.index["mesa_id"][0])]

    # convert into datetime
    data["stloutp5"] = pd.to_datetime(data["stloutp5"], format="%H:%M:%S").round("30s").dt.time
    data["stlonp5"] = pd.to_datetime(data["stlonp5"], format="%H:%M:%S").round("30s").dt.time

    time = pd.to_datetime(datapoint.time["linetime"], format="%H:%M:%S").dt.time
    bed_interval = []
    try:
        bed_interval.append(time.loc[time == data["stloutp5"].values[0]].index[0])
        bed_interval.append(time.loc[time == data["stlonp5"].values[0]].index[0])
        # bed_interval.append(int(bed_interval[0] + data["time_bed5"].values * 2))  # *2 for 30s interval!)
    except:
        bed_interval = [0, time.size - 1]

    assert bed_interval[1] > bed_interval[0], "Bed interval end must be larger than bed interval start"
    return bed_interval


def score(pipeline, datapoint: MesaDataset):
    pipeline.safe_run(datapoint)

    if pipeline.classification_type == "binary":
        return binary_score(pipeline, datapoint)
    else:
        return multiclass_score(pipeline, datapoint, pipeline.classification_type)


def binary_score(pipeline, datapoint: MesaDataset):
    """
    Scoring function to compute metrics of classification performance. This performance measures are based on the confusion matrix.
    :param pipeline: Pipeline of algorithm to be scored
    :param datapoint: Datapoint of participant(s)
    returns: measures of classification performance of Datapoints
    """

    ground_truth = datapoint.ground_truth[["sleep"]]

    if pipeline.epoch_length == 60:
        ground_truth = ground_truth[::2]

    # here:access to classification + ground truth --> scoring possible
    tn, fp, fn, tp = confusion_matrix(ground_truth, pipeline.classification_, labels=[False, True]).ravel()
    conf_matrix = confusion_matrix(ground_truth, pipeline.classification_)

    scoring = {
        "accuracy": sk_metrics.accuracy_score(ground_truth, pipeline.classification_),
        "precision": sk_metrics.precision_score(ground_truth, pipeline.classification_, zero_division=0),
        "recall": sk_metrics.recall_score(ground_truth, pipeline.classification_, zero_division=0),
        "f1": sk_metrics.f1_score(ground_truth, pipeline.classification_, zero_division=0),
        "kappa": sk_metrics.cohen_kappa_score(ground_truth, pipeline.classification_),
        "specificity": tn / (tn + fp),
        "mcc": matthews_corrcoef(ground_truth, pipeline.classification_),
        "confusion_matrix": NoAgg(conf_matrix),
    }

    bed_interval = compute_bed_interval_from_datapoint(datapoint)
    if (
        pipeline.algorithm == "sadeh"
        or pipeline.algorithm == "cole_kripke_alternative"
        or pipeline.algorithm == "webster"
    ):
        bed_interval = [int(bed_interval[0] / 2), int(bed_interval[1] / 2)]
        sleep_endpoints = compute_sleep_endpoints(
            pd.DataFrame(pipeline.classification_, columns=["sleep_wake"]), bed_interval
        )
        sleep_endpoints.update((x, y * 2) for x, y in sleep_endpoints.items())
        sleep_endpoints["sleep_efficiency"] /= 2
    else:
        sleep_endpoints = compute_sleep_endpoints(
            pd.DataFrame(pipeline.classification_, columns=["sleep_wake"]), bed_interval
        )

    if not sleep_endpoints:
        sleep_endpoints = _empty_sleep_metrics()

    scoring.update(sleep_endpoints)
    list(map(scoring.pop, ["date", "wake_bouts", "sleep_bouts", "number_wake_bouts"]))

    return scoring


def multiclass_score(pipeline, datapoint: MesaDataset, classification_type: str):
    """ Scoring function to compute metrics of multiclass classification performance. This performance measures are based on the confusion matrix."""
    ground_truth = datapoint.ground_truth[[classification_type]]

    # set labels depending on classification type. This is necessary to compute the confusion matrix.
    labels = []
    if classification_type == "5stage":
        # labels = "wake", "N1", "N2", "N3", "REM"
        labels = [0, 1, 2, 3, 4]
    elif classification_type == "4stage":
        # labels = "wake", "Light", "Deep", "REM"
        labels = [0, 1, 2, 3]
    elif classification_type == "3stage":
        # labels = "wake", "NREM", "REM"
        labels = [0, 1, 2]

    # Compute confusion matrix
    conf_matrix = confusion_matrix(ground_truth, pipeline.classification_, labels=labels)

    scoring = {
        "accuracy": sk_metrics.accuracy_score(ground_truth, pipeline.classification_),
        "precision": sk_metrics.precision_score(
            ground_truth, pipeline.classification_, zero_division=0, average="weighted"
        ),
        "recall": sk_metrics.recall_score(ground_truth, pipeline.classification_, zero_division=0, average="weighted"),
        "f1": sk_metrics.f1_score(ground_truth, pipeline.classification_, zero_division=0, average="weighted"),
        "kappa": sk_metrics.cohen_kappa_score(ground_truth, pipeline.classification_,),
        "specificity": multiclass_specificity(
            ground_truth, pipeline.classification_, labels=labels, average="weighted"
        ),
        "mcc": matthews_corrcoef(ground_truth, pipeline.classification_),
        "confusion_matrix": NoAgg(conf_matrix),
    }

    # Compute sleep metrics
    bed_interval = compute_bed_interval_from_datapoint(datapoint)
    pipeline.classification_[pipeline.classification_ != 0] = 1
    sleep_endpoints = compute_sleep_endpoints(
        pd.DataFrame(pipeline.classification_, columns=["sleep_wake"]), bed_interval
    )

    if not sleep_endpoints:
        sleep_endpoints = _empty_sleep_metrics()

    scoring.update(sleep_endpoints)
    list(map(scoring.pop, ["date", "wake_bouts", "sleep_bouts", "number_wake_bouts"]))
    return scoring


def multiclass_specificity(y_true, y_pred, labels, average="weighted"):
    """ Calculates the specificity for multiclass classification problems"""

    if average == "macro":
        raise NotImplementedError("Not implemented yet")
    elif average == "weighted":

        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
        specificity = []
        weights = []
        for l, label in zip(range(conf_matrix.shape[0]), labels):
            tp = conf_matrix[l][l]
            tn = np.sum(conf_matrix) - np.sum(conf_matrix[:, l]) - np.sum(conf_matrix[l, :]) + np.sum(conf_matrix[l][l])
            fp = np.sum(conf_matrix[l, :]) - conf_matrix[l][l]
            fn = np.sum(conf_matrix[:, l]) - conf_matrix[l][l]

            weight = np.sum(y_true == label)[0] / len(y_true)
            weights.append(weight)

            value = tn / (tn + fp)
            specificity.append(value)

        specificity = np.sum(np.array(specificity) * np.array(weights))

    else:
        raise ValueError("Unknown average method")

    return specificity


def apply_rescore_to_ml(prediction, ground_truth):
    """
    Helper function to apply Webster rescoring methods to ML-classification
    :param prediction
    :param ground_truth
    :return
    """
    from biopsykit.sleep.sleep_wake_detection.utils import rescore

    prediction = rescore(np.asarray(prediction))
    prediction = pd.DataFrame(prediction, columns=["sleep_wake"])
    ground_truth = ground_truth
    tn, fp, fn, tp = confusion_matrix(ground_truth, prediction).ravel()

    scoring = {
        "accuracy": sk_metrics.accuracy_score(ground_truth, prediction),
        "precision": sk_metrics.precision_score(ground_truth, prediction, zero_division=0),
        "recall": sk_metrics.recall_score(ground_truth, prediction, zero_division=0),
        "f1": sk_metrics.f1_score(ground_truth, prediction, zero_division=0),
        "kappa": sk_metrics.cohen_kappa_score(ground_truth, prediction),
        "specificity": tn / (tn + fp),
        "mcc": matthews_corrcoef(ground_truth, prediction),
    }

    return scoring


def _empty_sleep_metrics():
    return {
        "date": datetime.datetime.now(),
        "sleep_onset": 0,
        "wake_onset": 0,
        "total_sleep_duration": 0,
        "net_sleep_duration": 0,
        "bed_interval_start": 0,
        "bed_interval_end": 0,
        "sleep_efficiency": 0,
        "sleep_onset_latency": 0,
        "getup_latency": 0,
        "wake_after_sleep_onset": 0,
        "sleep_bouts": [],
        "wake_bouts": [],
        "number_wake_bouts": 0,
    }
