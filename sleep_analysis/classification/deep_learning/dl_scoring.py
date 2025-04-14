import datetime
import warnings

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from biopsykit.sleep.sleep_endpoints import compute_sleep_endpoints
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from tpcp.validate import no_agg

from sleep_analysis.classification.utils.scoring import compute_bed_interval_from_datapoint
from sleep_analysis.datasets.mesadataset import MesaDataset
from sleep_analysis.datasets.d04_main_dataset_control import D04MainStudy

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def dl_score(prediction, ground_truth, classification_type="binary", subject_id=None):
    if classification_type == "binary":
        return dl_score_binary(prediction, ground_truth, subject_id=subject_id)
    else:
        return dl_score_multiclass(prediction, ground_truth, classification_type, subject_id=subject_id)


def dl_score_binary(prediction, ground_truth, subject_id=None):
    prediction = _sanitize_sleep_wake_df(prediction)

    print("calculation of confusion matrix", "binary classification", flush=True)
    conf_matrix = confusion_matrix(ground_truth, prediction)

    scoring = {
        "accuracy": sk_metrics.accuracy_score(ground_truth, prediction),
        "precision": sk_metrics.precision_score(ground_truth, prediction, zero_division=0),
        "recall": sk_metrics.recall_score(ground_truth, prediction, zero_division=0),
        "f1": sk_metrics.f1_score(ground_truth, prediction, zero_division=0),
        "kappa": calculate_cohens_kappa(ground_truth, prediction),
        "specificity": calculate_specificity(ground_truth, prediction),
        "mcc": matthews_corrcoef(ground_truth, prediction),
        "confusion_matrix": no_agg(conf_matrix),
    }

    # if subject_id is not None:
    #    dataset = MesaDataset()
    #    datapoint = dataset.get_subset(mesa_id=[subject_id])
    #    bed_interval = compute_bed_interval_from_datapoint(datapoint)

    #    sleep_endpoints = compute_sleep_endpoints(pd.DataFrame(prediction, columns=["sleep_wake"]), bed_interval)

    #    if not sleep_endpoints:
    #        sleep_endpoints = _empty_sleep_metrics()

    #    scoring.update(sleep_endpoints)
    #    list(map(scoring.pop, ["date", "wake_bouts", "sleep_bouts", "number_wake_bouts"]))

    return scoring


def dl_score_multiclass(prediction, ground_truth, classification_type, subject_id=None):
    ground_truth = ground_truth[["sleep_stage"]]
    # convert ground truth to integer
    ground_truth = ground_truth.astype(int)

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
    else:
        raise ValueError(f"Invalid classification type: {classification_type}")

    # print("calculation of confusion matrix ... ", "multi-class classification ... labels: ", labels, flush=True)
    # here: access to classification + ground truth --> scoring possible
    conf_matrix = confusion_matrix(y_true=np.array(ground_truth["sleep_stage"]), y_pred=prediction, labels=labels)
    # print(conf_matrix)
    scoring = {
        "accuracy": sk_metrics.accuracy_score(ground_truth, prediction),
        "precision": sk_metrics.precision_score(ground_truth, prediction, zero_division=0, average="weighted"),
        "recall": sk_metrics.recall_score(ground_truth, prediction, zero_division=0, average="weighted"),
        "f1": sk_metrics.f1_score(ground_truth, prediction, zero_division=0, average="weighted"),
        "kappa": sk_metrics.cohen_kappa_score(
            ground_truth,
            prediction,
        ),
        "specificity": dl_multiclass_specificity(ground_truth, prediction, labels=labels, average="weighted"),
        "mcc": matthews_corrcoef(ground_truth, prediction),
        "confusion_matrix": no_agg(conf_matrix),
    }

    # if subject_id is not None:
    #    dataset = MesaDataset()
    #    datapoint = dataset.get_subset(mesa_id=[subject_id])
    #    bed_interval = compute_bed_interval_from_datapoint(datapoint)
    #    prediction[prediction != 0] = 1

    #    sleep_endpoints = compute_sleep_endpoints(pd.DataFrame(prediction, columns=["sleep_wake"]), bed_interval)

    #    if not sleep_endpoints:
    #        sleep_endpoints = _empty_sleep_metrics()

    #    scoring.update(sleep_endpoints)
    #    list(map(scoring.pop, ["date", "wake_bouts", "sleep_bouts", "number_wake_bouts"]))

    return scoring


def dl_multiclass_specificity(y_true, y_pred, labels, average="weighted"):
    if average == "macro":
        raise NotImplementedError("Not implemented yet")

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

        value = np.nan_to_num(tn / (tn + fp))
        specificity.append(value)

    specificity = np.sum(np.array(specificity) * np.array(weights))

    return specificity


def calculate_specificity(ground_truth, prediction):
    tn, fp, fn, tp = tuple(confusion_matrix(ground_truth, prediction, labels=[False, True]).ravel())

    if tn == 0 and fp == 0:
        return 0.0
    else:
        return tn / (tn + fp)


def calculate_cohens_kappa(ground_truth, prediction):
    tp, fn, fp, tn = tuple(confusion_matrix(ground_truth, prediction, labels=[False, True]).ravel())

    ### Handling of warnings caused by confusionmatrix in sk_metrics.cohen_kappa_score
    if tn == 0 and fp == 0:
        if fn == 0:
            return 1.0
        else:
            return 0.0
    elif fn == 0 and fp == 0 and tp == 0:
        return 1.0
    else:
        return sk_metrics.cohen_kappa_score(ground_truth, prediction)


def tensor_to_performance(y_true, y_pred, classification_type="binary"):
    y_pred = y_pred.cpu()
    y_pred = y_pred.detach().numpy()
    y_batch_val = y_true[:, 0].cpu()

    y_batch_val = pd.DataFrame(y_batch_val.detach().numpy(), columns=["sleep_stage"])

    if classification_type == "binary":
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)

    return dl_score(y_pred, y_batch_val, classification_type)


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


def _sanitize_sleep_wake_df(data: pd.DataFrame):
    # check if data is a dataframe
    if isinstance(data, pd.DataFrame):
        # change column name to sleep_wake
        return data.rename(columns={data.columns[0]: "sleep_wake"})
    # check if data is numpy array
    elif isinstance(data, np.ndarray):
        # convert to dataframe with column name sleep_wake
        return pd.DataFrame(data, columns=["sleep_wake"])
