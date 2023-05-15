from sklearn.model_selection import ParameterGrid

from sleep_analysis.classification.ml_algorithms.ml_pipeline_helper import hold_out_optimization
from sleep_analysis.classification.ml_algorithms.svm import SVMPipeline
from sleep_analysis.classification.utils.data_loading import load_dataset

algorithm = "svm"

"""
change list of modalities to select the data modality to train with - options:  ACT, HRV, RRV, EDR
change dataset_name to change between the different datasets - options: MESA_Sleep
"""

# modality MUST be a list from either ACT, HRV, RRV, or EDR
modality = ["ACT", "HRV", "EDR"]
# classification type: can be either binary, 3stage, 4stage or 5stage
classification = "binary"
# dataset_name: currently only MESA_Sleep supported
dataset_name = "MESA_Sleep"
small = False

print("Run with following parameters:")
print("modality " + " ,".join(modality))
print("classification: " + classification)
print("dataset_name: " + dataset_name)
print("small: " + str(small))

dataset, group_labels = load_dataset(dataset_name, small=small)

pipe = SVMPipeline(modality=modality, classification_type=classification)

# Parameter search space for Grid Search
parameters = ParameterGrid(
    {
        "class_weight": [None],  # ,'balanced'],
        "loss": ["hinge", "log", "modified_huber"],
        "penalty": ["l1", "l2"],
        "alpha": [0.0001, 0.001, 0.01],
        "l1_ratio": [0.1],
        "fit_intercept": [True],  # always True always better
        "warm_start": [False],
        "epsilon": [0.1],
        "learning_rate": ["optimal", "adaptive"],
        "eta0": [0.5],
        "power_t": [0.5],
        "max_iter": [1000, 1500],
        "early_stopping": [True, False],
        "k": [30, 150, 270, 300, 460],
    }
)

hold_out_optimization(
    pipe=pipe,
    parameters=parameters,
    algorithm=algorithm,
    dataset=dataset,
    dataset_name=dataset_name,
    modality=modality,
    n_jobs=-1,
    pre_dispatch="n_jobs/1.8",
    classification_type=classification,
)
