from sklearn.model_selection import ParameterGrid

from sleep_analysis.classification.ml_algorithms.ml_pipeline_helper import hold_out_optimization
from sleep_analysis.classification.ml_algorithms.mlp import MLPPipeline
from sleep_analysis.classification.utils.data_loading import load_dataset

algorithm = "mlp"

"""
change list of modalities to select the data modality to train with - options:  ACT, HRV, RRV, EDR
change dataset_name to change between the different datasets - options: MESA_Sleep
"""

# modality MUST be a list from either ACT, HRV, RRV, or EDR
modality = ["ACT", "HRV", "RRV"]
# classification type: can be either binary, 3stage, 4stage or 5stage
classification = "3stage"
# dataset_name: currently only MESA_Sleep supported
dataset_name = "MESA_Sleep"
small = False

print("Run with following parameters:")
print("modality " + " ,".join(modality))
print("classification: " + classification)
print("dataset_name: " + dataset_name)
print("small: " + str(small))

dataset, group_labels = load_dataset(dataset_name, small=small)

pipe = MLPPipeline(modality=modality, classification_type=classification)

# parameter search space for Grid Search
parameters = ParameterGrid(
    {
        "hidden_layer_sizes": [(50, 50, 50), (50, 100, 50), (100,)],
        "activation": ["tanh", "relu"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.05],
        "learning_rate": ["constant", "adaptive"],
    }
)

hold_out_optimization(
    pipe=pipe,
    parameters=parameters,
    algorithm=algorithm,
    dataset=dataset,
    dataset_name=dataset_name,
    modality=modality,
    classification_type=classification,
    n_jobs=-1,
    pre_dispatch="n_jobs/3.5",
)
