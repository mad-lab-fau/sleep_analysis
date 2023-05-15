from sleep_analysis.classification.deep_learning.tcnn.tcn_optuna import TCNOptuna
from sleep_analysis.classification.deep_learning.utils import load_dataset

"""
change list of modalities to select the data modality to train with - options:  ACT, HRV, RRV, EDR
change dataset_name to change between the different dataset_names - options: MESA_Sleep
"""

# modality MUST be a list from either ACT, HRV, RRV, or EDR
modality = ["ACT", "HRV"]

# classification type: can be either binary, 3stage, 4stage or 5stage
classification = "binary"

# dataset_name: currently only MESA_Sleep supported
dataset_name = "MESA_Sleep"
small = False


print("Run with following parameters:")
print("modality " + " ,".join(modality))
print("classification: " + classification)
print("dataset_name: " + dataset_name)


dataset = load_dataset(dataset_name, small=small)


tcn_optuna = TCNOptuna(
    modality=modality, dataset_name=dataset_name, seq_len=101, seed=10, classification_type=classification
)
tcn_optuna.optimize(dataset=dataset)
