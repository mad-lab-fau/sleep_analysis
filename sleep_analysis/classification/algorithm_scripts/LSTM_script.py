from sleep_analysis.classification.deep_learning.lstm.lstm_optuna import LSTM_Optuna
from sleep_analysis.classification.deep_learning.utils import load_dataset

"""
change list of modalities to select the data modality to train with - options:  ACT, HRV, RRV, EDR
change dataset_name to change between the different dataset_names - options: MESA_Sleep
"""

# modality MUST be a list from either ACT, HRV, RRV, or EDR
modality = ["ACT", "HRV", "RRV"]

# classification type: can be either binary, 3stage, 4stage or 5stage
classification = "3stage"

# dataset_name: currently only MESA_Sleep and Radar supported
dataset_name = "Radar"
small = False
retrain = True


print("Run with following parameters:")
print("modality " + " ,".join(modality))
print("classification: " + classification)
print("dataset_name: " + dataset_name)


dataset = load_dataset(dataset_name, small=small)

lstm_optuna = LSTM_Optuna(modality=modality, dataset_name=dataset_name, seed=20, classification_type=classification, retrain = retrain)
lstm_optuna.optimize(dataset=dataset)
