from heuristic_pipeline_helper import cv_optimization
from sklearn.model_selection import ParameterGrid

from sleep_analysis.classification.heuristic_algorithms.scale_pipeline import ScalePipeline
from sleep_analysis.classification.utils.data_loading import load_dataset

# dataset_name: currently only MESA_Sleep supported
dataset_name = "MESA_Sleep"
# classification type: only binary supported as this is a heuristic algorithm
classification = "binary"
small = True

# load dataset and group labels
dataset, group_labels = load_dataset(dataset_name, small=small)

# algorithm to optimize
algorithm = "cole_kripke"

# set parameter search space
parameters = ParameterGrid({"scale_value": [x / 100000.0 for x in range(5, 30, 5)], "rescore_data": [True, False]})

# create pipeline
pipe = ScalePipeline(algorithm=algorithm, dataset_name=dataset_name, classification_type=classification)


# cross validation of GridSearch
cv_optimization(pipe=pipe, parameters=parameters, dataset=dataset, algorithm=algorithm, dataset_name=dataset_name)
