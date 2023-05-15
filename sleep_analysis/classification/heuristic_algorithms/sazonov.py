from heuristic_pipeline_helper import cv_optimization, cv_optmization_group
from sklearn.model_selection import ParameterGrid

from sleep_analysis.classification.heuristic_algorithms.basic_pipeline import BasicPipeline
from sleep_analysis.classification.utils.data_loading import load_dataset

# dataset_name: currently only MESA_Sleep supported
dataset_name = "MESA_Sleep"
# classification type: only binary supported as this is a heuristic algorithm
classification = "binary"
small = False

# load dataset and group labels if real_world_acti
dataset, group_labels = load_dataset(dataset_name, small=small)

# algorithm to optimize
algorithm = "sazonov"

# set parameter search space
parameters = ParameterGrid({"rescore_data": [True, False]})

# create pipeline
pipe = BasicPipeline(algorithm=algorithm, dataset_name=dataset_name, classification_type=classification)


# cross validation of GridSearch
cv_optimization(pipe=pipe, parameters=parameters, dataset=dataset, algorithm=algorithm, dataset_name=dataset_name)
