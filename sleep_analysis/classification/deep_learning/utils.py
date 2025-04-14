from sleep_analysis.datasets.mesadataset import MesaDataset
from sleep_analysis.datasets.d04_main_dataset_control import D04MainStudy


def get_num_classes(classification_type):
    """
    Determine number of classes dependent on classification type
    """
    if classification_type == "binary":
        num_classes = 1
    elif classification_type == "3stage":
        num_classes = 3
    elif classification_type == "5stage":
        num_classes = 5
    elif classification_type == "4stage":
        num_classes = 4
    else:
        raise AttributeError("classification_type MUST be either binary, 3stage, 4stage or 5stage")

    return num_classes


def get_num_input(modality):
    """
    Determine number of inputs dependent on data source and input modality
    """

    num_inputs = 0
    if "ACT" in modality:
        num_inputs += 1
    if "HRV" in modality:
        num_inputs += 8
    if "RRV" in modality:
        num_inputs += 4
    if "EDR" in modality:
        num_inputs += 4

    mod_set = {"HRV", "ACT", "RRV", "EDR"}
    if not all({mod}.issubset(mod_set) for mod in modality):
        raise AttributeError("modality MUST be list of either HRV, ACT, RRV, EDR")

    return num_inputs


def load_dataset(dataset_name, small):
    """
    Load dataset for DL algorithms
    """
    if dataset_name == "MESA_Sleep":
        if small is True:
            dataset = MesaDataset()[0:20]
        else:
            dataset = MesaDataset()
        return dataset
    elif dataset_name == "Radar":
        if small is True:
            dataset = D04MainStudy(exclusion_criteria=["EEG"])[0:10]
        else:
            dataset = D04MainStudy(exclusion_criteria=["EEG"])
        return dataset
    else:
        raise ValueError("dataset_name not known")
