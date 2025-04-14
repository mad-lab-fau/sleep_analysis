from sleep_analysis.datasets.d04_main_dataset_control import D04MainStudy
from sleep_analysis.datasets.mesadataset import MesaDataset
from sleep_analysis.datasets.helper import get_random_split


def load_dataset(dataset_name, small=False):

    train, test = load_train_test_set(dataset_name, small=small)
    dataset = (train, test)
    group_labels = None

    return dataset, group_labels


def load_train_test_set(dataset_name, small=False):
    if dataset_name == "MESA_Sleep":
        if small:
            dataset = MesaDataset()[0:25]
        else:
            dataset = MesaDataset()
        train, test = get_random_split(dataset)

    if dataset_name == "Radar":
        if small:
            dataset = D04MainStudy(exclusion_criteria=["EEG"])[0:10]
        else:
            dataset = D04MainStudy()

        train, test = get_random_split(dataset)

    return train, test
