from sleep_analysis.datasets.mesadataset import MesaDataset


def load_dataset(dataset_name, small=False):
    if dataset_name == "MESA_Sleep":
        train, test = load_train_test_set(dataset_name, small=small)
        dataset = (train, test)
        group_labels = None

    else:
        raise ValueError("dataset_name not known")

    return dataset, group_labels


def load_train_test_set(dataset_name, small=False):
    if dataset_name == "MESA_Sleep":
        if small:
            dataset = MesaDataset()[0:25]
        else:
            dataset = MesaDataset()
        train, test = dataset.get_random_split(dataset)

    return train, test
