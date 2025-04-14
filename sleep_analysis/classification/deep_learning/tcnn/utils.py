def get_num_input(modality, dataset_name):
    if dataset_name == "MESA_Sleep":
        if modality == "ACT":
            num_inputs = 1
        elif modality == "ACT_HRV":
            num_inputs = 9
        elif modality == "HRV":
            num_inputs = 8
        elif modality == "ACC_RRV":
            num_inputs = 5
        elif modality == "RRV_HRV":
            num_inputs = 12
        elif modality == "all":
            num_inputs = 13
        elif modality == "RRV":
            num_inputs = 4
        else:
            raise ValueError("Modality not avaliable")
    else:
        raise ValueError("dataset_name not avaliable")

    return num_inputs
