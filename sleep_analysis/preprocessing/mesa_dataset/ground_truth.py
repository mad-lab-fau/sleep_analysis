"""Functions to convert ground truth labels to respective Experiments."""
import numpy as np
import pandas as pd


def sleep_stage_convert_binary(df_psg):
    """Convert sleep stages to binary sleep/wake."""
    psg = df_psg[["sleep"]]

    psg = np.asarray(psg)
    psg[psg == "Wake|0"] = 0  # wake = 0
    psg[psg == "Unscored|9"] = np.nan  # drop unscored epochs later

    psg[psg != 0] = 1  # sleep = 1

    # workaround to prevent "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame."
    df_psg = pd.DataFrame(psg, columns=["sleep"]).dropna()

    return df_psg
