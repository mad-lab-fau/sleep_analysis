import numpy as np
import yasa


def plot_sleep_stages_with_artefacts(df_sleep_stage, ax=None, fill_color="gainsboro"):
    df_sleep_stage = df_sleep_stage.replace({"A": -2, "Artefakt": -1, "Wach": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4})
    stage_array = np.asarray(df_sleep_stage)

    yasa.plot_hypnogram(stage_array, ax=ax, fill_color=fill_color)


def plot_sleep_stages_without_artefacts(df_sleep_stage, ax=None, fill_color="gainsboro"):

    df_sleep_stage = df_sleep_stage.replace({"A": -2, "Artefakt": -1, "Wach": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4})
    df_sleep_stage = df_sleep_stage[df_sleep_stage != -1]

    stage_array = np.asarray(df_sleep_stage)

    yasa.plot_hypnogram(stage_array, ax=ax, fill_color=fill_color)
