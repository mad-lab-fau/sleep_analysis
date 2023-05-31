from abc import abstractmethod

import pandas as pd
from tpcp import Algorithm, Parameter, make_action_safe
from typing import Optional, Union, List, Sequence


class BaseExtraction(Algorithm):
    """
    Base Class defining Interface for other Algorithm Classes
    Takes:  Pandas Dataframe containing the ECG-Signal
            float number representing the sampling rate
    Result: Saves resulting EDR Signal as attribute in respiratory_signal
    """

    _action_methods = "extract"
    # Results
    respiratory_signal: pd.DataFrame = None

    # __init__ should be subclass specific
    def __int__(self):
        self.respiratory_signal: pd.DataFrame = None

    # Interface Method
    @make_action_safe
    @abstractmethod
    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        pass

    def normalize(self, respiration_signal: pd.DataFrame):

        normalized_respiration_signal = (
            respiration_signal - respiration_signal.mean()
        ) / respiration_signal.std()

        scaled_respiration_signal = (
            normalized_respiration_signal - normalized_respiration_signal.min()
        ) / (normalized_respiration_signal.max() - normalized_respiration_signal.min())

        return scaled_respiration_signal
