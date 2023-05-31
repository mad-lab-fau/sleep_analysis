import pandas as pd
from biopsykit.signals.ecg import EcgProcessor

from sleep_analysis.preprocessing.mesa_dataset.edr.base_extraction import BaseExtraction

"""
 Feature based EDR algorithms XB1 to XB3 from https://doi.org/10.1088/1361-6579/aa670e
 Implementations of the BioPsyKit were used 
"""


class ExtractionCharlton(BaseExtraction):
    """
    extraction Algorithm XB1 from https://doi.org/10.1088/1361-6579/aa670e
    Returns:
    Mean amplitude of troughs and proceeding peaks (Charlton et al. 2016a)
    """

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        # Processor
        ecg_signal = ecg_signal.rename(columns={"CH1": "ecg"})
        processor = EcgProcessor(data=ecg_signal, sampling_rate=sampling_rate)
        processor.ecg_process(outlier_correction=None)
        # Generate respiration respiratory_signal
        self.respiratory_signal = EcgProcessor.ecg_estimate_rsp(
            ecg_processor=processor, key="Data", edr_type="peak_trough_mean",
        )
        # Normalize respiration respiratory_signal
        self.respiratory_signal = self.normalize(self.respiratory_signal)
        return self


class ExtractionKarlen(BaseExtraction):
    """Extraction Algorithm XB2 from https://doi.org/10.1088/1361-6579/aa670e"""

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        # Processor
        ecg_signal = ecg_signal.rename(columns={"CH1": "ecg"})
        processor = EcgProcessor(data=ecg_signal, sampling_rate=sampling_rate)
        processor.ecg_process(outlier_correction=None)

        self.respiratory_signal = EcgProcessor.ecg_estimate_rsp(
            ecg_processor=processor, key="Data", edr_type="peak_trough_diff",
        )

        self.respiratory_signal = self.normalize(self.respiratory_signal)
        return self


class ExtractionOrphandiou(BaseExtraction):
    """Extraction Algorithm XB3 from https://doi.org/10.1088/1361-6579/aa670e"""

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        # Processor
        ecg_signal = ecg_signal.rename(columns={"CH1": "ecg"})
        processor = EcgProcessor(data=ecg_signal, sampling_rate=sampling_rate)
        processor.ecg_process(outlier_correction=None)
        # Generate respiration respiratory_signal
        self.respiratory_signal = EcgProcessor.ecg_estimate_rsp(
            ecg_processor=processor, key="Data", edr_type="peak_peak_interval",
        )
        # Normalize respiration respiratory_signal
        self.respiratory_signal = self.normalize(self.respiratory_signal)
        return self
