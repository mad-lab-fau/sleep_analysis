import neurokit2 as nk
import numpy as np
import pandas as pd


def clean_ecg(ecg_signal, sampling_rate=256):
    """
    Clean ECG signal - Adopted from neurokit2 but with different parameters due to high T-waves
    in some of the participants in the radar bed dataset (EmpkinS D04 main study)
    """
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="biosppy", show=False)
    return ecg_cleaned


def ecg_peaks(cleaned_ecg, sampling_rate=256):
    """
    Detect peaks in ECG signal - Adopted from neurokit2 but with different parameters due to high T-waves
    in some of the participants in the radar bed dataset (EmpkinS D04 main study)
    """
    # Detect R-peaks
    instant_peaks, info = nk.ecg_peaks(
        ecg_cleaned=cleaned_ecg,
        sampling_rate=sampling_rate,
        method="neurokit",
        correct_artifacts=True,
    )

    return instant_peaks, info


def calculate_hr(instant_peaks, sampling_rate=256, desired_length=None):
    """
    Calculate heart rate - Adopted from neurokit2 but with different parameters due to high T-waves
    in some of the participants in the radar bed dataset (EmpkinS D04 main study)
    """
    rate = nk.signal_rate(instant_peaks, sampling_rate=sampling_rate, desired_length=desired_length)
    return rate


def ecg_process_modified(ecg_signal, sampling_rate=256):
    """
    Preprocess ECG signal using methods from neurokit2.
    """

    # clean ECG signal
    ecg_cleaned = clean_ecg(ecg_signal, sampling_rate=sampling_rate)

    # detect peaks
    instant_peaks, info = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)

    # calculate heart rate
    heart_rate = calculate_hr(instant_peaks, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned))

    # merge signals in a DataFrame
    signals = pd.DataFrame(
        {
            "ECG_Raw": ecg_signal,
            "ECG_Clean": ecg_cleaned,
            "ECG_Rate": heart_rate,
        }
    )

    # Delineate QRS complex
    delineate_signal, delineate_info = nk.ecg_delineate(
        ecg_cleaned=ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )
    info.update(delineate_info)  # Merge waves indices dict with info dict

    # Determine cardiac phases
    cardiac_phase = nk.ecg_phase(
        ecg_cleaned=ecg_cleaned,
        rpeaks=info["ECG_R_Peaks"],
        delineate_info=delineate_info,
    )

    # Add additional information to signals DataFrame
    signals = pd.concat([signals, instant_peaks, delineate_signal, cardiac_phase], axis=1)

    return signals, info
